"""
CUDA/C code generator for XGBoost tree ensemble models.

Architecture:
    XGBoost Model (.ubj/.json/.bin)
        → CudaCodeGenerator (Python)
            → parse tree JSON structures
            → calibrate base_score against XGBoost predictions
            → emit tree functions as nested if/else
            → emit scoring kernel (CUDA) or scoring loop (CPU)
            → emit host API + standalone main()
        → .cu or .c source file
            → nvcc / gcc
        → compiled binary or shared library

Each tree becomes an inline device function with nested if/else branches.
NaN handling mirrors XGBoost's "missing value" routing using IEEE 754
comparison semantics (NaN comparisons return false), avoiding explicit
isnan() calls in the two common cases.

The scoring kernel assigns one CUDA thread per input sample. Each thread
traverses all trees sequentially and sums leaf values, then applies
the output transformation (sigmoid for binary:logistic).
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Extracted model metadata embedded in the generated code header."""
    num_trees: int
    num_features: int
    feature_names: list[str]
    base_score: float       # Global bias in logit space (calibrated)
    objective: str          # e.g. 'binary:logistic', 'reg:squarederror'
    max_depth: int


# ---------------------------------------------------------------------------
# Objective → output transformation mapping
# ---------------------------------------------------------------------------

_CUDA_TRANSFORMS = {
    "binary:logistic":  "1.0f / (1.0f + expf(-raw))",    # sigmoid
    "binary:logitraw":  "raw",                             # identity
    "reg:squarederror": "raw",                             # identity
    "reg:linear":       "raw",                             # identity (deprecated alias)
    "rank:pairwise":    "1.0f / (1.0f + expf(-raw))",     # sigmoid
    "rank:ndcg":        "1.0f / (1.0f + expf(-raw))",     # sigmoid
}

_CPU_TRANSFORMS = {
    "binary:logistic":  "1.0f / (1.0f + expf(-raw))",
    "binary:logitraw":  "raw",
    "reg:squarederror": "raw",
    "reg:linear":       "raw",
    "rank:pairwise":    "1.0f / (1.0f + expf(-raw))",
    "rank:ndcg":        "1.0f / (1.0f + expf(-raw))",
}


class CudaCodeGenerator:
    """
    Generates CUDA or C source code from a trained XGBoost model.

    The generated code is fully self-contained — no runtime dependencies
    beyond CUDA (for .cu) or standard C math (for .c). It includes:

      - One inline function per tree (nested if/else traversal)
      - A score_all_trees() aggregator
      - A CUDA kernel (or CPU loop) that scores N samples in parallel
      - A host-side API for memory management and kernel launch
      - A standalone main() that reads/writes binary float32 files

    Usage:
        gen = CudaCodeGenerator("models/model.ubj")
        gen.generate_cuda("generated/scoring_kernel.cu")
        gen.generate_cpu("generated/scoring_kernel.c")   # for testing without GPU
    """

    def __init__(
        self,
        model_path: str,
        feature_names: Optional[list[str]] = None,
        metadata_path: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to XGBoost model file (.ubj, .json, .bin)
            feature_names: Optional explicit feature name list
            metadata_path: Optional path to training_result.json (for feature names)
        """
        self.model_path = Path(model_path)
        self.booster = self._load_model(model_path)
        self.trees = self._extract_trees()
        self.model_info = self._extract_model_info(feature_names, metadata_path)
        self._calibrate_base_score()

    # ------------------------------------------------------------------
    # Model loading & introspection
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str) -> xgb.Booster:
        booster = xgb.Booster()
        booster.load_model(model_path)
        return booster

    def _extract_trees(self) -> list[dict]:
        raw = self.booster.get_dump(dump_format="json")
        return [json.loads(t) for t in raw]

    def _extract_model_info(
        self,
        feature_names: Optional[list[str]],
        metadata_path: Optional[str],
    ) -> ModelInfo:
        config = json.loads(self.booster.save_config())
        learner = config["learner"]

        num_features = int(learner["learner_model_param"]["num_feature"])

        # Objective
        obj_cfg = learner.get("objective", {})
        if isinstance(obj_cfg, dict):
            objective = obj_cfg.get("name", "binary:logistic")
        else:
            objective = str(obj_cfg)

        # Max depth (best-effort extraction from config)
        max_depth = 6
        try:
            gbtree = learner["gradient_booster"]
            # XGBoost 2.x config layout
            updater_list = gbtree.get("gbtree_train_param", {})
            if "updater" in gbtree:
                updater = gbtree["updater"]
                if isinstance(updater, list) and len(updater) > 0:
                    tree_param = updater[0].get("train_param", {})
                    max_depth = int(tree_param.get("max_depth", 6))
                elif isinstance(updater, dict):
                    for key in updater:
                        tp = updater[key].get("train_param", {})
                        if "max_depth" in tp:
                            max_depth = int(tp["max_depth"])
                            break
        except (KeyError, TypeError, IndexError):
            pass

        # Feature names — explicit > metadata file > default f0..fN
        if feature_names:
            names = feature_names
        elif metadata_path:
            with open(metadata_path) as f:
                meta = json.load(f)
            names = meta.get("feature_names", [f"f{i}" for i in range(num_features)])
        else:
            names = [f"f{i}" for i in range(num_features)]

        return ModelInfo(
            num_trees=len(self.trees),
            num_features=num_features,
            feature_names=names,
            base_score=0.0,  # calibrated below
            objective=objective,
            max_depth=max_depth,
        )

    def _calibrate_base_score(self):
        """
        Determine the base_score offset by comparing the manual sum of tree
        leaf values against XGBoost's raw margin output on a probe sample.

        This makes the generator robust across XGBoost versions (which store
        base_score differently in the config) and avoids any guesswork.
        """
        rng = np.random.RandomState(42)
        x = rng.randn(1, self.model_info.num_features).astype(np.float32)

        dmat = xgb.DMatrix(x)
        raw_margin = float(self.booster.predict(dmat, output_margin=True)[0])

        manual_sum = sum(self._evaluate_tree(t, x[0]) for t in self.trees)

        self.model_info.base_score = raw_margin - manual_sum
        logger.info(
            "Calibrated base_score=%.8f (raw_margin=%.8f, tree_sum=%.8f)",
            self.model_info.base_score, raw_margin, manual_sum,
        )

    def _evaluate_tree(self, node: dict, features: np.ndarray) -> float:
        """Evaluate a single tree on a feature vector (used for calibration)."""
        if "leaf" in node:
            return node["leaf"]

        feat_idx = self._parse_feature_index(node["split"])
        threshold = node["split_condition"]
        yes_id = node["yes"]
        no_id = node["no"]
        missing_id = node.get("missing", yes_id)

        children = {c["nodeid"]: c for c in node["children"]}
        value = features[feat_idx]

        if np.isnan(value):
            return self._evaluate_tree(children[missing_id], features)
        elif value < threshold:
            return self._evaluate_tree(children[yes_id], features)
        else:
            return self._evaluate_tree(children[no_id], features)

    def _parse_feature_index(self, split) -> int:
        """Parse a feature index from XGBoost's split field."""
        if isinstance(split, int):
            return split
        if isinstance(split, str):
            # "f0", "f12", etc.
            if split.startswith("f") and split[1:].isdigit():
                return int(split[1:])
            # Named feature — look up in feature_names
            try:
                return self.model_info.feature_names.index(split)
            except ValueError:
                pass
            # Last resort: try parsing as int
            if split.isdigit():
                return int(split)
        raise ValueError(f"Cannot parse feature index from split={split!r}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_cuda(self, output_path: str) -> str:
        """Generate a self-contained CUDA (.cu) source file."""
        code = self._generate_code(cuda=True)
        self._write(output_path, code)
        logger.info(
            "Generated CUDA: %s (%d trees, %d features)",
            output_path, self.model_info.num_trees, self.model_info.num_features,
        )
        return output_path

    def generate_cpu(self, output_path: str) -> str:
        """Generate a self-contained C (.c) source file (CPU-only, for testing)."""
        code = self._generate_code(cuda=False)
        self._write(output_path, code)
        logger.info(
            "Generated CPU C: %s (%d trees, %d features)",
            output_path, self.model_info.num_trees, self.model_info.num_features,
        )
        return output_path

    @staticmethod
    def _write(path: str, content: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

    # ------------------------------------------------------------------
    # Code emission
    # ------------------------------------------------------------------

    def _generate_code(self, cuda: bool) -> str:
        sections = [
            self._emit_header(cuda),
            self._emit_includes(cuda),
            self._emit_constants(),
        ]
        # Tree functions
        for i, tree in enumerate(self.trees):
            sections.append(self._emit_tree_function(i, tree, cuda))

        sections.append(self._emit_score_all_trees(cuda))

        if cuda:
            sections.append(self._emit_cuda_kernel())
            sections.append(self._emit_cuda_host_api())
        else:
            sections.append(self._emit_cpu_scoring())

        sections.append(self._emit_main(cuda))
        return "\n".join(sections)

    # -- header & includes --

    def _emit_header(self, cuda: bool) -> str:
        lang = "CUDA" if cuda else "C"
        names_preview = ", ".join(self.model_info.feature_names[:10])
        if len(self.model_info.feature_names) > 10:
            names_preview += ", ..."
        return (
            f"/*\n"
            f" * Auto-generated {lang} scoring kernel for XGBoost model\n"
            f" *\n"
            f" * Source model:  {self.model_path.name}\n"
            f" * Trees:         {self.model_info.num_trees}\n"
            f" * Features:      {self.model_info.num_features}"
            f"  [{names_preview}]\n"
            f" * Objective:     {self.model_info.objective}\n"
            f" * Base score:    {self.model_info.base_score:.8f}\n"
            f" * Generated:     {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
            f" *\n"
            f" * DO NOT EDIT — this file is auto-generated by cuda_codegen.\n"
            f" */\n"
        )

    @staticmethod
    def _emit_includes(cuda: bool) -> str:
        lines = [
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "#include <math.h>",
            "#include <string.h>",
        ]
        if cuda:
            lines.append("#include <cuda_runtime.h>")
        lines.append("")
        return "\n".join(lines)

    def _emit_constants(self) -> str:
        return (
            f"#define NUM_TREES    {self.model_info.num_trees}\n"
            f"#define NUM_FEATURES {self.model_info.num_features}\n"
            f"#define BASE_SCORE   {self.model_info.base_score:.8f}f\n"
        )

    # -- tree functions --

    def _emit_tree_function(self, idx: int, tree: dict, cuda: bool) -> str:
        prefix = "__device__ __forceinline__" if cuda else "static inline"
        lines = [f"\n{prefix} float tree_{idx}(const float* f) {{"]
        lines.append(self._emit_node(tree, indent=1))
        lines.append("}\n")
        return "\n".join(lines)

    def _emit_node(self, node: dict, indent: int) -> str:
        pad = "    " * indent

        if "leaf" in node:
            return f"{pad}return {node['leaf']:.8f}f;"

        feat_idx = self._parse_feature_index(node["split"])
        threshold = node["split_condition"]
        yes_id = node["yes"]
        no_id = node["no"]
        missing_id = node.get("missing", yes_id)

        children = {c["nodeid"]: c for c in node["children"]}
        yes_child = children[yes_id]
        no_child = children[no_id]

        # Feature name for readability comment
        feat_name = (
            self.model_info.feature_names[feat_idx]
            if feat_idx < len(self.model_info.feature_names)
            else f"f{feat_idx}"
        )

        lines = []

        if missing_id == yes_id:
            # NaN routes same as "yes" (f < threshold).
            # !(f >= threshold) is true for both (f < threshold) AND NaN.
            lines.append(
                f"{pad}if (!(f[{feat_idx}] >= {threshold:.8f}f)) {{ "
                f"/* {feat_name} */"
            )
            lines.append(self._emit_node(yes_child, indent + 1))
            lines.append(f"{pad}}} else {{")
            lines.append(self._emit_node(no_child, indent + 1))
            lines.append(f"{pad}}}")
        elif missing_id == no_id:
            # NaN routes same as "no" (f >= threshold).
            # (f < threshold) is false for NaN, so NaN falls into else.
            lines.append(
                f"{pad}if (f[{feat_idx}] < {threshold:.8f}f) {{ "
                f"/* {feat_name} */"
            )
            lines.append(self._emit_node(yes_child, indent + 1))
            lines.append(f"{pad}}} else {{")
            lines.append(self._emit_node(no_child, indent + 1))
            lines.append(f"{pad}}}")
        else:
            # Rare: NaN goes a third direction. Explicit isnan check required.
            missing_child = children[missing_id]
            lines.append(
                f"{pad}if (isnan(f[{feat_idx}])) {{ /* {feat_name} missing */"
            )
            lines.append(self._emit_node(missing_child, indent + 1))
            lines.append(
                f"{pad}}} else if (f[{feat_idx}] < {threshold:.8f}f) {{"
            )
            lines.append(self._emit_node(yes_child, indent + 1))
            lines.append(f"{pad}}} else {{")
            lines.append(self._emit_node(no_child, indent + 1))
            lines.append(f"{pad}}}")

        return "\n".join(lines)

    def _emit_score_all_trees(self, cuda: bool) -> str:
        prefix = "__device__" if cuda else "static"
        lines = [f"\n{prefix} float score_all_trees(const float* f) {{"]
        lines.append("    float sum = 0.0f;")
        # Unroll in groups of 8 for readability
        for i in range(0, len(self.trees), 8):
            batch = range(i, min(i + 8, len(self.trees)))
            for j in batch:
                lines.append(f"    sum += tree_{j}(f);")
        lines.append("    return sum;")
        lines.append("}\n")
        return "\n".join(lines)

    # -- CUDA kernel & host API --

    def _get_transform_expr(self, cuda: bool) -> str:
        table = _CUDA_TRANSFORMS if cuda else _CPU_TRANSFORMS
        expr = table.get(self.model_info.objective)
        if expr is None:
            logger.warning(
                "Unknown objective %r, defaulting to identity transform",
                self.model_info.objective,
            )
            expr = "raw"
        return expr

    def _emit_cuda_kernel(self) -> str:
        transform = self._get_transform_expr(cuda=True)
        return f"""
/* ==== CUDA Scoring Kernel ==== */

__global__ void score_kernel(
    const float* __restrict__ features,  /* [n_samples * NUM_FEATURES] row-major */
    float* __restrict__ scores,          /* [n_samples] */
    int n_samples
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;

    const float* f = features + idx * NUM_FEATURES;
    float raw = BASE_SCORE + score_all_trees(f);
    scores[idx] = {transform};
}}
"""

    def _emit_cuda_host_api(self) -> str:
        return """
/* ==== Host API ==== */

typedef struct {
    float* d_features;
    float* d_scores;
    int    capacity;
} ScoringContext;

ScoringContext* scoring_context_create(int max_samples) {
    ScoringContext* ctx = (ScoringContext*)malloc(sizeof(ScoringContext));
    if (!ctx) return NULL;
    ctx->capacity = max_samples;
    cudaMalloc(&ctx->d_features,
               (size_t)max_samples * NUM_FEATURES * sizeof(float));
    cudaMalloc(&ctx->d_scores,
               (size_t)max_samples * sizeof(float));
    return ctx;
}

void scoring_context_destroy(ScoringContext* ctx) {
    if (!ctx) return;
    cudaFree(ctx->d_features);
    cudaFree(ctx->d_scores);
    free(ctx);
}

/**
 * Score a batch of samples on the GPU.
 *
 * @param ctx        Reusable context (pre-allocated device memory)
 * @param h_features Host pointer: [n_samples * NUM_FEATURES] row-major float32
 * @param h_scores   Host pointer: [n_samples] output probabilities
 * @param n_samples  Number of samples to score
 * @return 0 on success, -1 on error
 */
int score_batch(
    ScoringContext* ctx,
    const float* h_features,
    float* h_scores,
    int n_samples
) {
    if (n_samples > ctx->capacity) {
        fprintf(stderr, "ERROR: n_samples (%d) exceeds capacity (%d)\\n",
                n_samples, ctx->capacity);
        return -1;
    }

    cudaMemcpy(ctx->d_features, h_features,
               (size_t)n_samples * NUM_FEATURES * sizeof(float),
               cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size  = (n_samples + block_size - 1) / block_size;
    score_kernel<<<grid_size, block_size>>>(
        ctx->d_features, ctx->d_scores, n_samples);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_scores, ctx->d_scores,
               (size_t)n_samples * sizeof(float),
               cudaMemcpyDeviceToHost);
    return 0;
}
"""

    # -- CPU scoring --

    def _emit_cpu_scoring(self) -> str:
        transform = self._get_transform_expr(cuda=False)
        return f"""
/* ==== CPU Batch Scoring ==== */

/**
 * Score a batch of samples on the CPU.
 *
 * @param features  [n_samples * NUM_FEATURES] row-major float32
 * @param scores    [n_samples] output (probabilities or raw scores)
 * @param n_samples Number of samples to score
 */
void score_batch_cpu(
    const float* features,
    float* scores,
    int n_samples
) {{
    for (int i = 0; i < n_samples; i++) {{
        const float* f = features + i * NUM_FEATURES;
        float raw = BASE_SCORE + score_all_trees(f);
        scores[i] = {transform};
    }}
}}
"""

    # -- main() --

    def _emit_main(self, cuda: bool) -> str:
        if cuda:
            score_call = (
                "    ScoringContext* ctx = scoring_context_create(n_samples);\n"
                "    if (!ctx) { fprintf(stderr, \"Failed to create context\\n\"); return 1; }\n"
                "    int rc = score_batch(ctx, features, scores, n_samples);\n"
                "    scoring_context_destroy(ctx);\n"
                "    if (rc != 0) return 1;\n"
            )
        else:
            score_call = "    score_batch_cpu(features, scores, n_samples);\n"

        return f"""
/* ==== Standalone Entry Point ==== */

int main(int argc, char** argv) {{
    if (argc < 2) {{
        fprintf(stderr,
            "Usage: %s <input.bin> [output.bin]\\n"
            "\\n"
            "  input.bin:  binary float32 array, row-major [N x %d]\\n"
            "  output.bin: binary float32 array [N] scores\\n"
            "              (omit to print first 20 scores to stdout)\\n",
            argv[0], NUM_FEATURES);
        return 1;
    }}

    /* Read input */
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) {{ perror("fopen input"); return 1; }}
    fseek(fin, 0, SEEK_END);
    long fsize = ftell(fin);
    fseek(fin, 0, SEEK_SET);

    int n_samples = (int)(fsize / ((long)NUM_FEATURES * (long)sizeof(float)));
    if (n_samples <= 0) {{
        fprintf(stderr, "Empty or invalid input (size=%ld, expected multiple of %lu)\\n",
                fsize, (unsigned long)(NUM_FEATURES * sizeof(float)));
        fclose(fin);
        return 1;
    }}
    fprintf(stderr, "Scoring %d samples (%d features each)...\\n",
            n_samples, NUM_FEATURES);

    float* features = (float*)malloc((size_t)fsize);
    if (!features) {{ perror("malloc features"); return 1; }}
    fread(features, sizeof(float), (size_t)n_samples * NUM_FEATURES, fin);
    fclose(fin);

    float* scores = (float*)malloc((size_t)n_samples * sizeof(float));
    if (!scores) {{ perror("malloc scores"); return 1; }}

    /* Score */
{score_call}
    /* Output */
    if (argc >= 3) {{
        FILE* fout = fopen(argv[2], "wb");
        if (!fout) {{ perror("fopen output"); return 1; }}
        fwrite(scores, sizeof(float), (size_t)n_samples, fout);
        fclose(fout);
        fprintf(stderr, "Wrote %d scores to %s\\n", n_samples, argv[2]);
    }} else {{
        int limit = n_samples < 20 ? n_samples : 20;
        for (int i = 0; i < limit; i++) {{
            printf("sample[%d] = %.8f\\n", i, scores[i]);
        }}
        if (n_samples > 20)
            printf("... (%d more samples)\\n", n_samples - 20);
    }}

    free(features);
    free(scores);
    return 0;
}}
"""
