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

# Metal Shading Language uses exp() not expf() (overloaded for float in <metal_stdlib>)
_METAL_TRANSFORMS = {
    "binary:logistic":  "1.0f / (1.0f + exp(-raw))",
    "binary:logitraw":  "raw",
    "reg:squarederror": "raw",
    "reg:linear":       "raw",
    "rank:pairwise":    "1.0f / (1.0f + exp(-raw))",
    "rank:ndcg":        "1.0f / (1.0f + exp(-raw))",
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

    def generate_cuda_split(self, output_path: str, user_feature_count: int) -> str:
        """
        Generate CUDA code with split user/item features.

        The generated kernel keeps item features resident on GPU and only
        transfers the user feature vector per scoring request.

        Args:
            output_path: Output .cu file
            user_feature_count: Features 0..N-1 are user, N..end are item
        """
        self._validate_split(user_feature_count)
        code = self._generate_split_code(cuda=True, user_fc=user_feature_count)
        self._write(output_path, code)
        item_fc = self.model_info.num_features - user_feature_count
        logger.info(
            "Generated split CUDA: %s (%d user + %d item features, %d trees)",
            output_path, user_feature_count, item_fc, self.model_info.num_trees,
        )
        return output_path

    def generate_cpu_split(self, output_path: str, user_feature_count: int) -> str:
        """
        Generate CPU C code with split user/item features (for testing).
        """
        self._validate_split(user_feature_count)
        code = self._generate_split_code(cuda=False, user_fc=user_feature_count)
        self._write(output_path, code)
        item_fc = self.model_info.num_features - user_feature_count
        logger.info(
            "Generated split CPU C: %s (%d user + %d item features, %d trees)",
            output_path, user_feature_count, item_fc, self.model_info.num_trees,
        )
        return output_path

    def _validate_split(self, user_fc: int):
        if user_fc <= 0 or user_fc >= self.model_info.num_features:
            raise ValueError(
                f"user_feature_count must be in [1, {self.model_info.num_features - 1}], "
                f"got {user_fc}"
            )

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

    # ==================================================================
    # Split-feature mode: user features shared, item features per-row
    # ==================================================================

    def _generate_split_code(self, cuda: bool, user_fc: int) -> str:
        item_fc = self.model_info.num_features - user_fc
        sections = [
            self._emit_split_header(cuda, user_fc, item_fc),
            self._emit_includes(cuda),
            self._emit_split_constants(user_fc, item_fc),
        ]
        # Tree functions are identical — they take (const float* f)
        for i, tree in enumerate(self.trees):
            sections.append(self._emit_tree_function(i, tree, cuda))

        sections.append(self._emit_score_all_trees(cuda))

        if cuda:
            sections.append(self._emit_split_cuda_kernel())
            sections.append(self._emit_split_cuda_topk_kernel())
            sections.append(self._emit_split_cuda_host_api())
        else:
            sections.append(self._emit_split_cpu_scoring())

        sections.append(self._emit_split_main(cuda))
        return "\n".join(sections)

    def _emit_split_header(self, cuda: bool, user_fc: int, item_fc: int) -> str:
        lang = "CUDA" if cuda else "C"
        user_names = self.model_info.feature_names[:user_fc]
        item_names = self.model_info.feature_names[user_fc:]
        user_preview = ", ".join(user_names[:8])
        if len(user_names) > 8:
            user_preview += ", ..."
        item_preview = ", ".join(item_names[:8])
        if len(item_names) > 8:
            item_preview += ", ..."
        return (
            f"/*\n"
            f" * Auto-generated {lang} split-feature scoring kernel\n"
            f" *\n"
            f" * Source model:    {self.model_path.name}\n"
            f" * Trees:           {self.model_info.num_trees}\n"
            f" * Total features:  {self.model_info.num_features}\n"
            f" *   User (0..{user_fc - 1}):   {user_fc}  [{user_preview}]\n"
            f" *   Item ({user_fc}..{self.model_info.num_features - 1}):  {item_fc}  [{item_preview}]\n"
            f" * Objective:       {self.model_info.objective}\n"
            f" * Base score:      {self.model_info.base_score:.8f}\n"
            f" * Generated:       {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
            f" *\n"
            f" * Split-feature mode: item features stay resident on GPU.\n"
            f" * Only the user vector is transferred per scoring request.\n"
            f" *\n"
            f" * DO NOT EDIT — this file is auto-generated by cuda_codegen.\n"
            f" */\n"
        )

    def _emit_split_constants(self, user_fc: int, item_fc: int) -> str:
        return (
            f"#define NUM_TREES          {self.model_info.num_trees}\n"
            f"#define NUM_FEATURES       {self.model_info.num_features}\n"
            f"#define NUM_USER_FEATURES  {user_fc}\n"
            f"#define NUM_ITEM_FEATURES  {item_fc}\n"
            f"#define BASE_SCORE         {self.model_info.base_score:.8f}f\n"
        )

    def _emit_split_cuda_kernel(self) -> str:
        transform = self._get_transform_expr(cuda=True)
        return f"""
/* ==== Split-Feature CUDA Kernel ==== */
/*
 * User features (shared across all items) are broadcast via L1 cache.
 * Item features (unique per thread) are read from the item matrix.
 * Each thread assembles a combined feature vector in local memory.
 */

__global__ void score_kernel(
    const float* __restrict__ d_user,   /* [NUM_USER_FEATURES] one vector, shared */
    const float* __restrict__ d_items,  /* [n_items * NUM_ITEM_FEATURES] row-major */
    float* __restrict__ scores,         /* [n_items] output */
    int n_items
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_items) return;

    /* Compose full feature vector: user prefix + item suffix */
    float f[NUM_FEATURES];
    for (int i = 0; i < NUM_USER_FEATURES; i++)
        f[i] = d_user[i];
    const float* item = d_items + idx * NUM_ITEM_FEATURES;
    for (int i = 0; i < NUM_ITEM_FEATURES; i++)
        f[NUM_USER_FEATURES + i] = item[i];

    float raw = BASE_SCORE + score_all_trees(f);
    scores[idx] = {transform};
}}
"""

    def _emit_split_cuda_topk_kernel(self) -> str:
        transform = self._get_transform_expr(cuda=True)
        return f"""
/* ==== Fused Score + Per-Block Top-K Kernel ==== */
/*
 * Two-phase top-K selection for split-feature ad ranking:
 *
 * Phase 1 (this kernel):
 *   Each block processes a chunk of items. Threads score items in parallel
 *   (one item per thread per batch). After each batch, thread 0 merges
 *   the batch results into a sorted top-K list maintained in shared memory.
 *   Each block writes its local top-K candidates to global memory.
 *
 * Phase 2 (host — see score_user_topk()):
 *   Merge grid_size × K block candidates via qsort to produce the final
 *   global top-K result set.
 *
 * Shared memory layout:
 *   float  s_topk_scores [k]              — block top-K (sorted descending)
 *   int    s_topk_indices[k]              — corresponding item indices
 *   float  s_batch_scores [TOPK_BLOCK_SIZE] — batch scores buffer
 *   int    s_batch_indices[TOPK_BLOCK_SIZE] — batch indices buffer
 *
 * Example: 10M items, block_size=256, grid_size=4096, K=100
 *   → 4096 blocks × 100 candidates = 409,600 total
 *   → host sorts 409K items to find final top-100
 */

#define TOPK_BLOCK_SIZE 256
#define TOPK_MAX_BLOCKS 4096

__global__ void score_topk_kernel(
    const float* __restrict__ d_user,    /* [NUM_USER_FEATURES] shared */
    const float* __restrict__ d_items,   /* [n_items * NUM_ITEM_FEATURES] */
    float* __restrict__ block_topk_scores,  /* [gridDim.x * k] out */
    int*   __restrict__ block_topk_indices, /* [gridDim.x * k] out */
    int n_items,
    int k
) {{
    extern __shared__ char smem[];
    float* s_topk_scores   = (float*)smem;
    int*   s_topk_indices  = (int*)(s_topk_scores + k);
    float* s_batch_scores  = (float*)(s_topk_indices + k);
    int*   s_batch_indices = (int*)(s_batch_scores + blockDim.x);

    int tid = threadIdx.x;

    /* Compute this block's item range */
    int items_per_block = (n_items + gridDim.x - 1) / gridDim.x;
    int block_start = blockIdx.x * items_per_block;
    int block_end   = block_start + items_per_block;
    if (block_end > n_items) block_end = n_items;

    /* Cache user features in registers */
    float user[NUM_USER_FEATURES];
    for (int i = 0; i < NUM_USER_FEATURES; i++)
        user[i] = d_user[i];

    /* Initialize sorted top-K with sentinel values */
    for (int i = tid; i < k; i += blockDim.x) {{
        s_topk_scores[i]  = -1e30f;
        s_topk_indices[i] = -1;
    }}
    __syncthreads();

    /* Process items in batches of blockDim.x */
    for (int batch = block_start; batch < block_end; batch += blockDim.x) {{
        int idx = batch + tid;
        float score = -1e30f;

        if (idx < block_end) {{
            /* Compose feature vector: user prefix + item suffix */
            float f[NUM_FEATURES];
            for (int i = 0; i < NUM_USER_FEATURES; i++)
                f[i] = user[i];
            const float* item = d_items + idx * NUM_ITEM_FEATURES;
            for (int i = 0; i < NUM_ITEM_FEATURES; i++)
                f[NUM_USER_FEATURES + i] = item[i];

            float raw = BASE_SCORE + score_all_trees(f);
            score = {transform};
        }}

        /* Write batch scores to shared memory */
        s_batch_scores[tid]  = score;
        s_batch_indices[tid] = idx;
        __syncthreads();

        /* Thread 0: merge batch into sorted top-K (insertion sort) */
        if (tid == 0) {{
            int batch_size = block_end - batch;
            if (batch_size > (int)blockDim.x) batch_size = (int)blockDim.x;
            float min_topk = s_topk_scores[k - 1];

            for (int i = 0; i < batch_size; i++) {{
                float s = s_batch_scores[i];
                if (s > min_topk) {{
                    /* Binary search for insertion position (descending) */
                    int lo = 0, hi = k - 1;
                    while (lo < hi) {{
                        int mid = (lo + hi) >> 1;
                        if (s_topk_scores[mid] >= s)
                            lo = mid + 1;
                        else
                            hi = mid;
                    }}
                    /* Shift elements down to make room */
                    for (int j = k - 1; j > lo; j--) {{
                        s_topk_scores[j]  = s_topk_scores[j - 1];
                        s_topk_indices[j] = s_topk_indices[j - 1];
                    }}
                    s_topk_scores[lo]  = s;
                    s_topk_indices[lo] = s_batch_indices[i];
                    min_topk = s_topk_scores[k - 1];
                }}
            }}
        }}
        __syncthreads();
    }}

    /* Write block's top-K to global output */
    for (int i = tid; i < k; i += blockDim.x) {{
        int base = blockIdx.x * k;
        block_topk_scores[base + i]  = s_topk_scores[i];
        block_topk_indices[base + i] = s_topk_indices[i];
    }}
}}
"""

    def _emit_split_cuda_host_api(self) -> str:
        return """
/* ==== Split-Feature Host API ==== */
/*
 * Designed for ad ranking / recommendation scoring:
 *   - load_items(): upload item features to GPU once (or when inventory changes)
 *   - score_user(): per request, transfer only the user vector, score all items
 *   - score_user_topk(): fused scoring + top-K selection (avoids N-score transfer)
 *
 * Data transfer per score_user() call:
 *   Host → Device: NUM_USER_FEATURES × 4 bytes  (user vector only)
 *   Device → Host: items_loaded × 4 bytes        (one score per item)
 *
 * Data transfer per score_user_topk() call:
 *   Host → Device: NUM_USER_FEATURES × 4 bytes  (user vector only)
 *   Device → Host: grid_size × K × 8 bytes       (block candidates only)
 *   Item features: ZERO transfer (resident on GPU)
 */

/* Result type for top-K merge sort */
typedef struct {
    float score;
    int   index;
} ScoredItem;

static int scored_item_cmp_desc(const void* a, const void* b) {
    float sa = ((const ScoredItem*)a)->score;
    float sb = ((const ScoredItem*)b)->score;
    if (sa > sb) return -1;
    if (sa < sb) return 1;
    return 0;
}

typedef struct {
    float* d_user;                /* [NUM_USER_FEATURES] */
    float* d_items;               /* [capacity * NUM_ITEM_FEATURES] */
    float* d_scores;              /* [capacity] for full scoring */
    float* d_block_topk_scores;   /* [topk_max_blocks * topk_max_k] */
    int*   d_block_topk_indices;  /* [topk_max_blocks * topk_max_k] */
    int    capacity;
    int    items_loaded;
    int    topk_max_k;
    int    topk_max_blocks;
} ScoringContext;

ScoringContext* scoring_context_create(int max_items) {
    ScoringContext* ctx = (ScoringContext*)malloc(sizeof(ScoringContext));
    if (!ctx) return NULL;
    ctx->capacity = max_items;
    ctx->items_loaded = 0;
    cudaMalloc(&ctx->d_user,   NUM_USER_FEATURES * sizeof(float));
    cudaMalloc(&ctx->d_items,  (size_t)max_items * NUM_ITEM_FEATURES * sizeof(float));
    cudaMalloc(&ctx->d_scores, (size_t)max_items * sizeof(float));

    /* Top-K buffers */
    ctx->topk_max_blocks = (max_items + TOPK_BLOCK_SIZE - 1) / TOPK_BLOCK_SIZE;
    if (ctx->topk_max_blocks > TOPK_MAX_BLOCKS)
        ctx->topk_max_blocks = TOPK_MAX_BLOCKS;
    ctx->topk_max_k = 1024;
    cudaMalloc(&ctx->d_block_topk_scores,
               (size_t)ctx->topk_max_blocks * ctx->topk_max_k * sizeof(float));
    cudaMalloc(&ctx->d_block_topk_indices,
               (size_t)ctx->topk_max_blocks * ctx->topk_max_k * sizeof(int));
    return ctx;
}

void scoring_context_destroy(ScoringContext* ctx) {
    if (!ctx) return;
    cudaFree(ctx->d_user);
    cudaFree(ctx->d_items);
    cudaFree(ctx->d_scores);
    cudaFree(ctx->d_block_topk_scores);
    cudaFree(ctx->d_block_topk_indices);
    free(ctx);
}

/**
 * Load item feature vectors to GPU. Call once, or when items change.
 * Items remain resident on GPU across multiple score_user() / score_user_topk() calls.
 *
 * @param ctx     Scoring context
 * @param h_items Host: [n_items × NUM_ITEM_FEATURES] row-major float32
 * @param n_items Number of items
 * @return 0 on success, -1 on error
 */
int load_items(ScoringContext* ctx, const float* h_items, int n_items) {
    if (n_items > ctx->capacity) {
        fprintf(stderr, "ERROR: n_items (%d) exceeds capacity (%d)\\n",
                n_items, ctx->capacity);
        return -1;
    }
    cudaMemcpy(ctx->d_items, h_items,
               (size_t)n_items * NUM_ITEM_FEATURES * sizeof(float),
               cudaMemcpyHostToDevice);
    ctx->items_loaded = n_items;
    return 0;
}

/**
 * Score all loaded items against one user.
 * Only the user feature vector is transferred to GPU.
 *
 * @param ctx      Context with items already loaded via load_items()
 * @param h_user   Host: [NUM_USER_FEATURES] user feature vector
 * @param h_scores Host: [items_loaded] output scores
 * @return 0 on success, -1 on error
 */
int score_user(
    ScoringContext* ctx,
    const float* h_user,
    float* h_scores
) {
    if (ctx->items_loaded <= 0) {
        fprintf(stderr, "ERROR: no items loaded (call load_items first)\\n");
        return -1;
    }

    /* Transfer only the user vector */
    cudaMemcpy(ctx->d_user, h_user,
               NUM_USER_FEATURES * sizeof(float),
               cudaMemcpyHostToDevice);

    int n = ctx->items_loaded;
    int block_size = 256;
    int grid_size  = (n + block_size - 1) / block_size;
    score_kernel<<<grid_size, block_size>>>(
        ctx->d_user, ctx->d_items, ctx->d_scores, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_scores, ctx->d_scores,
               (size_t)n * sizeof(float),
               cudaMemcpyDeviceToHost);
    return 0;
}

/**
 * Score all loaded items and return only the top-K results.
 *
 * Uses a fused scoring + per-block top-K kernel:
 *   Phase 1: Each CUDA block scores its chunk and maintains a local top-K
 *            in shared memory (sorted insertion, no global atomics).
 *   Phase 2: Host merges block candidates via qsort for the global top-K.
 *
 * Compared to score_user(), this avoids transferring all N scores back to
 * the host — only grid_size × K candidates cross the PCIe bus.
 *
 * @param ctx             Context with items loaded via load_items()
 * @param h_user          [NUM_USER_FEATURES] user feature vector
 * @param k               Number of top results (must be <= topk_max_k)
 * @param h_topk_indices  Output: [k] item indices sorted by score descending
 * @param h_topk_scores   Output: [k] scores sorted descending
 * @return 0 on success, -1 on error
 */
int score_user_topk(
    ScoringContext* ctx,
    const float* h_user,
    int k,
    int* h_topk_indices,
    float* h_topk_scores
) {
    if (ctx->items_loaded <= 0) {
        fprintf(stderr, "ERROR: no items loaded (call load_items first)\\n");
        return -1;
    }
    if (k <= 0 || k > ctx->topk_max_k) {
        fprintf(stderr, "ERROR: k=%d out of range [1, %d]\\n",
                k, ctx->topk_max_k);
        return -1;
    }

    /* Transfer user vector (only data crossing PCIe per request) */
    cudaMemcpy(ctx->d_user, h_user,
               NUM_USER_FEATURES * sizeof(float),
               cudaMemcpyHostToDevice);

    int n = ctx->items_loaded;
    int grid_size = (n + TOPK_BLOCK_SIZE - 1) / TOPK_BLOCK_SIZE;
    if (grid_size > ctx->topk_max_blocks) grid_size = ctx->topk_max_blocks;

    /* Dynamic shared memory:
     *   (k + TOPK_BLOCK_SIZE) floats  — scores  (topk + batch)
     *   (k + TOPK_BLOCK_SIZE) ints    — indices (topk + batch)
     */
    size_t smem = (size_t)(k + TOPK_BLOCK_SIZE) * (sizeof(float) + sizeof(int));

    score_topk_kernel<<<grid_size, TOPK_BLOCK_SIZE, smem>>>(
        ctx->d_user, ctx->d_items,
        ctx->d_block_topk_scores, ctx->d_block_topk_indices,
        n, k);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\\n",
                cudaGetErrorString(err));
        return -1;
    }
    cudaDeviceSynchronize();

    /* Copy block candidates to host */
    int n_candidates = grid_size * k;
    float* h_blk_scores  = (float*)malloc(n_candidates * sizeof(float));
    int*   h_blk_indices = (int*)malloc(n_candidates * sizeof(int));
    if (!h_blk_scores || !h_blk_indices) {
        free(h_blk_scores);
        free(h_blk_indices);
        fprintf(stderr, "ERROR: malloc failed for block candidates\\n");
        return -1;
    }

    cudaMemcpy(h_blk_scores, ctx->d_block_topk_scores,
               n_candidates * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blk_indices, ctx->d_block_topk_indices,
               n_candidates * sizeof(int), cudaMemcpyDeviceToHost);

    /* Phase 2: host merge — sort all block candidates, extract top k */
    ScoredItem* candidates = (ScoredItem*)malloc(n_candidates * sizeof(ScoredItem));
    int valid = 0;
    for (int i = 0; i < n_candidates; i++) {
        if (h_blk_indices[i] >= 0) {
            candidates[valid].score = h_blk_scores[i];
            candidates[valid].index = h_blk_indices[i];
            valid++;
        }
    }
    qsort(candidates, valid, sizeof(ScoredItem), scored_item_cmp_desc);

    int result_k = k < valid ? k : valid;
    for (int i = 0; i < result_k; i++) {
        h_topk_scores[i]  = candidates[i].score;
        h_topk_indices[i] = candidates[i].index;
    }
    for (int i = result_k; i < k; i++) {
        h_topk_scores[i]  = -1e30f;
        h_topk_indices[i] = -1;
    }

    free(candidates);
    free(h_blk_scores);
    free(h_blk_indices);
    return 0;
}
"""

    def _emit_split_cpu_scoring(self) -> str:
        transform = self._get_transform_expr(cuda=False)
        return f"""
/* ==== Split-Feature CPU Scoring ==== */

/* Result type for top-K sort (also used by CPU path) */
typedef struct {{
    float score;
    int   index;
}} ScoredItem;

static int scored_item_cmp_desc(const void* a, const void* b) {{
    float sa = ((const ScoredItem*)a)->score;
    float sb = ((const ScoredItem*)b)->score;
    if (sa > sb) return -1;
    if (sa < sb) return 1;
    return 0;
}}

/**
 * Score all items against one user on CPU.
 *
 * @param user_features  [NUM_USER_FEATURES] single user vector
 * @param item_features  [n_items × NUM_ITEM_FEATURES] row-major
 * @param scores         [n_items] output
 * @param n_items        Number of items to score
 */
void score_user_cpu(
    const float* user_features,
    const float* item_features,
    float* scores,
    int n_items
) {{
    for (int idx = 0; idx < n_items; idx++) {{
        float f[NUM_FEATURES];
        for (int i = 0; i < NUM_USER_FEATURES; i++)
            f[i] = user_features[i];
        const float* item = item_features + idx * NUM_ITEM_FEATURES;
        for (int i = 0; i < NUM_ITEM_FEATURES; i++)
            f[NUM_USER_FEATURES + i] = item[i];

        float raw = BASE_SCORE + score_all_trees(f);
        scores[idx] = {transform};
    }}
}}
"""

    # ==================================================================
    # Library mode: separate core header + driver files + benchmark
    # ==================================================================

    def generate_split_library(
        self, output_dir: str, user_feature_count: int,
    ) -> dict[str, str]:
        """
        Generate split-feature scoring as a reusable library.

        Produces separate files so the core scoring engine can be
        incorporated into any runtime:

          scoring_split_core.cuh   — CUDA header-only library
          scoring_split_core.h     — CPU header-only library
          scoring_split_main.cu    — CUDA standalone driver
          scoring_split_main.c     — CPU standalone driver
          scoring_split_bench.cu   — CUDA benchmark driver
          scoring_split_bench.c    — CPU benchmark driver
          Makefile                 — builds all targets

        Args:
            output_dir: Directory for generated files
            user_feature_count: Features 0..N-1 are user, N..end are item

        Returns:
            dict mapping file role to path
        """
        self._validate_split(user_feature_count)
        user_fc = user_feature_count

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = {}

        for cuda, ext in [(True, "cuh"), (False, "h")]:
            core = self._generate_split_core(cuda, user_fc)
            p = out / f"scoring_split_core.{ext}"
            self._write(str(p), core)
            files[f"{'cuda' if cuda else 'c'}_core"] = str(p)

        for cuda, ext in [(True, "cu"), (False, "c")]:
            drv = self._emit_split_main_driver(cuda)
            p = out / f"scoring_split_main.{ext}"
            self._write(str(p), drv)
            files[f"{'cuda' if cuda else 'c'}_main"] = str(p)

        for cuda, ext in [(True, "cu"), (False, "c")]:
            bench = self._emit_split_bench_driver(cuda)
            p = out / f"scoring_split_bench.{ext}"
            self._write(str(p), bench)
            files[f"{'cuda' if cuda else 'c'}_bench"] = str(p)

        mf = self._emit_split_makefile()
        p = out / "Makefile"
        self._write(str(p), mf)
        files["makefile"] = str(p)

        logger.info(
            "Generated split library in %s (%d files)",
            output_dir, len(files),
        )
        return files

    def _generate_split_core(self, cuda: bool, user_fc: int) -> str:
        """Generate the header-only core library (everything except main)."""
        item_fc = self.model_info.num_features - user_fc
        guard = "SCORING_SPLIT_CORE_CUH" if cuda else "SCORING_SPLIT_CORE_H"

        sections = [
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
            self._emit_split_header(cuda, user_fc, item_fc),
            self._emit_includes(cuda),
            self._emit_split_constants(user_fc, item_fc),
        ]

        for i, tree in enumerate(self.trees):
            sections.append(self._emit_tree_function(i, tree, cuda))

        sections.append(self._emit_score_all_trees(cuda))

        if cuda:
            sections.append(self._emit_split_cuda_kernel())
            sections.append(self._emit_split_cuda_topk_kernel())
            sections.append(self._emit_split_cuda_host_api())
        else:
            sections.append(self._emit_split_cpu_scoring())

        sections.append(f"\n#endif /* {guard} */\n")
        return "\n".join(sections)

    def _emit_split_main_driver(self, cuda: bool) -> str:
        """Emit the standalone main() driver that includes the core header."""
        inc = "scoring_split_core.cuh" if cuda else "scoring_split_core.h"
        main_fn = self._emit_split_main(cuda)
        return f'#include "{inc}"\n{main_fn}'

    def _compute_feature_ranges(self) -> list[tuple[float, float]]:
        """Extract per-feature value ranges from tree split thresholds.

        Walks all trees to find min/max split thresholds per feature,
        then adds margin so random benchmark data covers realistic ranges.
        """
        n = self.model_info.num_features
        feat_min = [float('inf')] * n
        feat_max = [float('-inf')] * n

        def _walk(node):
            if "leaf" in node:
                return
            split_name = node["split"]
            fid = int(split_name[1:] if split_name.startswith("f") else split_name)
            thr = float(node["split_condition"])
            feat_min[fid] = min(feat_min[fid], thr)
            feat_max[fid] = max(feat_max[fid], thr)
            for child in node.get("children", []):
                _walk(child)

        for tree in self.trees:
            _walk(tree)

        # Add 20% margin; floor at 0 for features with all-positive thresholds
        ranges = []
        for i in range(n):
            lo, hi = feat_min[i], feat_max[i]
            if lo == float('inf'):  # feature never split on
                lo, hi = 0.0, 1.0
            span = max(hi - lo, 1.0)
            lo_out = max(0.0, lo - 0.2 * span) if lo >= 0 else lo - 0.2 * span
            hi_out = hi + 0.2 * span
            ranges.append((lo_out, hi_out))
        return ranges

    def _emit_feature_range_arrays(self) -> str:
        """Emit C arrays of per-feature min/max for realistic random data."""
        ranges = self._compute_feature_ranges()
        n = len(ranges)
        lines = []
        lines.append(f"static const float FEAT_MIN[{n}] = {{")
        for i in range(0, n, 8):
            chunk = ranges[i:i+8]
            vals = ", ".join(f"{lo:.2f}f" for lo, _ in chunk)
            lines.append(f"    {vals},")
        lines.append("};")
        lines.append(f"static const float FEAT_MAX[{n}] = {{")
        for i in range(0, n, 8):
            chunk = ranges[i:i+8]
            vals = ", ".join(f"{hi:.2f}f" for _, hi in chunk)
            lines.append(f"    {vals},")
        lines.append("};")
        return "\n".join(lines)

    def _emit_split_bench_driver(self, cuda: bool) -> str:
        """Emit a benchmark driver for split-feature scoring."""
        if cuda:
            return self._emit_split_bench_driver_cuda()
        return self._emit_split_bench_driver_cpu()

    def _emit_split_bench_driver_cuda(self) -> str:
        feat_arrays = self._emit_feature_range_arrays()
        lines = []
        a = lines.append
        a('#include "scoring_split_core.cuh"')
        a('#include <time.h>')
        a('')
        a(feat_arrays)
        a('')
        a('static double get_time_ms(void) {')
        a('    struct timespec ts;')
        a('    clock_gettime(CLOCK_MONOTONIC, &ts);')
        a('    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;')
        a('}')
        a('')
        a('/* RNG: uniform in [lo, hi] */')
        a('static float randf_range(unsigned int *state, float lo, float hi) {')
        a('    *state = *state * 1103515245u + 12345u;')
        a('    float t = (float)(*state >> 16) / 65535.0f;')
        a('    return lo + t * (hi - lo);')
        a('}')
        a('')
        a('int main(int argc, char** argv) {')
        a('    int n_items    = 1000000;')
        a('    int topk       = 100;')
        a('    int warmup     = 5;')
        a('    int iterations = 20;')
        a('    unsigned int seed = 42;')
        a('')
        a('    for (int i = 1; i < argc; i++) {')
        a('        if      (strcmp(argv[i], "-n") == 0 && i+1 < argc) n_items    = atoi(argv[++i]);')
        a('        else if (strcmp(argv[i], "-k") == 0 && i+1 < argc) topk       = atoi(argv[++i]);')
        a('        else if (strcmp(argv[i], "-w") == 0 && i+1 < argc) warmup     = atoi(argv[++i]);')
        a('        else if (strcmp(argv[i], "-i") == 0 && i+1 < argc) iterations = atoi(argv[++i]);')
        a('        else if (strcmp(argv[i], "-s") == 0 && i+1 < argc) seed       = (unsigned)atoi(argv[++i]);')
        a('        else {')
        a('            fprintf(stderr,')
        a('                "Usage: %s [-n items] [-k topK] [-w warmup] [-i iters] [-s seed]\\n"')
        a('                "\\n  Defaults: -n 1000000 -k 100 -w 5 -i 20 -s 42\\n", argv[0]);')
        a('            return (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) ? 0 : 1;')
        a('        }')
        a('    }')
        a('')
        a('    /* GPU info */')
        a('    cudaDeviceProp prop;')
        a('    cudaGetDeviceProperties(&prop, 0);')
        a('')
        a('    printf("=== CUDA Split-Feature Scoring Benchmark ===\\n");')
        a('    printf("  GPU:           %s\\n", prop.name);')
        a('    printf("  SMs:           %d\\n", prop.multiProcessorCount);')
        a('    printf("  Items:         %d\\n", n_items);')
        a('    printf("  Top-K:         %d\\n", topk);')
        a('    printf("  User features: %d\\n", NUM_USER_FEATURES);')
        a('    printf("  Item features: %d\\n", NUM_ITEM_FEATURES);')
        a('    printf("  Trees:         %d\\n", NUM_TREES);')
        a('    printf("  Warmup:        %d\\n", warmup);')
        a('    printf("  Iterations:    %d\\n", iterations);')
        a('    printf("  Seed:          %u\\n\\n", seed);')
        a('')
        a('    /* Generate random data in realistic feature ranges */')
        a('    unsigned int rng = seed;')
        a('    float user[NUM_USER_FEATURES];')
        a('    for (int i = 0; i < NUM_USER_FEATURES; i++)')
        a('        user[i] = randf_range(&rng, FEAT_MIN[i], FEAT_MAX[i]);')
        a('')
        a('    size_t items_bytes = (size_t)n_items * NUM_ITEM_FEATURES * sizeof(float);')
        a('    float* items = (float*)malloc(items_bytes);')
        a('    if (!items) { perror("malloc items"); return 1; }')
        a('    for (int j = 0; j < n_items; j++)')
        a('        for (int f = 0; f < NUM_ITEM_FEATURES; f++)')
        a('            items[j * NUM_ITEM_FEATURES + f] = randf_range(&rng, FEAT_MIN[NUM_USER_FEATURES + f], FEAT_MAX[NUM_USER_FEATURES + f]);')
        a('    printf("Generated %.1f MB of random item features\\n", items_bytes / (1024.0 * 1024.0));')
        a('')
        a('    /* Create context + load items */')
        a('    double t0 = get_time_ms();')
        a('    ScoringContext* ctx = scoring_context_create(n_items);')
        a('    if (!ctx) { fprintf(stderr, "Failed to create context\\n"); return 1; }')
        a('    if (load_items(ctx, items, n_items) != 0) return 1;')
        a('    double load_ms = get_time_ms() - t0;')
        a('    printf("Item load (H->D): %10.2f ms  (%.1f GB/s)\\n\\n", load_ms, items_bytes / load_ms / 1e6);')
        a('')
        a('    /* Allocate output buffers */')
        a('    float* scores = (float*)malloc((size_t)n_items * sizeof(float));')
        a('    int*   topk_idx = (int*)malloc(topk * sizeof(int));')
        a('    float* topk_sc  = (float*)malloc(topk * sizeof(float));')
        a('    if (!scores || !topk_idx || !topk_sc) { perror("malloc"); return 1; }')
        a('')
        a('    /* ---- Warmup ---- */')
        a('    printf("Warming up (%d runs)...\\n", warmup);')
        a('    for (int w = 0; w < warmup; w++) {')
        a('        score_user(ctx, user, scores);')
        a('        score_user_topk(ctx, user, topk, topk_idx, topk_sc);')
        a('    }')
        a('')
        a('    /* ---- Score verification ---- */')
        a('    score_user(ctx, user, scores);')
        a('    float s_min = scores[0], s_max = scores[0];')
        a('    double s_sum = 0.0;')
        a('    int unique_count = 0;')
        a('    for (int i = 0; i < n_items; i++) {')
        a('        if (scores[i] < s_min) s_min = scores[i];')
        a('        if (scores[i] > s_max) s_max = scores[i];')
        a('        s_sum += scores[i];')
        a('    }')
        a('    /* Count unique scores (sample first 100K) */')
        a('    {')
        a('        int check_n = n_items < 100000 ? n_items : 100000;')
        a('        float* sample = (float*)malloc(check_n * sizeof(float));')
        a('        for (int i = 0; i < check_n; i++) sample[i] = scores[i * ((n_items + check_n - 1) / check_n)];')
        a('        /* simple unique count via sort */')
        a('        for (int i = 0; i < check_n - 1; i++)')
        a('            for (int j = i + 1; j < check_n && j < i + 50; j++)')
        a('                if (sample[j] < sample[i]) { float t = sample[i]; sample[i] = sample[j]; sample[j] = t; }')
        a('        unique_count = 1;')
        a('        for (int i = 1; i < check_n; i++)')
        a('            if (sample[i] != sample[i-1]) unique_count++;')
        a('        free(sample);')
        a('    }')
        a('    printf("--- Score Verification ---\\n");')
        a('    printf("  Min score:     %.8f\\n", s_min);')
        a('    printf("  Max score:     %.8f\\n", s_max);')
        a('    printf("  Mean score:    %.8f\\n", s_sum / n_items);')
        a('    printf("  Unique scores: %d (sampled)\\n", unique_count);')
        a('    printf("  Score[0]:      %.8f\\n", scores[0]);')
        a('    printf("  Score[N/4]:    %.8f\\n", scores[n_items/4]);')
        a('    printf("  Score[N/2]:    %.8f\\n", scores[n_items/2]);')
        a('    printf("  Score[3N/4]:   %.8f\\n", scores[3*n_items/4]);')
        a('    printf("  Score[N-1]:    %.8f\\n\\n", scores[n_items-1]);')
        a('')
        a('    /* ---- Benchmark: full scoring ---- */')
        a('    t0 = get_time_ms();')
        a('    for (int it = 0; it < iterations; it++)')
        a('        score_user(ctx, user, scores);')
        a('    double full_ms = (get_time_ms() - t0) / iterations;')
        a('')
        a('    /* ---- Benchmark: top-K ---- */')
        a('    t0 = get_time_ms();')
        a('    for (int it = 0; it < iterations; it++)')
        a('        score_user_topk(ctx, user, topk, topk_idx, topk_sc);')
        a('    double topk_ms = (get_time_ms() - t0) / iterations;')
        a('')
        a('    /* ---- Results ---- */')
        a('    printf("\\n--- Results ---\\n");')
        a('    printf("  score_user():      %10.2f ms  (%8.2f M items/sec)\\n",')
        a('           full_ms, n_items / full_ms / 1000.0);')
        a('    printf("  score_user_topk(): %10.2f ms  (%8.2f M items/sec)\\n",')
        a('           topk_ms, n_items / topk_ms / 1000.0);')
        a('    printf("  D->H transfer:     %.1f MB (full) vs %.1f MB (topk blocks)\\n",')
        a('           n_items * 4.0 / (1024*1024),')
        a('           (double)((n_items + TOPK_BLOCK_SIZE - 1) / TOPK_BLOCK_SIZE) * topk * 8.0 / (1024*1024));')
        a('')
        a('    /* ---- Top-K scaling ---- */')
        a('    int k_values[] = {10, 50, 100, 500, 1000};')
        a('    int n_kv = 5;')
        a('    printf("\\n--- Top-K Scaling ---\\n");')
        a('    for (int ki = 0; ki < n_kv; ki++) {')
        a('        int k = k_values[ki];')
        a('        if (k > n_items || k > 1024) break;')
        a('        int*   ki_idx = (int*)malloc(k * sizeof(int));')
        a('        float* ki_sc  = (float*)malloc(k * sizeof(float));')
        a('        score_user_topk(ctx, user, k, ki_idx, ki_sc);  /* warmup */')
        a('        t0 = get_time_ms();')
        a('        for (int it = 0; it < iterations; it++)')
        a('            score_user_topk(ctx, user, k, ki_idx, ki_sc);')
        a('        double k_ms = (get_time_ms() - t0) / iterations;')
        a('        printf("  K=%-5d  %10.2f ms  (%8.2f M items/sec)\\n",')
        a('               k, k_ms, n_items / k_ms / 1000.0);')
        a('        free(ki_idx); free(ki_sc);')
        a('    }')
        a('')
        a('    /* ---- Top-K preview ---- */')
        a('    printf("\\nTop-%d preview:\\n", topk < 10 ? topk : 10);')
        a('    int show = topk < 10 ? topk : 10;')
        a('    for (int i = 0; i < show; i++)')
        a('        printf("  rank[%d] = item[%d]  score=%.8f\\n", i, topk_idx[i], topk_sc[i]);')
        a('    if (topk > 10) printf("  ... (%d more)\\n", topk - 10);')
        a('')
        a('    scoring_context_destroy(ctx);')
        a('    free(items); free(scores);')
        a('    free(topk_idx); free(topk_sc);')
        a('    printf("\\nDone.\\n");')
        a('    return 0;')
        a('}')
        return "\n".join(lines)

    def _emit_split_bench_driver_cpu(self) -> str:
        feat_arrays = self._emit_feature_range_arrays()
        lines = []
        a = lines.append
        a('#include "scoring_split_core.h"')
        a('#include <time.h>')
        a('')
        a(feat_arrays)
        a('')
        a('static double get_time_ms(void) {')
        a('    struct timespec ts;')
        a('    clock_gettime(CLOCK_MONOTONIC, &ts);')
        a('    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;')
        a('}')
        a('')
        a('/* RNG: uniform in [lo, hi] */')
        a('static float randf_range(unsigned int *state, float lo, float hi) {')
        a('    *state = *state * 1103515245u + 12345u;')
        a('    float t = (float)(*state >> 16) / 65535.0f;')
        a('    return lo + t * (hi - lo);')
        a('}')
        a('')
        a('int main(int argc, char** argv) {')
        a('    int n_items    = 100000;')
        a('    int topk       = 100;')
        a('    int warmup     = 2;')
        a('    int iterations = 5;')
        a('    unsigned int seed = 42;')
        a('')
        a('    for (int i = 1; i < argc; i++) {')
        a('        if      (strcmp(argv[i], "-n") == 0 && i+1 < argc) n_items    = atoi(argv[++i]);')
        a('        else if (strcmp(argv[i], "-k") == 0 && i+1 < argc) topk       = atoi(argv[++i]);')
        a('        else if (strcmp(argv[i], "-w") == 0 && i+1 < argc) warmup     = atoi(argv[++i]);')
        a('        else if (strcmp(argv[i], "-i") == 0 && i+1 < argc) iterations = atoi(argv[++i]);')
        a('        else if (strcmp(argv[i], "-s") == 0 && i+1 < argc) seed       = (unsigned)atoi(argv[++i]);')
        a('        else {')
        a('            fprintf(stderr,')
        a('                "Usage: %s [-n items] [-k topK] [-w warmup] [-i iters] [-s seed]\\n"')
        a('                "\\n  Defaults: -n 100000 -k 100 -w 2 -i 5 -s 42\\n", argv[0]);')
        a('            return (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) ? 0 : 1;')
        a('        }')
        a('    }')
        a('')
        a('    printf("=== CPU Split-Feature Scoring Benchmark ===\\n");')
        a('    printf("  Items:         %d\\n", n_items);')
        a('    printf("  Top-K:         %d\\n", topk);')
        a('    printf("  User features: %d\\n", NUM_USER_FEATURES);')
        a('    printf("  Item features: %d\\n", NUM_ITEM_FEATURES);')
        a('    printf("  Trees:         %d\\n", NUM_TREES);')
        a('    printf("  Warmup:        %d\\n", warmup);')
        a('    printf("  Iterations:    %d\\n", iterations);')
        a('    printf("  Seed:          %u\\n\\n", seed);')
        a('')
        a('    /* Generate random data in realistic feature ranges */')
        a('    unsigned int rng = seed;')
        a('    float user[NUM_USER_FEATURES];')
        a('    for (int i = 0; i < NUM_USER_FEATURES; i++)')
        a('        user[i] = randf_range(&rng, FEAT_MIN[i], FEAT_MAX[i]);')
        a('')
        a('    size_t items_bytes = (size_t)n_items * NUM_ITEM_FEATURES * sizeof(float);')
        a('    float* items = (float*)malloc(items_bytes);')
        a('    if (!items) { perror("malloc items"); return 1; }')
        a('    for (int j = 0; j < n_items; j++)')
        a('        for (int f = 0; f < NUM_ITEM_FEATURES; f++)')
        a('            items[j * NUM_ITEM_FEATURES + f] = randf_range(&rng, FEAT_MIN[NUM_USER_FEATURES + f], FEAT_MAX[NUM_USER_FEATURES + f]);')
        a('    printf("Generated %.1f MB of random item features\\n", items_bytes / (1024.0 * 1024.0));')
        a('')
        a('    float* scores = (float*)malloc((size_t)n_items * sizeof(float));')
        a('    if (!scores) { perror("malloc scores"); return 1; }')
        a('')
        a('    /* ---- Warmup ---- */')
        a('    printf("Warming up (%d runs)...\\n", warmup);')
        a('    for (int w = 0; w < warmup; w++)')
        a('        score_user_cpu(user, items, scores, n_items);')
        a('')
        a('    /* ---- Score verification ---- */')
        a('    score_user_cpu(user, items, scores, n_items);')
        a('    float s_min = scores[0], s_max = scores[0];')
        a('    double s_sum = 0.0;')
        a('    int unique_count = 0;')
        a('    for (int i = 0; i < n_items; i++) {')
        a('        if (scores[i] < s_min) s_min = scores[i];')
        a('        if (scores[i] > s_max) s_max = scores[i];')
        a('        s_sum += scores[i];')
        a('    }')
        a('    /* Count unique scores (sample first 100K) */')
        a('    {')
        a('        int check_n = n_items < 100000 ? n_items : 100000;')
        a('        float* sample = (float*)malloc(check_n * sizeof(float));')
        a('        int stride = (n_items + check_n - 1) / check_n;')
        a('        for (int i = 0; i < check_n; i++) {')
        a('            int idx = i * stride;')
        a('            if (idx >= n_items) idx = n_items - 1;')
        a('            sample[i] = scores[idx];')
        a('        }')
        a('        /* simple unique count via sort */')
        a('        for (int i = 0; i < check_n - 1; i++)')
        a('            for (int j = i + 1; j < check_n && j < i + 50; j++)')
        a('                if (sample[j] < sample[i]) { float t = sample[i]; sample[i] = sample[j]; sample[j] = t; }')
        a('        unique_count = 1;')
        a('        for (int i = 1; i < check_n; i++)')
        a('            if (sample[i] != sample[i-1]) unique_count++;')
        a('        free(sample);')
        a('    }')
        a('    printf("--- Score Verification ---\\n");')
        a('    printf("  Min score:     %.8f\\n", s_min);')
        a('    printf("  Max score:     %.8f\\n", s_max);')
        a('    printf("  Mean score:    %.8f\\n", s_sum / n_items);')
        a('    printf("  Unique scores: %d (sampled)\\n", unique_count);')
        a('    printf("  Score[0]:      %.8f\\n", scores[0]);')
        a('    printf("  Score[N/4]:    %.8f\\n", scores[n_items/4]);')
        a('    printf("  Score[N/2]:    %.8f\\n", scores[n_items/2]);')
        a('    printf("  Score[3N/4]:   %.8f\\n", scores[3*n_items/4]);')
        a('    printf("  Score[N-1]:    %.8f\\n\\n", scores[n_items-1]);')
        a('')
        a('    /* ---- Benchmark: full scoring ---- */')
        a('    double t0 = get_time_ms();')
        a('    for (int it = 0; it < iterations; it++)')
        a('        score_user_cpu(user, items, scores, n_items);')
        a('    double full_ms = (get_time_ms() - t0) / iterations;')
        a('')
        a('    /* ---- Benchmark: score + top-K (qsort) ---- */')
        a('    ScoredItem* ranked = (ScoredItem*)malloc(n_items * sizeof(ScoredItem));')
        a('    int*   topk_idx = (int*)malloc(topk * sizeof(int));')
        a('    float* topk_sc  = (float*)malloc(topk * sizeof(float));')
        a('    if (!ranked || !topk_idx || !topk_sc) { perror("malloc"); return 1; }')
        a('')
        a('    t0 = get_time_ms();')
        a('    for (int it = 0; it < iterations; it++) {')
        a('        score_user_cpu(user, items, scores, n_items);')
        a('        for (int i = 0; i < n_items; i++) {')
        a('            ranked[i].score = scores[i];')
        a('            ranked[i].index = i;')
        a('        }')
        a('        qsort(ranked, n_items, sizeof(ScoredItem), scored_item_cmp_desc);')
        a('        int rk = topk < n_items ? topk : n_items;')
        a('        for (int i = 0; i < rk; i++) {')
        a('            topk_idx[i] = ranked[i].index;')
        a('            topk_sc[i]  = ranked[i].score;')
        a('        }')
        a('    }')
        a('    double topk_ms = (get_time_ms() - t0) / iterations;')
        a('')
        a('    /* ---- Results ---- */')
        a('    printf("\\n--- Results ---\\n");')
        a('    printf("  Full scoring:      %10.2f ms  (%8.2f K items/sec)\\n",')
        a('           full_ms, n_items / full_ms);')
        a('    printf("  Score + top-%-4d:  %10.2f ms  (%8.2f K items/sec)\\n",')
        a('           topk, topk_ms, n_items / topk_ms);')
        a('    printf("  Top-K overhead:    %10.2f ms  (%.1f%% of scoring)\\n",')
        a('           topk_ms - full_ms,')
        a('           full_ms > 0 ? (topk_ms - full_ms) / full_ms * 100.0 : 0.0);')
        a('')
        a('    /* ---- Top-K scaling ---- */')
        a('    int k_values[] = {10, 50, 100, 500, 1000};')
        a('    int n_kv = 5;')
        a('    printf("\\n--- Top-K Scaling (score + qsort) ---\\n");')
        a('    for (int ki = 0; ki < n_kv; ki++) {')
        a('        int k = k_values[ki];')
        a('        if (k > n_items) break;')
        a('        int*   ki_idx = (int*)malloc(k * sizeof(int));')
        a('        float* ki_sc  = (float*)malloc(k * sizeof(float));')
        a('        if (!ki_idx || !ki_sc) continue;')
        a('        t0 = get_time_ms();')
        a('        for (int it = 0; it < iterations; it++) {')
        a('            score_user_cpu(user, items, scores, n_items);')
        a('            for (int j = 0; j < n_items; j++) {')
        a('                ranked[j].score = scores[j];')
        a('                ranked[j].index = j;')
        a('            }')
        a('            qsort(ranked, n_items, sizeof(ScoredItem), scored_item_cmp_desc);')
        a('            int rk = k < n_items ? k : n_items;')
        a('            for (int j = 0; j < rk; j++) {')
        a('                ki_idx[j] = ranked[j].index;')
        a('                ki_sc[j]  = ranked[j].score;')
        a('            }')
        a('        }')
        a('        double k_ms = (get_time_ms() - t0) / iterations;')
        a('        printf("  K=%-5d  %10.2f ms  (%8.2f K items/sec)\\n",')
        a('               k, k_ms, n_items / k_ms);')
        a('        free(ki_idx); free(ki_sc);')
        a('    }')
        a('')
        a('    /* ---- Top-K preview ---- */')
        a('    printf("\\nTop-%d preview:\\n", topk < 10 ? topk : 10);')
        a('    int show = topk < 10 ? topk : 10;')
        a('    for (int i = 0; i < show; i++)')
        a('        printf("  rank[%d] = item[%d]  score=%.8f\\n", i, topk_idx[i], topk_sc[i]);')
        a('    if (topk > 10) printf("  ... (%d more)\\n", topk - 10);')
        a('')
        a('    free(items); free(scores); free(ranked);')
        a('    free(topk_idx); free(topk_sc);')
        a('    printf("\\nDone.\\n");')
        a('    return 0;')
        a('}')
        return "\n".join(lines)

    @staticmethod
    def _emit_split_makefile() -> str:
        return """# Auto-generated Makefile for split-feature scoring library
#
# Targets:
#   make cpu          — build CPU main + benchmark
#   make cuda         — build CUDA main + benchmark (requires nvcc)
#   make bench_cpu    — build & run CPU benchmark
#   make bench_cuda   — build & run CUDA benchmark
#   make all          — build everything available
#   make clean        — remove build artifacts
#
# Override compiler:  make CC=gcc-13 cpu
# Override flags:     make CFLAGS="-O3 -march=native" cpu

CC        ?= cc
NVCC      ?= nvcc
CFLAGS    ?= -O2 -Wall
NVFLAGS   ?= -O2
BUILD_DIR ?= ../build

# Detect nvcc
HAS_NVCC := $(shell command -v $(NVCC) 2>/dev/null)

.PHONY: all cpu cuda bench_cpu bench_cuda clean

all: cpu
	if [ -n "$(HAS_NVCC)" ]; then $(MAKE) cuda; fi

# ---- CPU targets ----

cpu: $(BUILD_DIR)/scoring_split_cpu $(BUILD_DIR)/scoring_split_bench_cpu

$(BUILD_DIR)/scoring_split_cpu: scoring_split_main.c scoring_split_core.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ scoring_split_main.c -lm
	@echo "Built: $@"

$(BUILD_DIR)/scoring_split_bench_cpu: scoring_split_bench.c scoring_split_core.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ scoring_split_bench.c -lm
	@echo "Built: $@"

bench_cpu: $(BUILD_DIR)/scoring_split_bench_cpu
	$(BUILD_DIR)/scoring_split_bench_cpu $(BENCH_ARGS)

# ---- CUDA targets ----

cuda: $(BUILD_DIR)/scoring_split_cuda $(BUILD_DIR)/scoring_split_bench_cuda

$(BUILD_DIR)/scoring_split_cuda: scoring_split_main.cu scoring_split_core.cuh
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVFLAGS) -o $@ scoring_split_main.cu
	@echo "Built: $@"

$(BUILD_DIR)/scoring_split_bench_cuda: scoring_split_bench.cu scoring_split_core.cuh
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVFLAGS) -o $@ scoring_split_bench.cu
	@echo "Built: $@"

bench_cuda: $(BUILD_DIR)/scoring_split_bench_cuda
	$(BUILD_DIR)/scoring_split_bench_cuda $(BENCH_ARGS)

clean:
	rm -f $(BUILD_DIR)/scoring_split_cpu
	rm -f $(BUILD_DIR)/scoring_split_cuda
	rm -f $(BUILD_DIR)/scoring_split_bench_cpu
	rm -f $(BUILD_DIR)/scoring_split_bench_cuda
"""

    def _emit_split_main(self, cuda: bool) -> str:
        if cuda:
            score_block = (
                "    ScoringContext* ctx = scoring_context_create(n_items);\n"
                "    if (!ctx) { fprintf(stderr, \"Failed to create context\\n\"); return 1; }\n"
                "    if (load_items(ctx, items, n_items) != 0) return 1;\n"
                "\n"
                "    if (topk > 0) {\n"
                "        /* Top-K mode: fused scoring + selection */\n"
                "        int* topk_indices = (int*)malloc(topk * sizeof(int));\n"
                "        float* topk_scores = (float*)malloc(topk * sizeof(float));\n"
                "        if (!topk_indices || !topk_scores) { perror(\"malloc topk\"); return 1; }\n"
                "        if (score_user_topk(ctx, user, topk, topk_indices, topk_scores) != 0) return 1;\n"
                "        scoring_context_destroy(ctx);\n"
                "\n"
                "        if (output_path) {\n"
                "            FILE* fout = fopen(output_path, \"wb\");\n"
                "            if (!fout) { perror(\"fopen output\"); return 1; }\n"
                "            fwrite(topk_indices, sizeof(int), topk, fout);\n"
                "            fwrite(topk_scores, sizeof(float), topk, fout);\n"
                "            fclose(fout);\n"
                "            fprintf(stderr, \"Wrote top-%d results to %s\\n\", topk, output_path);\n"
                "        } else {\n"
                "            for (int i = 0; i < topk; i++)\n"
                "                printf(\"rank[%d] = item[%d]  score=%.8f\\n\", i, topk_indices[i], topk_scores[i]);\n"
                "        }\n"
                "        free(topk_indices);\n"
                "        free(topk_scores);\n"
                "    } else {\n"
                "        /* Full scoring mode */\n"
                "        if (score_user(ctx, user, scores) != 0) return 1;\n"
                "        scoring_context_destroy(ctx);\n"
                "\n"
                "        if (output_path) {\n"
                "            FILE* fout = fopen(output_path, \"wb\");\n"
                "            if (!fout) { perror(\"fopen output\"); return 1; }\n"
                "            fwrite(scores, sizeof(float), (size_t)n_items, fout);\n"
                "            fclose(fout);\n"
                "            fprintf(stderr, \"Wrote %d scores to %s\\n\", n_items, output_path);\n"
                "        } else {\n"
                "            int limit = n_items < 20 ? n_items : 20;\n"
                "            for (int i = 0; i < limit; i++)\n"
                "                printf(\"item[%d] = %.8f\\n\", i, scores[i]);\n"
                "            if (n_items > 20)\n"
                "                printf(\"... (%d more items)\\n\", n_items - 20);\n"
                "        }\n"
                "    }\n"
            )
        else:
            score_block = (
                "    score_user_cpu(user, items, scores, n_items);\n"
                "\n"
                "    if (topk > 0) {\n"
                "        /* Top-K: sort all scores, output top K */\n"
                "        ScoredItem* ranked = (ScoredItem*)malloc(n_items * sizeof(ScoredItem));\n"
                "        for (int i = 0; i < n_items; i++) {\n"
                "            ranked[i].score = scores[i];\n"
                "            ranked[i].index = i;\n"
                "        }\n"
                "        qsort(ranked, n_items, sizeof(ScoredItem), scored_item_cmp_desc);\n"
                "        int result_k = topk < n_items ? topk : n_items;\n"
                "\n"
                "        if (output_path) {\n"
                "            FILE* fout = fopen(output_path, \"wb\");\n"
                "            if (!fout) { perror(\"fopen output\"); return 1; }\n"
                "            for (int i = 0; i < result_k; i++) {\n"
                "                int idx = ranked[i].index;\n"
                "                fwrite(&idx, sizeof(int), 1, fout);\n"
                "            }\n"
                "            for (int i = result_k; i < topk; i++) {\n"
                "                int neg = -1;\n"
                "                fwrite(&neg, sizeof(int), 1, fout);\n"
                "            }\n"
                "            for (int i = 0; i < result_k; i++) {\n"
                "                float s = ranked[i].score;\n"
                "                fwrite(&s, sizeof(float), 1, fout);\n"
                "            }\n"
                "            for (int i = result_k; i < topk; i++) {\n"
                "                float neg = -1e30f;\n"
                "                fwrite(&neg, sizeof(float), 1, fout);\n"
                "            }\n"
                "            fclose(fout);\n"
                "            fprintf(stderr, \"Wrote top-%d results to %s\\n\", topk, output_path);\n"
                "        } else {\n"
                "            for (int i = 0; i < result_k; i++)\n"
                "                printf(\"rank[%d] = item[%d]  score=%.8f\\n\", i, ranked[i].index, ranked[i].score);\n"
                "        }\n"
                "        free(ranked);\n"
                "    } else {\n"
                "        /* Full scoring mode */\n"
                "        if (output_path) {\n"
                "            FILE* fout = fopen(output_path, \"wb\");\n"
                "            if (!fout) { perror(\"fopen output\"); return 1; }\n"
                "            fwrite(scores, sizeof(float), (size_t)n_items, fout);\n"
                "            fclose(fout);\n"
                "            fprintf(stderr, \"Wrote %d scores to %s\\n\", n_items, output_path);\n"
                "        } else {\n"
                "            int limit = n_items < 20 ? n_items : 20;\n"
                "            for (int i = 0; i < limit; i++)\n"
                "                printf(\"item[%d] = %.8f\\n\", i, scores[i]);\n"
                "            if (n_items > 20)\n"
                "                printf(\"... (%d more items)\\n\", n_items - 20);\n"
                "        }\n"
                "    }\n"
            )

        return f"""
/* ==== Standalone Entry Point ==== */

int main(int argc, char** argv) {{
    if (argc < 3) {{
        fprintf(stderr,
            "Usage: %s <user.bin> <items.bin> [output.bin] [-k topK]\\n"
            "\\n"
            "  user.bin:   binary float32 [%d] (one user feature vector)\\n"
            "  items.bin:  binary float32 [N x %d] (item features, row-major)\\n"
            "  output.bin: binary output (omit for stdout)\\n"
            "  -k topK:    return only the top K results (ranked by score)\\n",
            argv[0], NUM_USER_FEATURES, NUM_ITEM_FEATURES);
        return 1;
    }}

    /* Parse arguments */
    const char* user_path = argv[1];
    const char* items_path = argv[2];
    const char* output_path = NULL;
    int topk = 0;

    for (int a = 3; a < argc; a++) {{
        if (strcmp(argv[a], "-k") == 0 && a + 1 < argc) {{
            topk = atoi(argv[++a]);
        }} else if (!output_path) {{
            output_path = argv[a];
        }}
    }}

    /* Read user features */
    FILE* fu = fopen(user_path, "rb");
    if (!fu) {{ perror("fopen user"); return 1; }}
    float user[NUM_USER_FEATURES];
    if (fread(user, sizeof(float), NUM_USER_FEATURES, fu) != NUM_USER_FEATURES) {{
        fprintf(stderr, "Failed to read %d user features\\n", NUM_USER_FEATURES);
        fclose(fu);
        return 1;
    }}
    fclose(fu);

    /* Read item features */
    FILE* fi = fopen(items_path, "rb");
    if (!fi) {{ perror("fopen items"); return 1; }}
    fseek(fi, 0, SEEK_END);
    long fsize = ftell(fi);
    fseek(fi, 0, SEEK_SET);
    int n_items = (int)(fsize / ((long)NUM_ITEM_FEATURES * (long)sizeof(float)));
    if (n_items <= 0) {{
        fprintf(stderr, "Invalid items file\\n");
        fclose(fi);
        return 1;
    }}

    if (topk > 0) {{
        fprintf(stderr, "Top-%d ranking of %d items (%d user + %d item features)...\\n",
                topk, n_items, NUM_USER_FEATURES, NUM_ITEM_FEATURES);
    }} else {{
        fprintf(stderr, "Scoring %d items against 1 user (%d user + %d item features)...\\n",
                n_items, NUM_USER_FEATURES, NUM_ITEM_FEATURES);
    }}

    float* items = (float*)malloc((size_t)fsize);
    if (!items) {{ perror("malloc items"); return 1; }}
    fread(items, sizeof(float), (size_t)n_items * NUM_ITEM_FEATURES, fi);
    fclose(fi);

    float* scores = (float*)malloc((size_t)n_items * sizeof(float));
    if (!scores) {{ perror("malloc scores"); return 1; }}

    /* Score */
{score_block}
    free(items);
    free(scores);
    return 0;
}}
"""

    # ==================================================================
    # Metal (Apple Silicon GPU) code generation
    # ==================================================================

    def generate_metal_split_library(
        self, output_dir: str, user_feature_count: int,
    ) -> dict[str, str]:
        """
        Generate split-feature scoring for Apple Silicon GPU (Metal).

        Produces:
          scoring_split_core.metal        — standalone Metal shader
          scoring_split_metal_core.h      — Obj-C++ header-only library
          scoring_split_main_metal.mm     — Metal standalone driver
          scoring_split_bench_metal.mm    — Metal benchmark driver
          Makefile.metal                  — build instructions

        The generated code uses MTLResourceStorageModeShared (zero-copy
        unified memory) for all buffers, which is optimal on Apple Silicon
        where CPU and GPU share the same physical DRAM.  The host API
        surface is identical to the CUDA version so calling code can swap
        backends by changing the header include.

        Args:
            output_dir: Directory for generated files
            user_feature_count: Features 0..N-1 are user, N..end are item

        Returns:
            dict mapping file role to path
        """
        self._validate_split(user_feature_count)
        user_fc = user_feature_count

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = {}

        # Standalone Metal shader (.metal) — for Xcode / xcrun precompile workflows
        shader = self._generate_metal_shader(user_fc)
        p = out / "scoring_split_core.metal"
        self._write(str(p), shader)
        files["metal_shader"] = str(p)

        # Obj-C++ header-only library: embeds shader, implements C API
        core_h = self._generate_metal_host_header(user_fc)
        p = out / "scoring_split_metal_core.h"
        self._write(str(p), core_h)
        files["metal_core"] = str(p)

        # Metal main driver
        p = out / "scoring_split_main_metal.mm"
        self._write(str(p), self._emit_metal_main_driver())
        files["metal_main"] = str(p)

        # Metal benchmark driver
        p = out / "scoring_split_bench_metal.mm"
        self._write(str(p), self._emit_metal_bench_driver(user_fc))
        files["metal_bench"] = str(p)

        # Makefile for Metal
        p = out / "Makefile.metal"
        self._write(str(p), self._emit_metal_makefile())
        files["metal_makefile"] = str(p)

        item_fc = self.model_info.num_features - user_fc
        logger.info(
            "Generated Metal library in %s (%d user + %d item features, %d trees)",
            output_dir, user_fc, item_fc, self.model_info.num_trees,
        )
        return files

    # ------------------------------------------------------------------
    # Metal helpers
    # ------------------------------------------------------------------

    def _get_metal_transform_expr(self) -> str:
        expr = _METAL_TRANSFORMS.get(self.model_info.objective)
        if expr is None:
            logger.warning(
                "Unknown objective %r, defaulting to identity transform",
                self.model_info.objective,
            )
            expr = "raw"
        return expr

    def _emit_metal_tree_function(self, idx: int, tree: dict) -> str:
        """Emit a Metal device tree function.

        Tree functions use thread address space for the feature pointer
        because callers assemble f[] in thread-local (register/stack) memory.
        """
        lines = [f"\nstatic inline float tree_{idx}(thread const float* f) {{"]
        lines.append(self._emit_node(tree, indent=1))
        lines.append("}\n")
        return "\n".join(lines)

    def _emit_metal_score_all_trees(self) -> str:
        lines = ["\nstatic inline float score_all_trees(thread const float* f) {"]
        lines.append("    float sum = 0.0f;")
        for i in range(0, len(self.trees), 8):
            for j in range(i, min(i + 8, len(self.trees))):
                lines.append(f"    sum += tree_{j}(f);")
        lines.append("    return sum;")
        lines.append("}\n")
        return "\n".join(lines)

    def _generate_metal_shader(self, user_fc: int) -> str:
        """Generate the standalone .metal shader file."""
        item_fc = self.model_info.num_features - user_fc
        transform = self._get_metal_transform_expr()

        user_names = ", ".join(self.model_info.feature_names[:user_fc][:8])
        if user_fc > 8:
            user_names += ", ..."
        item_names = ", ".join(self.model_info.feature_names[user_fc:][:8])
        if item_fc > 8:
            item_names += ", ..."

        sections = [
            f"/*\n"
            f" * Auto-generated Metal compute shader for XGBoost scoring\n"
            f" *\n"
            f" * Source model:    {self.model_path.name}\n"
            f" * Trees:           {self.model_info.num_trees}\n"
            f" * Total features:  {self.model_info.num_features}\n"
            f" *   User (0..{user_fc - 1}):   {user_fc}  [{user_names}]\n"
            f" *   Item ({user_fc}..{self.model_info.num_features - 1}):  {item_fc}  [{item_names}]\n"
            f" * Objective:       {self.model_info.objective}\n"
            f" * Base score:      {self.model_info.base_score:.8f}\n"
            f" * Generated:       {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
            f" *\n"
            f" * Compile: xcrun -sdk macosx metal -c scoring_split_core.metal -o scoring_split_core.air\n"
            f" *          xcrun -sdk macosx metallib scoring_split_core.air -o scoring_split_core.metallib\n"
            f" *\n"
            f" * DO NOT EDIT — auto-generated by cuda_codegen.\n"
            f" */\n"
            f"\n"
            f"#include <metal_stdlib>\n"
            f"using namespace metal;\n"
            f"\n"
            f"#define NUM_TREES         {self.model_info.num_trees}\n"
            f"#define NUM_FEATURES      {self.model_info.num_features}\n"
            f"#define NUM_USER_FEATURES {user_fc}\n"
            f"#define NUM_ITEM_FEATURES {item_fc}\n"
            f"#define BASE_SCORE        {self.model_info.base_score:.8f}f\n"
            f"#define TOPK_BLOCK_SIZE   256\n"
        ]

        for i, tree in enumerate(self.trees):
            sections.append(self._emit_metal_tree_function(i, tree))

        sections.append(self._emit_metal_score_all_trees())
        sections.append(self._emit_metal_score_kernel(transform))
        sections.append(self._emit_metal_topk_kernel(transform))

        return "\n".join(sections)

    def _emit_metal_score_kernel(self, transform: str) -> str:
        return f"""
/* ==== Metal Split-Feature Scoring Kernel ==== */
/*
 * One Metal thread per item. Each thread assembles a full feature vector
 * from the shared user prefix and its item suffix, traverses all decision
 * trees, and writes its score to the output buffer.
 *
 * Buffers:
 *   0: d_user  [NUM_USER_FEATURES]           — user vector (broadcast via L1)
 *   1: d_items [n_items × NUM_ITEM_FEATURES] — item matrix
 *   2: scores  [n_items]                      — output
 *   3: n_items (constant uint)
 */
kernel void score_kernel(
    device const float* d_user  [[buffer(0)]],
    device const float* d_items [[buffer(1)]],
    device float*       scores  [[buffer(2)]],
    constant uint&      n_items [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {{
    if (idx >= n_items) return;

    float f[NUM_FEATURES];
    for (uint i = 0; i < NUM_USER_FEATURES; i++)
        f[i] = d_user[i];
    for (uint i = 0; i < NUM_ITEM_FEATURES; i++)
        f[NUM_USER_FEATURES + i] = d_items[idx * NUM_ITEM_FEATURES + i];

    float raw = BASE_SCORE + score_all_trees(f);
    scores[idx] = {transform};
}}
"""

    def _emit_metal_topk_kernel(self, transform: str) -> str:
        return f"""
/* ==== Metal Fused Score + Per-Threadgroup Top-K Kernel ==== */
/*
 * Mirrors the CUDA split-feature top-K kernel exactly.
 *
 * Phase 1 (this kernel):
 *   Each threadgroup processes a range of items. Threads score items in
 *   parallel (one item per thread per batch). After each batch, thread 0
 *   merges results into a sorted top-K list in threadgroup shared memory.
 *   Each threadgroup writes its local top-K candidates to global output.
 *
 * Phase 2 (host — score_user_topk()):
 *   Merge grid_size × K threadgroup candidates via qsort.
 *
 * Threadgroup memory layout (threadgroup(0), size set by host):
 *   float  s_topk_scores  [k]              — sorted descending
 *   int    s_topk_indices [k]              — corresponding item indices
 *   float  s_batch_scores [TOPK_BLOCK_SIZE]— current batch scores
 *   int    s_batch_indices[TOPK_BLOCK_SIZE]— current batch item indices
 *
 * params buffer: [n_items, k]
 */
kernel void score_topk_kernel(
    device const float* d_user             [[buffer(0)]],
    device const float* d_items            [[buffer(1)]],
    device float*       block_topk_scores  [[buffer(2)]],
    device int*         block_topk_indices [[buffer(3)]],
    constant int*       params             [[buffer(4)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint bid        [[threadgroup_position_in_grid]],
    uint block_size [[threads_per_threadgroup]],
    uint grid_size  [[threadgroups_per_grid]],
    threadgroup char*   smem               [[threadgroup(0)]]
) {{
    int n_items = params[0];
    int k       = params[1];

    threadgroup float* s_topk_scores  = (threadgroup float*)smem;
    threadgroup int*   s_topk_indices = (threadgroup int*)(s_topk_scores + k);
    threadgroup float* s_batch_scores  = (threadgroup float*)(s_topk_indices + k);
    threadgroup int*   s_batch_indices = (threadgroup int*)(s_batch_scores + block_size);

    int items_per_block = (n_items + (int)grid_size - 1) / (int)grid_size;
    int block_start = (int)bid * items_per_block;
    int block_end   = block_start + items_per_block;
    if (block_end > n_items) block_end = n_items;

    /* Cache user features in thread-local registers */
    float user[NUM_USER_FEATURES];
    for (uint i = 0; i < NUM_USER_FEATURES; i++)
        user[i] = d_user[i];

    /* Initialize top-K with sentinel values */
    for (int i = (int)tid; i < k; i += (int)block_size) {{
        s_topk_scores[i]  = -1e30f;
        s_topk_indices[i] = -1;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Process items in batches of block_size */
    for (int batch = block_start; batch < block_end; batch += (int)block_size) {{
        int idx   = batch + (int)tid;
        float score = -1e30f;

        if (idx < block_end) {{
            float f[NUM_FEATURES];
            for (uint i = 0; i < NUM_USER_FEATURES; i++)
                f[i] = user[i];
            for (uint i = 0; i < NUM_ITEM_FEATURES; i++)
                f[NUM_USER_FEATURES + i] = d_items[idx * NUM_ITEM_FEATURES + i];
            float raw = BASE_SCORE + score_all_trees(f);
            score = {transform};
        }}

        s_batch_scores[tid]  = score;
        s_batch_indices[tid] = idx;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* Thread 0: merge this batch into sorted top-K (insertion sort) */
        if (tid == 0) {{
            int batch_size = block_end - batch;
            if (batch_size > (int)block_size) batch_size = (int)block_size;
            float min_topk = s_topk_scores[k - 1];

            for (int i = 0; i < batch_size; i++) {{
                float s = s_batch_scores[i];
                if (s > min_topk) {{
                    /* Binary search for insertion position (descending) */
                    int lo = 0, hi = k - 1;
                    while (lo < hi) {{
                        int mid = (lo + hi) >> 1;
                        if (s_topk_scores[mid] >= s) lo = mid + 1;
                        else                          hi = mid;
                    }}
                    /* Shift elements down to make room */
                    for (int j = k - 1; j > lo; j--) {{
                        s_topk_scores[j]  = s_topk_scores[j - 1];
                        s_topk_indices[j] = s_topk_indices[j - 1];
                    }}
                    s_topk_scores[lo]  = s;
                    s_topk_indices[lo] = s_batch_indices[i];
                    min_topk = s_topk_scores[k - 1];
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    /* Write this threadgroup's top-K to global output */
    int base = (int)bid * k;
    for (int i = (int)tid; i < k; i += (int)block_size) {{
        block_topk_scores[base + i]  = s_topk_scores[i];
        block_topk_indices[base + i] = s_topk_indices[i];
    }}
}}
"""

    def _generate_metal_host_header(self, user_fc: int) -> str:
        """
        Generate the Obj-C++ header-only library.

        Embeds the Metal shader as a C++11 raw string so the file is
        self-contained — no separate .metallib required at runtime.
        The shader is compiled once via [MTLDevice newLibraryWithSource:].

        The public API mirrors the CUDA split-feature API exactly:
          scoring_context_create / scoring_context_destroy
          load_items / score_user / score_user_topk
        so call-sites can swap backends with a single #include change.
        """
        item_fc = self.model_info.num_features - user_fc
        shader_content = self._generate_metal_shader(user_fc)

        # Use a raw-string delimiter unlikely to appear in generated shader code
        DELIM = "METALSHADER_END"

        part1 = (
            f"#ifndef SCORING_SPLIT_METAL_CORE_H\n"
            f"#define SCORING_SPLIT_METAL_CORE_H\n"
            f"\n"
            f"/*\n"
            f" * Auto-generated Metal (Apple Silicon GPU) host library\n"
            f" * for XGBoost split-feature scoring.\n"
            f" *\n"
            f" * Include this header in a .mm (Objective-C++) translation unit.\n"
            f" * Compile with:  clang++ -fobjc-arc -std=c++17 \\\n"
            f" *                  -framework Metal -framework Foundation\n"
            f" *\n"
            f" * Public API (identical to the CUDA split-feature API):\n"
            f" *   ScoringContext* scoring_context_create(int max_items)\n"
            f" *   void scoring_context_destroy(ScoringContext* ctx)\n"
            f" *   int  load_items(ctx, h_items, n_items)\n"
            f" *   int  score_user(ctx, h_user, h_scores)\n"
            f" *   int  score_user_topk(ctx, h_user, k, h_topk_indices, h_topk_scores)\n"
            f" *\n"
            f" * On Apple Silicon, all buffers use MTLResourceStorageModeShared\n"
            f" * (zero-copy unified memory).  load_items() and score results are\n"
            f" * transferred via memcpy — no PCIe bus involved.\n"
            f" *\n"
            f" * DO NOT EDIT — auto-generated by cuda_codegen.\n"
            f" */\n"
            f"\n"
            f"#import <Metal/Metal.h>\n"
            f"#import <Foundation/Foundation.h>\n"
            f"#include <stdio.h>\n"
            f"#include <stdlib.h>\n"
            f"#include <string.h>\n"
            f"#include <math.h>\n"
            f"\n"
            f"#define NUM_TREES         {self.model_info.num_trees}\n"
            f"#define NUM_FEATURES      {self.model_info.num_features}\n"
            f"#define NUM_USER_FEATURES {user_fc}\n"
            f"#define NUM_ITEM_FEATURES {item_fc}\n"
            f"#define BASE_SCORE        {self.model_info.base_score:.8f}f\n"
            f"#define TOPK_BLOCK_SIZE   256\n"
            f"#define TOPK_MAX_BLOCKS   4096\n"
        )

        part2 = (
            "\n/* Embedded Metal shader — compiled at context creation time. */\n"
            f'static const char* METAL_SHADER_SOURCE = R"{DELIM}(\n'
            + shader_content + "\n"
            + f'){DELIM}";\n'
        )

        part3 = r"""
/* Result type for top-K host-side merge sort */
typedef struct {
    float score;
    int   index;
} ScoredItem;

static int scored_item_cmp_desc(const void* a, const void* b) {
    float sa = ((const ScoredItem*)a)->score;
    float sb = ((const ScoredItem*)b)->score;
    if (sa > sb) return -1;
    if (sa < sb) return  1;
    return 0;
}

/* ---------------------------------------------------------------------------
 * Internal Objective-C context — holds all Metal state with ARC ownership.
 * Exposed to callers as an opaque ScoringContext* via __bridge_retained.
 * --------------------------------------------------------------------------- */
@interface _MetalScoringCtx : NSObject
@property (nonatomic, strong) id<MTLDevice>               device;
@property (nonatomic, strong) id<MTLCommandQueue>         commandQueue;
@property (nonatomic, strong) id<MTLComputePipelineState> scorePipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> topkPipeline;
@property (nonatomic, strong) id<MTLBuffer>               userBuffer;
@property (nonatomic, strong) id<MTLBuffer>               itemBuffer;
@property (nonatomic, strong) id<MTLBuffer>               scoreBuffer;
@property (nonatomic, strong) id<MTLBuffer>               topkScoreBuffer;
@property (nonatomic, strong) id<MTLBuffer>               topkIndexBuffer;
@property (nonatomic, strong) id<MTLBuffer>               paramsBuffer;
@property (nonatomic) int capacity;
@property (nonatomic) int items_loaded;
@property (nonatomic) int topk_max_k;
@property (nonatomic) int topk_max_blocks;
@end
@implementation _MetalScoringCtx
@end

/* Opaque C handle — stored as void* so ARC bridge casts work correctly.
 * ScoringContext* is a void* alias; internally it holds a _MetalScoringCtx. */
typedef void ScoringContext;

/**
 * Create a Metal scoring context.
 *
 * Compiles the embedded Metal shader at runtime (one-time, ~50–200 ms).
 * All GPU buffers use MTLResourceStorageModeShared: zero-copy on Apple
 * Silicon (CPU and GPU share the same physical DRAM).
 *
 * @param max_items  Maximum number of items that will ever be scored
 * @return Opaque context pointer, or NULL on failure
 */
ScoringContext* scoring_context_create(int max_items) {
    _MetalScoringCtx* ctx = [_MetalScoringCtx new];

    ctx.device = MTLCreateSystemDefaultDevice();
    if (!ctx.device) {
        fprintf(stderr, "ERROR: No Metal GPU found (requires Apple Silicon or Metal-capable GPU)\n");
        return NULL;
    }
    fprintf(stderr, "Metal device: %s\n", [ctx.device.name UTF8String]);

    ctx.commandQueue = [ctx.device newCommandQueue];

    /* Compile embedded shader source */
    NSError* error = nil;
    NSString* src  = [NSString stringWithUTF8String:METAL_SHADER_SOURCE];
    MTLCompileOptions* opts = [MTLCompileOptions new];
    id<MTLLibrary> lib = [ctx.device newLibraryWithSource:src options:opts error:&error];
    if (!lib) {
        fprintf(stderr, "ERROR: Metal shader compilation failed:\n%s\n",
                [[error localizedDescription] UTF8String]);
        return NULL;
    }

    id<MTLFunction> scoreFunc = [lib newFunctionWithName:@"score_kernel"];
    id<MTLFunction> topkFunc  = [lib newFunctionWithName:@"score_topk_kernel"];
    if (!scoreFunc || !topkFunc) {
        fprintf(stderr, "ERROR: Metal kernel functions not found in compiled library\n");
        return NULL;
    }

    ctx.scorePipeline = [ctx.device newComputePipelineStateWithFunction:scoreFunc error:&error];
    ctx.topkPipeline  = [ctx.device newComputePipelineStateWithFunction:topkFunc  error:&error];
    if (!ctx.scorePipeline || !ctx.topkPipeline) {
        fprintf(stderr, "ERROR: Failed to create Metal pipeline states: %s\n",
                [[error localizedDescription] UTF8String]);
        return NULL;
    }

    ctx.capacity      = max_items;
    ctx.items_loaded  = 0;
    ctx.topk_max_k    = 1024;
    ctx.topk_max_blocks = (max_items + TOPK_BLOCK_SIZE - 1) / TOPK_BLOCK_SIZE;
    if (ctx.topk_max_blocks > TOPK_MAX_BLOCKS)
        ctx.topk_max_blocks = TOPK_MAX_BLOCKS;

    /* MTLResourceStorageModeShared: CPU and GPU share the same physical pages.
     * memcpy from host to [buffer contents] is all that's needed — no cudaMemcpy. */
    ctx.userBuffer  = [ctx.device newBufferWithLength:NUM_USER_FEATURES * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    ctx.itemBuffer  = [ctx.device newBufferWithLength:(size_t)max_items * NUM_ITEM_FEATURES * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    ctx.scoreBuffer = [ctx.device newBufferWithLength:(size_t)max_items * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    ctx.topkScoreBuffer = [ctx.device newBufferWithLength:
                           (size_t)ctx.topk_max_blocks * ctx.topk_max_k * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
    ctx.topkIndexBuffer = [ctx.device newBufferWithLength:
                           (size_t)ctx.topk_max_blocks * ctx.topk_max_k * sizeof(int)
                                                  options:MTLResourceStorageModeShared];
    ctx.paramsBuffer    = [ctx.device newBufferWithLength:4 * sizeof(int)
                                                  options:MTLResourceStorageModeShared];

    if (!ctx.userBuffer || !ctx.itemBuffer || !ctx.scoreBuffer ||
        !ctx.topkScoreBuffer || !ctx.topkIndexBuffer || !ctx.paramsBuffer) {
        fprintf(stderr, "ERROR: Failed to allocate Metal buffers\n");
        return NULL;
    }

    /* Transfer ownership to caller; must be released via scoring_context_destroy(). */
    return (__bridge_retained ScoringContext*)ctx;
}

void scoring_context_destroy(ScoringContext* ctx) {
    if (!ctx) return;
    /* __bridge_transfer gives ownership to ARC, which releases on scope exit. */
    _MetalScoringCtx* impl = (__bridge_transfer _MetalScoringCtx*)ctx;
    (void)impl;
}

/**
 * Load item feature vectors into GPU-accessible memory.
 * On Apple Silicon, this is a direct memcpy — no PCIe transfer.
 * Items remain resident in GPU memory across score_user() calls.
 *
 * @param ctx     Scoring context
 * @param h_items [n_items × NUM_ITEM_FEATURES] row-major float32
 * @param n_items Number of items
 * @return 0 on success, -1 on error
 */
int load_items(ScoringContext* ctx, const float* h_items, int n_items) {
    _MetalScoringCtx* impl = (__bridge _MetalScoringCtx*)ctx;
    if (n_items > impl.capacity) {
        fprintf(stderr, "ERROR: n_items (%d) exceeds capacity (%d)\n",
                n_items, impl.capacity);
        return -1;
    }
    memcpy([impl.itemBuffer contents], h_items,
           (size_t)n_items * NUM_ITEM_FEATURES * sizeof(float));
    impl.items_loaded = n_items;
    return 0;
}

/**
 * Score all loaded items against one user.
 * Only the user feature vector is written before dispatch.
 * On Apple Silicon: no PCIe transfers anywhere in this call.
 *
 * @param ctx      Context with items loaded via load_items()
 * @param h_user   [NUM_USER_FEATURES] user feature vector
 * @param h_scores [items_loaded] output probabilities
 * @return 0 on success, -1 on error
 */
int score_user(ScoringContext* ctx, const float* h_user, float* h_scores) {
    _MetalScoringCtx* impl = (__bridge _MetalScoringCtx*)ctx;
    if (impl.items_loaded <= 0) {
        fprintf(stderr, "ERROR: no items loaded (call load_items first)\n");
        return -1;
    }

    memcpy([impl.userBuffer contents], h_user, NUM_USER_FEATURES * sizeof(float));

    int n = impl.items_loaded;
    uint n_items_u = (uint)n;

    id<MTLCommandBuffer>         cmdBuf = [impl.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc    = [cmdBuf computeCommandEncoder];

    [enc setComputePipelineState:impl.scorePipeline];
    [enc setBuffer:impl.userBuffer  offset:0 atIndex:0];
    [enc setBuffer:impl.itemBuffer  offset:0 atIndex:1];
    [enc setBuffer:impl.scoreBuffer offset:0 atIndex:2];
    [enc setBytes:&n_items_u length:sizeof(uint) atIndex:3];

    /* dispatchThreads: handles non-multiples of threadgroup size automatically */
    MTLSize gridSize        = MTLSizeMake((NSUInteger)n, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [enc endEncoding];

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    /* Shared memory — direct read, no device-to-host copy needed */
    memcpy(h_scores, [impl.scoreBuffer contents], (size_t)n * sizeof(float));
    return 0;
}

/**
 * Score all loaded items and return only the top-K results.
 *
 * Uses a fused scoring + per-threadgroup top-K kernel:
 *   Phase 1: Each threadgroup scores its range and maintains a sorted
 *            local top-K in threadgroup shared memory (no global atomics).
 *   Phase 2: Host merges threadgroup candidates via qsort for global top-K.
 *
 * On Apple Silicon, phase 2 reads candidates directly from shared memory —
 * no device-to-host transfer needed.
 *
 * @param ctx             Context with items loaded via load_items()
 * @param h_user          [NUM_USER_FEATURES] user feature vector
 * @param k               Number of top results (1 ≤ k ≤ 1024)
 * @param h_topk_indices  Output: [k] item indices sorted by score descending
 * @param h_topk_scores   Output: [k] corresponding scores
 * @return 0 on success, -1 on error
 */
int score_user_topk(
    ScoringContext* ctx,
    const float*    h_user,
    int             k,
    int*            h_topk_indices,
    float*          h_topk_scores
) {
    _MetalScoringCtx* impl = (__bridge _MetalScoringCtx*)ctx;
    if (impl.items_loaded <= 0) {
        fprintf(stderr, "ERROR: no items loaded (call load_items first)\n");
        return -1;
    }
    if (k <= 0 || k > impl.topk_max_k) {
        fprintf(stderr, "ERROR: k=%d out of range [1, %d]\n", k, impl.topk_max_k);
        return -1;
    }

    memcpy([impl.userBuffer contents], h_user, NUM_USER_FEATURES * sizeof(float));

    int n = impl.items_loaded;
    int grid_size = (n + TOPK_BLOCK_SIZE - 1) / TOPK_BLOCK_SIZE;
    if (grid_size > impl.topk_max_blocks)
        grid_size = impl.topk_max_blocks;

    int* params = (int*)[impl.paramsBuffer contents];
    params[0] = n;
    params[1] = k;

    /* Threadgroup memory: (k + TOPK_BLOCK_SIZE) × (float + int) */
    NSUInteger smem_size = (NSUInteger)(k + TOPK_BLOCK_SIZE) * (sizeof(float) + sizeof(int));

    id<MTLCommandBuffer>         cmdBuf = [impl.commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc    = [cmdBuf computeCommandEncoder];

    [enc setComputePipelineState:impl.topkPipeline];
    [enc setBuffer:impl.userBuffer        offset:0 atIndex:0];
    [enc setBuffer:impl.itemBuffer        offset:0 atIndex:1];
    [enc setBuffer:impl.topkScoreBuffer   offset:0 atIndex:2];
    [enc setBuffer:impl.topkIndexBuffer   offset:0 atIndex:3];
    [enc setBuffer:impl.paramsBuffer      offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:smem_size atIndex:0];

    MTLSize tgroups = MTLSizeMake((NSUInteger)grid_size, 1, 1);
    MTLSize threads = MTLSizeMake(TOPK_BLOCK_SIZE, 1, 1);
    [enc dispatchThreadgroups:tgroups threadsPerThreadgroup:threads];
    [enc endEncoding];

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    /* Phase 2: host merge — identical algorithm to CUDA version */
    int n_candidates = grid_size * k;
    float* blk_scores  = (float*)[impl.topkScoreBuffer contents];
    int*   blk_indices = (int*)  [impl.topkIndexBuffer  contents];

    ScoredItem* candidates = (ScoredItem*)malloc((size_t)n_candidates * sizeof(ScoredItem));
    if (!candidates) {
        fprintf(stderr, "ERROR: malloc failed for candidates\n");
        return -1;
    }
    int valid = 0;
    for (int i = 0; i < n_candidates; i++) {
        if (blk_indices[i] >= 0) {
            candidates[valid].score = blk_scores[i];
            candidates[valid].index = blk_indices[i];
            valid++;
        }
    }
    qsort(candidates, (size_t)valid, sizeof(ScoredItem), scored_item_cmp_desc);

    int result_k = k < valid ? k : valid;
    for (int i = 0; i < result_k; i++) {
        h_topk_scores[i]  = candidates[i].score;
        h_topk_indices[i] = candidates[i].index;
    }
    for (int i = result_k; i < k; i++) {
        h_topk_scores[i]  = -1e30f;
        h_topk_indices[i] = -1;
    }
    free(candidates);
    return 0;
}

#endif /* SCORING_SPLIT_METAL_CORE_H */
"""
        return part1 + part2 + part3

    def _emit_metal_main_driver(self) -> str:
        """Emit the Metal standalone main() driver.

        The score_block logic is identical to the CUDA main driver because
        the Metal API has the same function signatures.  The only additions
        are the Obj-C++ header and @autoreleasepool.
        """
        score_block = (
            "    ScoringContext* ctx = scoring_context_create(n_items);\n"
            "    if (!ctx) { fprintf(stderr, \"Failed to create Metal context\\n\"); return 1; }\n"
            "    if (load_items(ctx, items, n_items) != 0) return 1;\n"
            "\n"
            "    if (topk > 0) {\n"
            "        int*   topk_indices = (int*)malloc(topk * sizeof(int));\n"
            "        float* topk_scores  = (float*)malloc(topk * sizeof(float));\n"
            "        if (!topk_indices || !topk_scores) { perror(\"malloc topk\"); return 1; }\n"
            "        if (score_user_topk(ctx, user, topk, topk_indices, topk_scores) != 0) return 1;\n"
            "        scoring_context_destroy(ctx);\n"
            "        if (output_path) {\n"
            "            FILE* fout = fopen(output_path, \"wb\");\n"
            "            if (!fout) { perror(\"fopen output\"); return 1; }\n"
            "            fwrite(topk_indices, sizeof(int),   topk, fout);\n"
            "            fwrite(topk_scores,  sizeof(float), topk, fout);\n"
            "            fclose(fout);\n"
            "            fprintf(stderr, \"Wrote top-%d results to %s\\n\", topk, output_path);\n"
            "        } else {\n"
            "            for (int i = 0; i < topk; i++)\n"
            "                printf(\"rank[%d] = item[%d]  score=%.8f\\n\", i, topk_indices[i], topk_scores[i]);\n"
            "        }\n"
            "        free(topk_indices); free(topk_scores);\n"
            "    } else {\n"
            "        if (score_user(ctx, user, scores) != 0) return 1;\n"
            "        scoring_context_destroy(ctx);\n"
            "        if (output_path) {\n"
            "            FILE* fout = fopen(output_path, \"wb\");\n"
            "            if (!fout) { perror(\"fopen output\"); return 1; }\n"
            "            fwrite(scores, sizeof(float), (size_t)n_items, fout);\n"
            "            fclose(fout);\n"
            "            fprintf(stderr, \"Wrote %d scores to %s\\n\", n_items, output_path);\n"
            "        } else {\n"
            "            int limit = n_items < 20 ? n_items : 20;\n"
            "            for (int i = 0; i < limit; i++)\n"
            "                printf(\"item[%d] = %.8f\\n\", i, scores[i]);\n"
            "            if (n_items > 20)\n"
            "                printf(\"... (%d more items)\\n\", n_items - 20);\n"
            "        }\n"
            "    }\n"
        )

        return f"""#include "scoring_split_metal_core.h"

/* ==== Standalone Entry Point ==== */

int main(int argc, char** argv) {{
    @autoreleasepool {{

    if (argc < 3) {{
        fprintf(stderr,
            "Usage: %s <user.bin> <items.bin> [output.bin] [-k topK]\\n"
            "\\n"
            "  user.bin:   binary float32 [%d] (one user feature vector)\\n"
            "  items.bin:  binary float32 [N x %d] (item features, row-major)\\n"
            "  output.bin: binary output (omit to print first 20 scores)\\n"
            "  -k topK:    return only the top K results (ranked by score)\\n",
            argv[0], NUM_USER_FEATURES, NUM_ITEM_FEATURES);
        return 1;
    }}

    const char* user_path  = argv[1];
    const char* items_path = argv[2];
    const char* output_path = NULL;
    int topk = 0;

    for (int a = 3; a < argc; a++) {{
        if (strcmp(argv[a], "-k") == 0 && a + 1 < argc) {{
            topk = atoi(argv[++a]);
        }} else if (!output_path) {{
            output_path = argv[a];
        }}
    }}

    FILE* fu = fopen(user_path, "rb");
    if (!fu) {{ perror("fopen user"); return 1; }}
    float user[NUM_USER_FEATURES];
    if (fread(user, sizeof(float), NUM_USER_FEATURES, fu) != NUM_USER_FEATURES) {{
        fprintf(stderr, "Failed to read %d user features\\n", NUM_USER_FEATURES);
        fclose(fu); return 1;
    }}
    fclose(fu);

    FILE* fi = fopen(items_path, "rb");
    if (!fi) {{ perror("fopen items"); return 1; }}
    fseek(fi, 0, SEEK_END);
    long fsize = ftell(fi);
    fseek(fi, 0, SEEK_SET);
    int n_items = (int)(fsize / ((long)NUM_ITEM_FEATURES * (long)sizeof(float)));
    if (n_items <= 0) {{ fprintf(stderr, "Invalid items file\\n"); fclose(fi); return 1; }}

    fprintf(stderr, "Scoring %d items (%d user + %d item features)...\\n",
            n_items, NUM_USER_FEATURES, NUM_ITEM_FEATURES);

    float* items  = (float*)malloc((size_t)fsize);
    float* scores = (float*)malloc((size_t)n_items * sizeof(float));
    if (!items || !scores) {{ perror("malloc"); return 1; }}
    fread(items, sizeof(float), (size_t)n_items * NUM_ITEM_FEATURES, fi);
    fclose(fi);

{score_block}
    free(items); free(scores);
    }} /* @autoreleasepool */
    return 0;
}}
"""

    def _emit_metal_bench_driver(self, user_fc: int) -> str:
        feat_arrays = self._emit_feature_range_arrays()
        lines = []
        a = lines.append
        a('#include "scoring_split_metal_core.h"')
        a('#include <time.h>')
        a('')
        a(feat_arrays)
        a('')
        a('static double get_time_ms(void) {')
        a('    struct timespec ts;')
        a('    clock_gettime(CLOCK_MONOTONIC, &ts);')
        a('    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;')
        a('}')
        a('')
        a('static float randf_range(unsigned int *state, float lo, float hi) {')
        a('    *state = *state * 1103515245u + 12345u;')
        a('    float t = (float)(*state >> 16) / 65535.0f;')
        a('    return lo + t * (hi - lo);')
        a('}')
        a('')
        a('int main(int argc, char** argv) {')
        a('    @autoreleasepool {')
        a('    int n_items    = 1000000;')
        a('    int topk       = 100;')
        a('    int warmup     = 3;')
        a('    int iterations = 20;')
        a('    unsigned int seed = 42;')
        a('')
        a('    for (int i = 1; i < argc; i++) {')
        a('        if      (strcmp(argv[i], "-n") == 0 && i+1 < argc) n_items    = atoi(argv[++i]);')
        a('        else if (strcmp(argv[i], "-k") == 0 && i+1 < argc) topk       = atoi(argv[++i]);')
        a('        else if (strcmp(argv[i], "-w") == 0 && i+1 < argc) warmup     = atoi(argv[++i]);')
        a('        else if (strcmp(argv[i], "-i") == 0 && i+1 < argc) iterations = atoi(argv[++i]);')
        a('        else if (strcmp(argv[i], "-s") == 0 && i+1 < argc) seed       = (unsigned)atoi(argv[++i]);')
        a('        else {')
        a('            fprintf(stderr,')
        a('                "Usage: %s [-n items] [-k topK] [-w warmup] [-i iters] [-s seed]\\n"')
        a('                "\\n  Defaults: -n 1000000 -k 100 -w 3 -i 20 -s 42\\n", argv[0]);')
        a('            return (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) ? 0 : 1;')
        a('        }')
        a('    }')
        a('')
        a('    printf("=== Metal (Apple Silicon GPU) Split-Feature Scoring Benchmark ===\\n");')
        a('    printf("  Items:         %d\\n", n_items);')
        a('    printf("  Top-K:         %d\\n", topk);')
        a('    printf("  User features: %d\\n", NUM_USER_FEATURES);')
        a('    printf("  Item features: %d\\n", NUM_ITEM_FEATURES);')
        a('    printf("  Trees:         %d\\n", NUM_TREES);')
        a('    printf("  Warmup:        %d\\n", warmup);')
        a('    printf("  Iterations:    %d\\n\\n", iterations);')
        a('')
        a('    /* Generate random data in realistic feature ranges */')
        a('    unsigned int rng = seed;')
        a('    float user[NUM_USER_FEATURES];')
        a('    for (int i = 0; i < NUM_USER_FEATURES; i++)')
        a('        user[i] = randf_range(&rng, FEAT_MIN[i], FEAT_MAX[i]);')
        a('')
        a('    size_t items_bytes = (size_t)n_items * NUM_ITEM_FEATURES * sizeof(float);')
        a('    float* items = (float*)malloc(items_bytes);')
        a('    if (!items) { perror("malloc items"); return 1; }')
        a('    for (int j = 0; j < n_items; j++)')
        a('        for (int f = 0; f < NUM_ITEM_FEATURES; f++)')
        a('            items[j * NUM_ITEM_FEATURES + f] =')
        a('                randf_range(&rng, FEAT_MIN[NUM_USER_FEATURES + f], FEAT_MAX[NUM_USER_FEATURES + f]);')
        a('    printf("Generated %.1f MB of random item features\\n", items_bytes / (1024.0 * 1024.0));')
        a('')
        a('    /* Create context (compiles Metal shader) */')
        a('    double t0 = get_time_ms();')
        a('    ScoringContext* ctx = scoring_context_create(n_items);')
        a('    if (!ctx) { fprintf(stderr, "Failed to create Metal context\\n"); return 1; }')
        a('    double compile_ms = get_time_ms() - t0;')
        a('    printf("Metal shader compile: %.0f ms\\n", compile_ms);')
        a('')
        a('    /* Load items (memcpy to unified memory — no PCIe on Apple Silicon) */')
        a('    t0 = get_time_ms();')
        a('    if (load_items(ctx, items, n_items) != 0) return 1;')
        a('    double load_ms = get_time_ms() - t0;')
        a('    printf("Item load (memcpy):   %10.2f ms  (%.1f GB/s)\\n\\n",')
        a('           load_ms, items_bytes / load_ms / 1e6);')
        a('')
        a('    float* scores   = (float*)malloc((size_t)n_items * sizeof(float));')
        a('    int*   topk_idx = (int*)malloc(topk * sizeof(int));')
        a('    float* topk_sc  = (float*)malloc(topk * sizeof(float));')
        a('    if (!scores || !topk_idx || !topk_sc) { perror("malloc"); return 1; }')
        a('')
        a('    /* Warmup */')
        a('    printf("Warming up (%d runs)...\\n", warmup);')
        a('    for (int w = 0; w < warmup; w++) {')
        a('        score_user(ctx, user, scores);')
        a('        score_user_topk(ctx, user, topk, topk_idx, topk_sc);')
        a('    }')
        a('')
        a('    /* Score verification */')
        a('    score_user(ctx, user, scores);')
        a('    float s_min = scores[0], s_max = scores[0];')
        a('    double s_sum = 0.0;')
        a('    for (int i = 0; i < n_items; i++) {')
        a('        if (scores[i] < s_min) s_min = scores[i];')
        a('        if (scores[i] > s_max) s_max = scores[i];')
        a('        s_sum += scores[i];')
        a('    }')
        a('    printf("--- Score Verification ---\\n");')
        a('    printf("  Min score:  %.8f\\n", s_min);')
        a('    printf("  Max score:  %.8f\\n", s_max);')
        a('    printf("  Mean score: %.8f\\n", s_sum / n_items);')
        a('    printf("  Score[0]:   %.8f\\n", scores[0]);')
        a('    printf("  Score[N/2]: %.8f\\n\\n", scores[n_items/2]);')
        a('')
        a('    /* Benchmark: full scoring */')
        a('    t0 = get_time_ms();')
        a('    for (int it = 0; it < iterations; it++)')
        a('        score_user(ctx, user, scores);')
        a('    double full_ms = (get_time_ms() - t0) / iterations;')
        a('')
        a('    /* Benchmark: top-K */')
        a('    t0 = get_time_ms();')
        a('    for (int it = 0; it < iterations; it++)')
        a('        score_user_topk(ctx, user, topk, topk_idx, topk_sc);')
        a('    double topk_ms = (get_time_ms() - t0) / iterations;')
        a('')
        a('    printf("\\n--- Results ---\\n");')
        a('    printf("  score_user():      %10.2f ms  (%8.2f M items/sec)\\n",')
        a('           full_ms, n_items / full_ms / 1000.0);')
        a('    printf("  score_user_topk(): %10.2f ms  (%8.2f M items/sec)\\n",')
        a('           topk_ms, n_items / topk_ms / 1000.0);')
        a('    printf("  Note: unified memory — no PCIe transfers on Apple Silicon\\n");')
        a('')
        a('    /* Top-K scaling */')
        a('    int k_values[] = {10, 50, 100, 500, 1000};')
        a('    int n_kv = 5;')
        a('    printf("\\n--- Top-K Scaling ---\\n");')
        a('    for (int ki = 0; ki < n_kv; ki++) {')
        a('        int k = k_values[ki];')
        a('        if (k > n_items || k > 1024) break;')
        a('        int*   ki_idx = (int*)malloc(k * sizeof(int));')
        a('        float* ki_sc  = (float*)malloc(k * sizeof(float));')
        a('        score_user_topk(ctx, user, k, ki_idx, ki_sc); /* warmup */')
        a('        t0 = get_time_ms();')
        a('        for (int it = 0; it < iterations; it++)')
        a('            score_user_topk(ctx, user, k, ki_idx, ki_sc);')
        a('        double k_ms = (get_time_ms() - t0) / iterations;')
        a('        printf("  K=%-5d  %10.2f ms  (%8.2f M items/sec)\\n",')
        a('               k, k_ms, n_items / k_ms / 1000.0);')
        a('        free(ki_idx); free(ki_sc);')
        a('    }')
        a('')
        a('    /* Top-K preview */')
        a('    printf("\\nTop-%d preview:\\n", topk < 10 ? topk : 10);')
        a('    int show = topk < 10 ? topk : 10;')
        a('    for (int i = 0; i < show; i++)')
        a('        printf("  rank[%d] = item[%d]  score=%.8f\\n", i, topk_idx[i], topk_sc[i]);')
        a('    if (topk > 10) printf("  ... (%d more)\\n", topk - 10);')
        a('')
        a('    scoring_context_destroy(ctx);')
        a('    free(items); free(scores);')
        a('    free(topk_idx); free(topk_sc);')
        a('    printf("\\nDone.\\n");')
        a('    } /* @autoreleasepool */')
        a('    return 0;')
        a('}')
        return "\n".join(lines)

    @staticmethod
    def _emit_metal_makefile() -> str:
        return """# Auto-generated Makefile for Metal (Apple Silicon GPU) split-feature scoring
#
# Requires: macOS with Xcode command-line tools (xcrun, clang++)
#           Apple Silicon Mac (M1/M2/M3/M4) for best performance
#
# Targets:
#   make -f Makefile.metal bench     — build & run Metal benchmark
#   make -f Makefile.metal main      — build Metal main driver
#   make -f Makefile.metal metallib  — precompile .metal → .metallib
#   make -f Makefile.metal clean
#
# Override build dir:  make -f Makefile.metal BUILD_DIR=./out bench

CXX       ?= clang++
CXXFLAGS  ?= -O2 -std=c++17 -fobjc-arc
FRAMEWORKS = -framework Metal -framework Foundation
BUILD_DIR ?= ../build

# Detect xcrun (required for Metal shader precompilation)
HAS_XCRUN := $(shell command -v xcrun 2>/dev/null)

.PHONY: all bench main metallib clean

all: bench main

# ---- Main driver ----

main: $(BUILD_DIR)/scoring_split_metal

$(BUILD_DIR)/scoring_split_metal: scoring_split_main_metal.mm scoring_split_metal_core.h
\t@mkdir -p $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o $@ scoring_split_main_metal.mm
\t@echo "Built: $@"

# ---- Benchmark ----

bench: $(BUILD_DIR)/scoring_split_bench_metal
\t$(BUILD_DIR)/scoring_split_bench_metal $(BENCH_ARGS)

$(BUILD_DIR)/scoring_split_bench_metal: scoring_split_bench_metal.mm scoring_split_metal_core.h
\t@mkdir -p $(BUILD_DIR)
\t$(CXX) $(CXXFLAGS) $(FRAMEWORKS) -o $@ scoring_split_bench_metal.mm
\t@echo "Built: $@"

# ---- Precompile Metal shader (optional — for Xcode / production use) ----

metallib: $(BUILD_DIR)/scoring_split_core.metallib

$(BUILD_DIR)/scoring_split_core.metallib: scoring_split_core.metal
\t@if [ -z "$(HAS_XCRUN)" ]; then echo "ERROR: xcrun not found"; exit 1; fi
\t@mkdir -p $(BUILD_DIR)
\txcrun -sdk macosx metal -c scoring_split_core.metal -o $(BUILD_DIR)/scoring_split_core.air
\txcrun -sdk macosx metallib $(BUILD_DIR)/scoring_split_core.air -o $@
\t@echo "Built: $@"

clean:
\trm -f $(BUILD_DIR)/scoring_split_metal
\trm -f $(BUILD_DIR)/scoring_split_bench_metal
\trm -f $(BUILD_DIR)/scoring_split_core.air
\trm -f $(BUILD_DIR)/scoring_split_core.metallib
"""
