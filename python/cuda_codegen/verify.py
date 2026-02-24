"""
Verification: compare generated C scoring code against XGBoost Python predictions.

Workflow:
  1. Load the XGBoost model in Python
  2. Generate random test feature vectors (with optional NaN injection)
  3. Score with XGBoost Python → ground truth
  4. Generate CPU C code, compile with gcc/clang, run on test vectors
  5. Compare outputs within tolerance

This allows correctness verification on any machine without CUDA.
"""

import logging
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb

from .generator import CudaCodeGenerator

logger = logging.getLogger(__name__)


def verify(
    model_path: str,
    n_samples: int = 1000,
    metadata_path: Optional[str] = None,
    tolerance: float = 1e-5,
    nan_fraction: float = 0.05,
    cc: str = "gcc",
    user_feature_count: Optional[int] = None,
    topk: Optional[int] = None,
) -> dict:
    """
    End-to-end verification of generated scoring code.

    Generates CPU C code from the model, compiles it, runs it on random
    test data, and compares against XGBoost Python predictions.

    Args:
        model_path: Path to XGBoost model file
        n_samples: Number of random test samples
        metadata_path: Optional path to training_result.json
        tolerance: Maximum allowed absolute difference per sample
        nan_fraction: Fraction of feature values to set to NaN (tests missing handling)
        cc: C compiler to use (gcc, clang, cc)
        user_feature_count: If set, verify split-feature mode (0..N-1 user, N..end item)
        topk: If set, verify top-K ranking mode (requires user_feature_count)

    Returns:
        dict with keys: success, n_samples, max_diff, mean_diff, tolerance,
                        expected_range, actual_range, and optionally error
    """
    # --- Build the generator (which also loads the model) ---
    gen = CudaCodeGenerator(model_path, metadata_path=metadata_path)
    num_features = gen.model_info.num_features

    if user_feature_count and topk:
        return _verify_split_topk(gen, n_samples, tolerance, nan_fraction, cc,
                                  user_feature_count, topk)

    if user_feature_count:
        return _verify_split(gen, n_samples, tolerance, nan_fraction, cc,
                             user_feature_count)

    # --- Standard mode ---
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, num_features).astype(np.float32)

    # Inject some NaNs to test missing-value handling
    if nan_fraction > 0:
        n_nan = int(n_samples * num_features * nan_fraction)
        nan_rows = rng.randint(0, n_samples, size=n_nan)
        nan_cols = rng.randint(0, num_features, size=n_nan)
        X[nan_rows, nan_cols] = np.nan
        logger.info("Injected %d NaN values (%.1f%%)", n_nan, nan_fraction * 100)

    # Ground truth from XGBoost Python
    dmat = xgb.DMatrix(X)
    expected = gen.booster.predict(dmat).astype(np.float32)

    # Generate, compile, and run C code
    with tempfile.TemporaryDirectory(prefix="cuda_codegen_verify_") as tmpdir:
        tmpdir = Path(tmpdir)
        c_path = tmpdir / "scoring_kernel.c"
        gen.generate_cpu(str(c_path))

        binary_path = tmpdir / "scoring_test"
        compile_cmd = [cc, "-O2", "-Wall", "-o", str(binary_path), str(c_path), "-lm"]
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {"success": False, "error": f"Compilation failed:\n{result.stderr}"}

        input_path = tmpdir / "features.bin"
        X.tofile(str(input_path))
        output_path = tmpdir / "scores.bin"

        result = subprocess.run(
            [str(binary_path), str(input_path), str(output_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return {"success": False, "error": f"Execution failed:\n{result.stderr}"}

        actual = np.fromfile(str(output_path), dtype=np.float32)

    return _compare(expected, actual, n_samples, nan_fraction, tolerance)


def _verify_split(
    gen: CudaCodeGenerator,
    n_items: int,
    tolerance: float,
    nan_fraction: float,
    cc: str,
    user_fc: int,
) -> dict:
    """
    Verify split-feature mode.

    Generates one random user vector and n_items random item vectors.
    Constructs the full feature matrix (by tiling user features across rows),
    scores with XGBoost Python, then compares against the compiled split C code.
    """
    num_features = gen.model_info.num_features
    item_fc = num_features - user_fc

    rng = np.random.RandomState(42)
    user = rng.randn(user_fc).astype(np.float32)
    items = rng.randn(n_items, item_fc).astype(np.float32)

    # Inject NaNs into item features
    if nan_fraction > 0:
        n_nan = int(n_items * item_fc * nan_fraction)
        nan_rows = rng.randint(0, n_items, size=n_nan)
        nan_cols = rng.randint(0, item_fc, size=n_nan)
        items[nan_rows, nan_cols] = np.nan

    # Construct full feature matrix for XGBoost ground truth
    user_tiled = np.tile(user, (n_items, 1))  # [n_items, user_fc]
    X = np.hstack([user_tiled, items])         # [n_items, num_features]

    dmat = xgb.DMatrix(X)
    expected = gen.booster.predict(dmat).astype(np.float32)

    # Generate, compile, and run split C code
    with tempfile.TemporaryDirectory(prefix="cuda_codegen_verify_split_") as tmpdir:
        tmpdir = Path(tmpdir)
        c_path = tmpdir / "scoring_split.c"
        gen.generate_cpu_split(str(c_path), user_fc)

        binary_path = tmpdir / "scoring_split_test"
        compile_cmd = [cc, "-O2", "-Wall", "-o", str(binary_path), str(c_path), "-lm"]
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {"success": False, "error": f"Compilation failed:\n{result.stderr}"}

        user_path = tmpdir / "user.bin"
        items_path = tmpdir / "items.bin"
        output_path = tmpdir / "scores.bin"

        user.tofile(str(user_path))
        items.tofile(str(items_path))

        result = subprocess.run(
            [str(binary_path), str(user_path), str(items_path), str(output_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return {"success": False, "error": f"Execution failed:\n{result.stderr}"}

        actual = np.fromfile(str(output_path), dtype=np.float32)

    return _compare(expected, actual, n_items, nan_fraction, tolerance)


def _verify_split_topk(
    gen: CudaCodeGenerator,
    n_items: int,
    tolerance: float,
    nan_fraction: float,
    cc: str,
    user_fc: int,
    topk: int,
) -> dict:
    """
    Verify split-feature top-K mode.

    Generates one random user and n_items random items, scores with XGBoost,
    sorts to find true top-K, then compares against the compiled C code's
    top-K output (which uses qsort on the CPU path).
    """
    num_features = gen.model_info.num_features
    item_fc = num_features - user_fc

    rng = np.random.RandomState(42)
    user = rng.randn(user_fc).astype(np.float32)
    items = rng.randn(n_items, item_fc).astype(np.float32)

    # Inject NaNs into item features
    if nan_fraction > 0:
        n_nan = int(n_items * item_fc * nan_fraction)
        nan_rows = rng.randint(0, n_items, size=n_nan)
        nan_cols = rng.randint(0, item_fc, size=n_nan)
        items[nan_rows, nan_cols] = np.nan

    # Construct full feature matrix for XGBoost ground truth
    user_tiled = np.tile(user, (n_items, 1))
    X = np.hstack([user_tiled, items])

    dmat = xgb.DMatrix(X)
    all_scores = gen.booster.predict(dmat).astype(np.float32)

    # Expected top-K: sort by score descending
    sorted_indices = np.argsort(-all_scores)
    k = min(topk, n_items)
    expected_indices = sorted_indices[:k].astype(np.int32)
    expected_scores = all_scores[expected_indices]

    # Generate, compile, and run split C code with -k flag
    with tempfile.TemporaryDirectory(prefix="cuda_codegen_verify_topk_") as tmpdir:
        tmpdir = Path(tmpdir)
        c_path = tmpdir / "scoring_split_topk.c"
        gen.generate_cpu_split(str(c_path), user_fc)

        binary_path = tmpdir / "scoring_split_topk_test"
        compile_cmd = [cc, "-O2", "-Wall", "-o", str(binary_path), str(c_path), "-lm"]
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {"success": False, "error": f"Compilation failed:\n{result.stderr}"}

        user_path = tmpdir / "user.bin"
        items_path = tmpdir / "items.bin"
        output_path = tmpdir / "topk.bin"

        user.tofile(str(user_path))
        items.tofile(str(items_path))

        result = subprocess.run(
            [str(binary_path), str(user_path), str(items_path),
             str(output_path), "-k", str(topk)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return {"success": False, "error": f"Execution failed:\n{result.stderr}"}

        # Read binary output: K ints (indices) followed by K floats (scores)
        raw = np.fromfile(str(output_path), dtype=np.uint8)
        int_bytes = topk * 4
        float_bytes = topk * 4
        if len(raw) < int_bytes + float_bytes:
            return {
                "success": False,
                "error": f"Output too small: {len(raw)} bytes, expected {int_bytes + float_bytes}",
            }
        actual_indices = np.frombuffer(raw[:int_bytes], dtype=np.int32)
        actual_scores = np.frombuffer(raw[int_bytes:int_bytes + float_bytes], dtype=np.float32)

    # Compare top-K: check that the index sets match and scores are close
    # (Indices at boundaries with identical scores may differ in order)
    score_diffs = np.abs(expected_scores - actual_scores)
    max_diff = float(np.max(score_diffs))
    mean_diff = float(np.mean(score_diffs))

    # Check index agreement: compare sets for the top k
    expected_set = set(expected_indices.tolist())
    actual_set = set(actual_indices[actual_indices >= 0].tolist())
    index_agreement = len(expected_set & actual_set) / k if k > 0 else 1.0

    # Score-based pass: even if indices differ at tie boundaries, scores must match
    passed = max_diff < tolerance

    return {
        "success": passed,
        "n_samples": n_items,
        "topk": topk,
        "nan_fraction": nan_fraction,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "tolerance": tolerance,
        "index_agreement": index_agreement,
        "expected_range": [float(expected_scores.min()), float(expected_scores.max())],
        "actual_range": [float(actual_scores.min()), float(actual_scores.max())],
    }


def _compare(
    expected: np.ndarray,
    actual: np.ndarray,
    n_samples: int,
    nan_fraction: float,
    tolerance: float,
) -> dict:
    """Compare expected vs actual scores and return result dict."""
    if len(actual) != len(expected):
        return {
            "success": False,
            "error": f"Output size mismatch: expected {len(expected)}, got {len(actual)}",
        }

    diffs = np.abs(expected - actual)
    max_diff = float(np.max(diffs))
    mean_diff = float(np.mean(diffs))
    passed = max_diff < tolerance

    worst_indices = np.argsort(diffs)[-5:][::-1]
    worst_samples = [
        {
            "index": int(idx),
            "expected": float(expected[idx]),
            "actual": float(actual[idx]),
            "diff": float(diffs[idx]),
        }
        for idx in worst_indices
    ]

    return {
        "success": passed,
        "n_samples": n_samples,
        "nan_fraction": nan_fraction,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "tolerance": tolerance,
        "expected_range": [float(expected.min()), float(expected.max())],
        "actual_range": [float(actual.min()), float(actual.max())],
        "worst_samples": worst_samples,
    }
