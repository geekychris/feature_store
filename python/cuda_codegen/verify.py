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

    Returns:
        dict with keys: success, n_samples, max_diff, mean_diff, tolerance,
                        expected_range, actual_range, and optionally error
    """
    # --- Build the generator (which also loads the model) ---
    gen = CudaCodeGenerator(model_path, metadata_path=metadata_path)
    num_features = gen.model_info.num_features

    # --- Generate random test data ---
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, num_features).astype(np.float32)

    # Inject some NaNs to test missing-value handling
    if nan_fraction > 0:
        n_nan = int(n_samples * num_features * nan_fraction)
        nan_rows = rng.randint(0, n_samples, size=n_nan)
        nan_cols = rng.randint(0, num_features, size=n_nan)
        X[nan_rows, nan_cols] = np.nan
        logger.info("Injected %d NaN values (%.1f%%)", n_nan, nan_fraction * 100)

    # --- Ground truth from XGBoost Python ---
    dmat = xgb.DMatrix(X)
    expected = gen.booster.predict(dmat).astype(np.float32)

    # --- Generate, compile, and run C code ---
    with tempfile.TemporaryDirectory(prefix="cuda_codegen_verify_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Generate CPU C code
        c_path = tmpdir / "scoring_kernel.c"
        gen.generate_cpu(str(c_path))

        # Compile
        binary_path = tmpdir / "scoring_test"
        compile_cmd = [cc, "-O2", "-Wall", "-o", str(binary_path), str(c_path), "-lm"]
        logger.info("Compiling: %s", " ".join(compile_cmd))

        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {
                "success": False,
                "error": f"Compilation failed:\n{result.stderr}",
                "compile_cmd": " ".join(compile_cmd),
            }

        # Write test features to binary file
        input_path = tmpdir / "features.bin"
        X.tofile(str(input_path))

        output_path = tmpdir / "scores.bin"

        # Run the compiled binary
        run_cmd = [str(binary_path), str(input_path), str(output_path)]
        logger.info("Running: %s", " ".join(run_cmd))

        result = subprocess.run(run_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {
                "success": False,
                "error": f"Execution failed:\n{result.stderr}\n{result.stdout}",
            }

        # Read results
        actual = np.fromfile(str(output_path), dtype=np.float32)

    if len(actual) != len(expected):
        return {
            "success": False,
            "error": (
                f"Output size mismatch: expected {len(expected)}, got {len(actual)}"
            ),
        }

    # --- Compare ---
    diffs = np.abs(expected - actual)
    max_diff = float(np.max(diffs))
    mean_diff = float(np.mean(diffs))
    passed = max_diff < tolerance

    # Find worst samples for debugging
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
