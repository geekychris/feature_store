#!/usr/bin/env bash
# Verify that generated C scoring code matches XGBoost Python predictions.
#
# This generates random test data, scores with both XGBoost Python and the
# compiled C code, and compares outputs. No CUDA hardware required.
#
# Usage:
#   ./scripts/verify_scoring.sh [model_path] [n_samples]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="${1:-${PROJECT_DIR}/models/model.ubj}"
METADATA_PATH="${PROJECT_DIR}/models/training_result.json"
N_SAMPLES="${2:-1000}"

echo "=== Verification: XGBoost Python vs Generated C ==="
echo

PYTHONPATH="${PROJECT_DIR}/python" python3 -m cuda_codegen verify \
    --model "$MODEL_PATH" \
    --metadata "$METADATA_PATH" \
    --n-samples "$N_SAMPLES"
