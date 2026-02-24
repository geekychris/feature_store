#!/usr/bin/env bash
# Generate CUDA and CPU scoring code from a trained XGBoost model.
#
# Usage:
#   ./scripts/generate_cuda.sh [model_path] [metadata_path]
#
# Defaults:
#   model_path:    models/model.ubj
#   metadata_path: models/training_result.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="${1:-${PROJECT_DIR}/models/model.ubj}"
METADATA_PATH="${2:-${PROJECT_DIR}/models/training_result.json}"
OUTPUT_DIR="${PROJECT_DIR}/generated"

mkdir -p "$OUTPUT_DIR"

echo "=== XGBoost → CUDA/C Code Generator ==="
echo "  Model:    $MODEL_PATH"
echo "  Metadata: $METADATA_PATH"
echo "  Output:   $OUTPUT_DIR"
echo

# Generate CUDA version
echo "--- Generating CUDA kernel ---"
PYTHONPATH="${PROJECT_DIR}/python" python3 -m cuda_codegen generate \
    --model "$MODEL_PATH" \
    --metadata "$METADATA_PATH" \
    --output "$OUTPUT_DIR/scoring_kernel.cu"

echo

# Generate CPU version (for testing without CUDA hardware)
echo "--- Generating CPU version ---"
PYTHONPATH="${PROJECT_DIR}/python" python3 -m cuda_codegen generate \
    --model "$MODEL_PATH" \
    --metadata "$METADATA_PATH" \
    --output "$OUTPUT_DIR/scoring_kernel.c" \
    --cpu

echo
echo "=== Generated files ==="
ls -lh "$OUTPUT_DIR"/scoring_kernel.*
