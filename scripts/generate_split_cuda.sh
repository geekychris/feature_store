#!/usr/bin/env bash
# Generate split-feature CUDA and CPU scoring code.
#
# Split-feature mode keeps item features resident on GPU and only
# transfers the user feature vector per scoring request.
#
# Usage:
#   ./scripts/generate_split_cuda.sh <user_feature_count> [model_path] [metadata_path]
#
# Example (13 user features = I1..I13, 26 item features = C1..C26):
#   ./scripts/generate_split_cuda.sh 13

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

USER_FEATURES="${1:?Usage: $0 <user_feature_count> [model_path] [metadata_path]}"
MODEL_PATH="${2:-${PROJECT_DIR}/models/model.ubj}"
METADATA_PATH="${3:-${PROJECT_DIR}/models/training_result.json}"
OUTPUT_DIR="${PROJECT_DIR}/generated"

mkdir -p "$OUTPUT_DIR"

echo "=== Split-Feature Code Generator ==="
echo "  Model:         $MODEL_PATH"
echo "  User features: $USER_FEATURES"
echo "  Output:        $OUTPUT_DIR"
echo

# Generate CUDA version
echo "--- Generating split CUDA kernel ---"
PYTHONPATH="${PROJECT_DIR}/python" python3 -m cuda_codegen generate \
    --model "$MODEL_PATH" \
    --metadata "$METADATA_PATH" \
    --output "$OUTPUT_DIR/scoring_split.cu" \
    --user-features "$USER_FEATURES"

echo

# Generate CPU version
echo "--- Generating split CPU version ---"
PYTHONPATH="${PROJECT_DIR}/python" python3 -m cuda_codegen generate \
    --model "$MODEL_PATH" \
    --metadata "$METADATA_PATH" \
    --output "$OUTPUT_DIR/scoring_split.c" \
    --cpu \
    --user-features "$USER_FEATURES"

echo
echo "=== Generated files ==="
ls -lh "$OUTPUT_DIR"/scoring_split.*
