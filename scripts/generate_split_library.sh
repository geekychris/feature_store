#!/usr/bin/env bash
# Generate split-feature scoring library with separate core, drivers, and benchmarks.
#
# This produces a self-contained directory with:
#   scoring_split_core.cuh   — CUDA header-only library (include in your runtime)
#   scoring_split_core.h     — CPU header-only library
#   scoring_split_main.cu/c  — Standalone main drivers
#   scoring_split_bench.cu/c — Benchmark drivers
#   Makefile                 — Build all targets
#
# Usage:
#   ./scripts/generate_split_library.sh <user_feature_count> [model_path] [metadata_path]
#
# Example:
#   ./scripts/generate_split_library.sh 13
#   cd generated && make cpu
#   cd generated && make bench_cpu BENCH_ARGS="-n 500000 -k 100"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

USER_FEATURES="${1:?Usage: $0 <user_feature_count> [model_path] [metadata_path]}"
MODEL_PATH="${2:-${PROJECT_DIR}/models/model.ubj}"
METADATA_PATH="${3:-${PROJECT_DIR}/models/training_result.json}"
OUTPUT_DIR="${PROJECT_DIR}/generated"

echo "=== Split-Feature Library Generator ==="
echo "  Model:         $MODEL_PATH"
echo "  User features: $USER_FEATURES"
echo "  Output:        $OUTPUT_DIR"
echo

PYTHONPATH="${PROJECT_DIR}/python" python3 -m cuda_codegen generate \
    --model "$MODEL_PATH" \
    --metadata "$METADATA_PATH" \
    --output "$OUTPUT_DIR" \
    --user-features "$USER_FEATURES" \
    --library

echo
echo "=== Generated files ==="
ls -lh "$OUTPUT_DIR"/scoring_split_* "$OUTPUT_DIR"/Makefile 2>/dev/null || true
