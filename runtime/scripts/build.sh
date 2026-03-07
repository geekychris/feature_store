#!/usr/bin/env bash
#
# build.sh — Build the scorched scoring runtime.
#
# Usage:
#   ./scripts/build.sh              # Debug build, CPU only
#   ./scripts/build.sh --release    # Release (optimized) build
#   ./scripts/build.sh --cuda       # Build with CUDA backend
#   ./scripts/build.sh --check      # Type-check only (no codegen)
#   ./scripts/build.sh --clean      # Clean before building
#   ./scripts/build.sh --release --cuda  # Combine flags
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GENERATED_DIR="$(cd "$RUNTIME_DIR/../generated" && pwd 2>/dev/null || true)"

# Defaults
PROFILE="dev"
CARGO_FLAGS=()
FEATURES=()
CLEAN=false
CHECK_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --release)  PROFILE="release"; CARGO_FLAGS+=(--release) ;;
        --cuda)     FEATURES+=(cuda) ;;
        --clean)    CLEAN=true ;;
        --check)    CHECK_ONLY=true ;;
        --help|-h)
            head -12 "$0" | tail -10
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# ---- Pre-flight checks ----
echo "=== scorched build ==="
echo "  Profile:  $PROFILE"
echo "  Features: ${FEATURES[*]:-none}"
echo "  Dir:      $RUNTIME_DIR"

# Verify generated scoring code exists
if [[ -z "$GENERATED_DIR" ]] || [[ ! -f "$GENERATED_DIR/scoring_split_core.h" ]]; then
    echo "ERROR: generated/scoring_split_core.h not found."
    echo "  Expected at: $RUNTIME_DIR/../generated/scoring_split_core.h"
    echo "  Run the code generator first (see scripts/ in the repo root)."
    exit 1
fi
echo "  Generated: $GENERATED_DIR/scoring_split_core.h ✓"

# Verify protoc is available (needed by tonic-build)
if ! command -v protoc &>/dev/null; then
    echo "WARNING: protoc not found. tonic-build may use bundled protoc."
fi

# ---- Build ----
if [[ "$CLEAN" == true ]]; then
    echo ""
    echo "--- Cleaning ---"
    cargo clean --manifest-path "$RUNTIME_DIR/Cargo.toml"
fi

# Assemble feature flags
if [[ ${#FEATURES[@]} -gt 0 ]]; then
    CARGO_FLAGS+=(--features "$(IFS=,; echo "${FEATURES[*]}")")
fi

START=$(date +%s)

if [[ "$CHECK_ONLY" == true ]]; then
    echo ""
    echo "--- Checking (no codegen) ---"
    cargo check --manifest-path "$RUNTIME_DIR/Cargo.toml" "${CARGO_FLAGS[@]}"
else
    echo ""
    echo "--- Building ---"
    cargo build --manifest-path "$RUNTIME_DIR/Cargo.toml" "${CARGO_FLAGS[@]}"
fi

END=$(date +%s)
ELAPSED=$((END - START))

echo ""
echo "=== Done ($ELAPSED s) ==="

if [[ "$CHECK_ONLY" == false ]]; then
    BINARY="$RUNTIME_DIR/target/${PROFILE}/scorched"
    if [[ -f "$BINARY" ]]; then
        SIZE=$(du -h "$BINARY" | cut -f1)
        echo "  Binary: $BINARY ($SIZE)"
    fi
fi
