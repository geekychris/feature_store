#!/usr/bin/env bash
# Compile generated CUDA/C scoring code.
#
# Usage:
#   ./scripts/build_cuda.sh [cuda|cpu|both]
#
# For CUDA builds, requires nvcc (CUDA Toolkit).
# For CPU builds, requires gcc or clang.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

GENERATED_DIR="${PROJECT_DIR}/generated"
BUILD_DIR="${PROJECT_DIR}/build"

mkdir -p "$BUILD_DIR"

MODE="${1:-both}"

build_cpu() {
    echo "=== Building CPU version ==="
    local src="$GENERATED_DIR/scoring_kernel.c"
    local out="$BUILD_DIR/scoring_cpu"

    if [ ! -f "$src" ]; then
        echo "ERROR: $src not found. Run scripts/generate_cuda.sh first."
        exit 1
    fi

    CC="${CC:-cc}"
    echo "  Compiler: $CC"
    echo "  Source:   $src"
    $CC -O2 -Wall -o "$out" "$src" -lm
    echo "  Built:    $out ($(ls -lh "$out" | awk '{print $5}'))"
}

build_cuda() {
    echo "=== Building CUDA version ==="
    local src="$GENERATED_DIR/scoring_kernel.cu"
    local out="$BUILD_DIR/scoring_cuda"

    if [ ! -f "$src" ]; then
        echo "ERROR: $src not found. Run scripts/generate_cuda.sh first."
        exit 1
    fi

    if ! command -v nvcc &> /dev/null; then
        echo "ERROR: nvcc not found. Install CUDA Toolkit to build CUDA version."
        echo "  CPU version can still be built with: $0 cpu"
        exit 1
    fi

    echo "  Compiler: nvcc $(nvcc --version | tail -1)"
    echo "  Source:   $src"
    nvcc -O2 -o "$out" "$src"
    echo "  Built:    $out ($(ls -lh "$out" | awk '{print $5}'))"
}

case "$MODE" in
    cpu)   build_cpu ;;
    cuda)  build_cuda ;;
    both)
        build_cpu
        echo
        build_cuda
        ;;
    *)
        echo "Usage: $0 [cuda|cpu|both]"
        exit 1
        ;;
esac

echo
echo "Done."
