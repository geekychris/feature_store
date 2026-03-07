#!/usr/bin/env bash
#
# test.sh — Run tests, linting, and formatting checks for scorched.
#
# Usage:
#   ./scripts/test.sh              # Run all checks (tests + clippy + fmt)
#   ./scripts/test.sh --unit       # Unit tests only
#   ./scripts/test.sh --lint       # Clippy only
#   ./scripts/test.sh --fmt        # Format check only
#   ./scripts/test.sh --filter <name>  # Run tests matching <name>
#   ./scripts/test.sh --verbose    # Show test output
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
RUN_TESTS=true
RUN_CLIPPY=true
RUN_FMT=true
TEST_FILTER=""
VERBOSE=false
FAILED=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)     RUN_CLIPPY=false; RUN_FMT=false ;;
        --lint)     RUN_TESTS=false; RUN_FMT=false ;;
        --fmt)      RUN_TESTS=false; RUN_CLIPPY=false ;;
        --filter)   shift; TEST_FILTER="$1" ;;
        --verbose)  VERBOSE=true ;;
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

echo "=== scorched test suite ==="
echo "  Dir: $RUNTIME_DIR"
echo ""

# ---- Format check ----
if [[ "$RUN_FMT" == true ]]; then
    echo "--- cargo fmt --check ---"
    if cargo fmt --manifest-path "$RUNTIME_DIR/Cargo.toml" -- --check 2>&1; then
        echo "  ✓ Formatting OK"
    else
        echo "  ✗ Formatting issues found (run: cargo fmt)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
fi

# ---- Clippy ----
if [[ "$RUN_CLIPPY" == true ]]; then
    echo "--- cargo clippy ---"
    if cargo clippy --manifest-path "$RUNTIME_DIR/Cargo.toml" -- -D warnings 2>&1; then
        echo "  ✓ Clippy clean"
    else
        echo "  ✗ Clippy found issues"
        FAILED=$((FAILED + 1))
    fi
    echo ""
fi

# ---- Tests ----
if [[ "$RUN_TESTS" == true ]]; then
    echo "--- cargo test ---"
    TEST_ARGS=(--manifest-path "$RUNTIME_DIR/Cargo.toml")
    if [[ "$VERBOSE" == true ]]; then
        TEST_ARGS+=(-- --nocapture)
    fi
    if [[ -n "$TEST_FILTER" ]]; then
        # Filter goes after -- separator
        if [[ "$VERBOSE" == true ]]; then
            # Already has --, append filter
            TEST_ARGS+=("$TEST_FILTER")
        else
            TEST_ARGS+=(-- "$TEST_FILTER")
        fi
    fi

    if cargo test "${TEST_ARGS[@]}" 2>&1; then
        echo "  ✓ All tests passed"
    else
        echo "  ✗ Some tests failed"
        FAILED=$((FAILED + 1))
    fi
    echo ""
fi

# ---- Summary ----
echo "=== Results ==="
if [[ $FAILED -eq 0 ]]; then
    echo "  ✓ All checks passed"
    exit 0
else
    echo "  ✗ $FAILED check(s) failed"
    exit 1
fi
