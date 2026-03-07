# scorched — GBDT Scoring Runtime

A gRPC scoring runtime that wraps auto-generated C/CUDA GBDT ranking kernels, with multi-dataset support, Prometheus metrics, and OpenTelemetry tracing.

## Prerequisites

- **Rust** 1.75+ (with `cargo`)
- **protoc** (Protocol Buffers compiler) — tonic-build may bundle this, but having it installed avoids warnings
- **Generated scoring code** at `../generated/scoring_split_core.h` (produced by the code generator in `scripts/`)
- **CUDA toolkit** (optional, only if building with `--cuda`)

## Building

```bash
# Debug build (CPU only)
./scripts/build.sh

# Release build
./scripts/build.sh --release

# With CUDA support
./scripts/build.sh --release --cuda

# Type-check only (fast, no binary output)
./scripts/build.sh --check

# Clean before building
./scripts/build.sh --clean --release
```

The binary is produced at `target/{debug,release}/scorched`.

### Build flags

| Flag        | Description                                |
|-------------|--------------------------------------------|
| `--release` | Optimized release build                    |
| `--cuda`    | Enable CUDA backend (`--features cuda`)    |
| `--check`   | Type-check only, no binary produced        |
| `--clean`   | Run `cargo clean` before building          |

## Testing

```bash
# Run all checks: unit tests, clippy, format check
./scripts/test.sh

# Unit tests only
./scripts/test.sh --unit

# Linting only
./scripts/test.sh --lint

# Format check only
./scripts/test.sh --fmt

# Filter tests by name
./scripts/test.sh --filter dataset

# Verbose output (shows println! in tests)
./scripts/test.sh --verbose
```

### Test categories

- **`dataset`** — DatasetManager: load/unload/replace, dimension validation, capacity limits, feature ranges, binary file loading
- **`config`** — CLI config defaults, worker thread resolution, backend name
- **`metrics`** — Prometheus metric creation, encoding, scoring timer
- **`ffi`** — ModelInfo constants, raw FFI scoring (calls real C scoring code), top-K sort verification
- **`engine`** — CpuScorer: top-K, score-all, user vector resolution (empty/wrong length)
- **`grpc_service`** — Full gRPC handler tests: LoadDataset, ListDatasets, UnloadDataset, GetDatasetInfo, ScoreTopK, ScoreBatch, Healthz, GetStats — exercises the entire pipeline without a TCP listener

## Running

```bash
# Start with defaults (gRPC on :50051, metrics on :9091)
./target/release/scorched

# Custom ports
./target/release/scorched --port 9090 --metrics-port 9191

# With debug logging
./target/release/scorched --log-level debug

# JSON structured logging
./target/release/scorched --json-log

# Limit max datasets
./target/release/scorched --max-datasets 8
```

### Environment variables

All CLI args can be set via environment variables with a `SCORCHED_` prefix:

```bash
SCORCHED_PORT=9090 SCORCHED_LOG_LEVEL=debug ./target/release/scorched
```

### Endpoints

- **gRPC** — `localhost:50051` (default) — ScoreService, DatasetService, StatsService
- **Prometheus** — `http://localhost:9091/metrics` (default)

## Client Libraries

| Language | Location          | Build                              |
|----------|-------------------|------------------------------------|
| Rust     | `clients/rust/`   | `cargo build`                      |
| Go       | `clients/go/`     | `make proto && go build ./...`     |
| Java     | `clients/java/`   | `mvn compile`                      |

See each client directory for examples and benchmarks.

## Project Structure

```
runtime/
├── scripts/
│   ├── build.sh          # Build script
│   └── test.sh           # Test/lint/fmt script
├── proto/
│   └── scoring.proto     # gRPC service definitions
├── ffi/
│   └── scoring_shim.c    # C shim wrapping generated scoring header
├── src/
│   ├── main.rs           # Server bootstrap, signal handling
│   ├── config.rs         # CLI argument parsing
│   ├── ffi.rs            # Rust FFI bindings to C scoring library
│   ├── engine.rs         # Scorer trait + CpuScorer implementation
│   ├── dataset.rs        # DatasetManager (named, concurrent)
│   ├── grpc_service.rs   # gRPC handler implementations
│   └── metrics.rs        # Prometheus metrics + HTTP server
├── clients/
│   ├── rust/             # Rust client + examples
│   ├── go/               # Go client + example
│   └── java/             # Java client + example
├── Cargo.toml
├── build.rs              # Proto + C/CUDA compilation
└── Dockerfile            # Multi-stage (CPU + CUDA)
```
