# Multi-stage Dockerfile for scorched runtime
# Build: docker build -t scorched:cpu .
# CUDA:  docker build -t scorched:cuda --build-arg ENABLE_CUDA=1 .

ARG ENABLE_CUDA=0

# ==== Builder stage ====
FROM rust:1.82-bookworm AS builder

RUN apt-get update && apt-get install -y protobuf-compiler libprotobuf-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy generated scoring code (consumed by FFI)
COPY generated/ /build/generated/

# Copy runtime source
COPY runtime/ /build/runtime/

WORKDIR /build/runtime

ARG ENABLE_CUDA
RUN if [ "$ENABLE_CUDA" = "1" ]; then \
        cargo build --release --features cuda; \
    else \
        cargo build --release; \
    fi

# ==== CPU runtime stage ====
FROM debian:bookworm-slim AS runtime-cpu

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/runtime/target/release/scorched /usr/local/bin/scorched

EXPOSE 50051 9091

ENTRYPOINT ["scorched"]
CMD ["--port", "50051", "--metrics-port", "9091"]

# ==== CUDA runtime stage ====
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime-cuda

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/runtime/target/release/scorched /usr/local/bin/scorched

EXPOSE 50051 9091

ENTRYPOINT ["scorched"]
CMD ["--port", "50051", "--metrics-port", "9091", "--cuda"]

# ==== Final stage selection ====
FROM runtime-cpu
