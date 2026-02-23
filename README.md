# ML Feature Store

A production-grade ML Feature Store built with Java Spring Boot, providing dual online/offline storage, gRPC + REST APIs, a Vaadin Flow admin UI, and training integration in both Python and Java.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌───────────────┐
│  REST API   │     │  gRPC API   │     │  Vaadin UI    │
│  (port 8085)│     │  (port 9090)│     │  (port 8086)  │
└──────┬──────┘     └──────┬──────┘     └──────┬────────┘
       │                   │                   │
       └─────────┬─────────┴───────────────────┘
                 │
    ┌────────────┴────────────┐
    │     Service Layer       │
    │  Registry │ Serving     │
    │  Material │ Validation  │
    │  Training │ Offline     │
    └────────────┬────────────┘
                 │
    ┌────────┬───┴───┬──────────┐
    │        │       │          │
┌───┴──┐ ┌──┴──┐ ┌──┴───┐ ┌───┴────┐
│Redis │ │Rocks│ │Iceberg│ │Postgres│
│Cache │ │DB   │ │Parquet│ │Registry│
└──────┘ └─────┘ └──────┘ └────────┘
```

### Storage Layers

| Layer | Technology | Purpose | Latency |
|-------|-----------|---------|--------|
| Hot cache | Redis 7 | Sub-millisecond lookups | <1ms |
| Online store | RocksDB (JNI) | Pre-materialized vectors + scalar fallback | ~1ms |
| Offline store | Apache Iceberg/Parquet | Training data with ASOF joins | seconds |
| Metadata registry | PostgreSQL 16 | Entity/feature/view definitions | ~5ms |

### Read Path Priority

1. **Redis** — hot cache, protobuf byte arrays with TTL
2. **RocksDB vector** — pre-materialized `FeatureVector` in `feature_vectors` column family
3. **RocksDB scalar assembly** — MultiGet individual features, assemble vector on-the-fly

## Prerequisites

- Java 21+
- Maven 3.9+
- Docker & Docker Compose (for PostgreSQL and Redis)
- Python 3.9+ (optional, for Python training examples)

## Quick Start

### 1. Start Infrastructure

```bash
docker compose up -d
```

This starts PostgreSQL 16 (port 5433) and Redis 7 (port 6379).

### 2. Build and Run

```bash
# Compile (includes protobuf code generation)
mvn compile

# Run the application
mvn spring-boot:run
```

The app starts with:
- REST API on `http://localhost:8085`
- gRPC on `localhost:9090`
- Vaadin admin UI on `http://localhost:8086/ui/`
- Flyway auto-migrates the PostgreSQL schema

### 3. Dev Profile (No Docker Required)

```bash
mvn spring-boot:run -Dspring-boot.run.profiles=dev
```

Uses H2 in-memory DB and disables Redis.

## Vaadin Admin UI

A web-based admin interface at `http://localhost:8086/ui/` providing:

- **Dashboard** — overview statistics, entity/view/feature counts
- **Entity Browser** — list and drill into entities with their features
- **Feature Views** — browse views, see schema, feature membership
- **Feature Lookup** — query the online store by entity ID, inspect vectors
- **Training** — trigger and monitor Python training jobs (GBDT, ranking, CTR)
- **Statistics** — data explorer with sample data and scoring

## REST API Reference

### Registry

```bash
# Create entity
curl -X POST http://localhost:8085/api/v1/entities \
  -H 'Content-Type: application/json' \
  -d '{"name":"merchant","description":"Merchant entity","joinKey":"merchant_id","joinKeyType":"STRING"}'

# Find entity by name
curl http://localhost:8085/api/v1/entities/by-name/merchant

# Create feature
curl -X POST http://localhost:8085/api/v1/features \
  -H 'Content-Type: application/json' \
  -d '{"name":"gmv_30d","entityId":"<ENTITY_UUID>","dtype":"FLOAT64","description":"GMV last 30d","owner":"risk-team","sourcePipeline":"daily","updateFrequency":"DAILY","maxAgeSeconds":86400,"defaultValue":"0.0"}'

# Create feature view
curl -X POST http://localhost:8085/api/v1/feature-views \
  -H 'Content-Type: application/json' \
  -d '{"name":"merchant_fraud","version":1,"entityId":"<ENTITY_UUID>","description":"Fraud features","modelName":"fraud_xgb_v1","mlFramework":"XGBOOST","featureIds":["<FEAT1_UUID>","<FEAT2_UUID>"]}'

# List entities / features / views
curl http://localhost:8085/api/v1/entities
curl http://localhost:8085/api/v1/features?entityId=<ENTITY_UUID>
curl http://localhost:8085/api/v1/feature-views
curl http://localhost:8085/api/v1/feature-views/merchant_fraud/latest
```

### Materialization

```bash
# Materialize single vector
curl -X POST http://localhost:8085/api/v1/materialize/vector \
  -H 'Content-Type: application/json' \
  -d '{"viewName":"merchant_fraud","viewVersion":1,"entityType":"merchant","entityId":"m_001","values":[45230.0,0.008,423],"schemaHash":0}'

# Batch materialize
curl -X POST http://localhost:8085/api/v1/materialize/batch \
  -H 'Content-Type: application/json' \
  -d '{"vectors":[{"viewName":"merchant_fraud","viewVersion":1,"entityType":"merchant","entityId":"m_001","values":[45230.0,0.008],"schemaHash":0}]}'
```

### Online Serving

```bash
# Get single vector
curl http://localhost:8085/api/v1/serving/online/merchant_fraud/1/m_001

# Batch get
curl -X POST http://localhost:8085/api/v1/serving/online \
  -H 'Content-Type: application/json' \
  -d '{"viewName":"merchant_fraud","viewVersion":1,"entityIds":["m_001","m_002"]}'
```

### Offline Store

```bash
# Write feature records to Iceberg (attribute form)
curl -X POST http://localhost:8085/api/v1/offline/write-records \
  -H 'Content-Type: application/json' \
  -d '{"records":[{"entityType":"merchant","entityId":"m_001","featureName":"gmv_30d","valueFloat":45230.0,"eventTime":"2025-01-01T00:00:00Z","pipelineId":"daily","viewVersion":"1"}]}'

# Generate ASOF-joined training data
curl -X POST http://localhost:8085/api/v1/offline/training-data \
  -H 'Content-Type: application/json' \
  -d '{"entityType":"merchant","viewName":"merchant_fraud","viewVersion":"1","labelEvents":[{"entityId":"m_001","label":1,"eventTime":"2025-01-01T00:00:00Z"}]}'
```

### Validation

```bash
# Validate vector against schema
curl http://localhost:8085/api/v1/validate/merchant_fraud/1/m_001

# Validate model metrics
curl -X POST http://localhost:8085/api/v1/validate/model \
  -H 'Content-Type: application/json' \
  -d '{"aucRoc":0.85,"aucPr":0.55,"scoreStd":0.10,"baselineAucRoc":0.83}'
```

## gRPC API

The gRPC service runs on port 9090 and supports:

- `GetOnlineFeatures` — batch feature vector retrieval
- `PutFeatureVector` — write a single vector
- `PutFeatureVectorBatch` — write multiple vectors
- `PutScalarFeatures` — write individual scalar features
- `GetViewSchema` — retrieve view schema

Proto definitions are in `src/main/proto/feature_store.proto`.

## Examples

### Java Client Example (examples/)

A standalone Java module demonstrating the full ML lifecycle with 50K synthetic merchants and XGBoost4J training. See [examples/EXAMPLE.md](examples/EXAMPLE.md) for detailed documentation.

```bash
cd examples
mvn compile
mvn exec:java -Dexec.mainClass="com.platform.featurestore.examples.MerchantFraudExample"
```

**Steps**: Generate 50K merchants → Register entity/features/view → Batch materialize → Scalar writes → Online fetch + verify → Offline (Iceberg) write → Parquet export (attribute + materialized) → XGBoost train → Evaluate (AUC, precision, recall) → Inference (fetch from store → predict → risk labels)

### Python: GBDT Merchant Fraud (python/gbdt_example/)

XGBoost GBDT for merchant fraud risk scoring with 10K merchants, 15 features, stratified CV, hyperparameter search.

```bash
cd python
pip install -r requirements.txt

# Standalone (no server needed)
python gbdt_example/run_gbdt_example.py --standalone

# Full pipeline (requires running server)
python gbdt_example/run_gbdt_example.py

# Inference only
python gbdt_example/run_gbdt_example.py --infer-only --standalone --entity-ids m_000001 m_000050
```

### Python: MSLR-WEB10K Ranking (python/mslr_example/)

LambdaMART ranking model on Microsoft Learning to Rank dataset. Registers query/document entities, trains pairwise ranker.

```bash
python python/mslr_example/run_mslr_example.py --standalone
```

### Python: Criteo CTR Prediction (python/criteo_example/)

Click-through rate prediction on Criteo display advertising data. Mixed dense+sparse features, log-loss optimization.

```bash
python python/criteo_example/run_criteo_example.py --standalone
```

### Dataset Discovery & Import

The `python/dataset_tool.py` utility auto-discovers training datasets and generates manifests:

```bash
python python/dataset_tool.py discover
python python/import_dataset.py --dataset gbdt_example
```

## Python Client

The gRPC client (`python/grpc_client.py`) provides a `FeatureStoreClient` class:

```bash
# Generate protobuf stubs first
cd python && python -m grpc_tools.protoc -I../src/main/proto \
    --python_out=. --grpc_python_out=. feature_store.proto
```

## Testing

```bash
# Unit tests only (no Docker required)
mvn test -Dtest="RocksDBFeatureStoreTest,ValidationServiceTest,FeatureRegistryServiceTest,OnlineServingServiceTest,SerializationTest"

# Integration tests (requires Docker)
mvn test -Dtest="FeatureStoreIntegrationTest"

# All tests
mvn test
```

### Test Coverage

- **RocksDBFeatureStoreTest** — key encoding, vector/scalar/embedding CRUD, batch ops, schema storage, scalar assembly
- **ValidationServiceTest** — schema hash validation, staleness SLAs, model metric gates
- **FeatureRegistryServiceTest** — schema hash computation (determinism, order-sensitivity)
- **OnlineServingServiceTest** — Redis→RocksDB→scalar fallback chain, batch serving
- **SerializationTest** — Protobuf round-trip for all message types
- **FeatureStoreIntegrationTest** — Full REST API through PostgreSQL with Testcontainers

## Key Design Decisions

- **Key encoding**: Vector keys use `[MD5_prefix:4B][version:2B][entity_id:var]` for data locality; scalar keys use `[type_hash:4B][entity_id:var][NUL][feature_name:var]`
- **Schema hash**: `MD5(comma-joined feature names)[:8 hex] % Integer.MAX_VALUE` — used for train/serve consistency validation
- **Column families**: 5 RocksDB CFs (`default`, `feature_vectors`, `feature_scalars`, `embeddings`, `schemas`) with tuned options per workload
- **Cache-aside**: Redis stores protobuf byte arrays with configurable TTL; falls through to RocksDB on miss
- **Validation gates**: AUC-ROC ≥ 0.75, AUC-PR ≥ 0.40, score std > 0.05, degradation < 0.02 vs baseline

## Project Structure

```
src/main/
├── proto/                          # Protobuf definitions
├── resources/
│   ├── application.yml             # Config (default + dev profiles)
│   └── db/migration/V1__init.sql   # Flyway DDL
└── java/com/platform/featurestore/
    ├── config/                     # RocksDB, Redis, Iceberg configs
    ├── controller/                 # REST controllers (Registry, Serving, Offline)
    ├── domain/                     # JPA entities
    ├── dto/                        # REST DTOs (Java records)
    ├── grpc/                       # gRPC service implementation
    ├── repository/                 # Spring Data JPA repos
    ├── service/                    # Business logic
    ├── store/
    │   ├── online/                 # RocksDB + Redis
    │   └── offline/                # Iceberg/Parquet
    └── ui/                         # Vaadin Flow admin views
examples/
├── pom.xml                         # Standalone Maven project
├── EXAMPLE.md                      # Detailed Java example docs
└── src/main/java/.../examples/
    ├── MerchantFraudExample.java   # 10-step ML lifecycle
    ├── FeatureStoreClient.java     # REST + gRPC client
    ├── MerchantFraudDataGenerator.java  # 50K synthetic merchants
    └── ParquetExportHelper.java    # Parquet export
python/
├── grpc_client.py                  # gRPC client library
├── dataset_tool.py                 # Dataset discovery & manifest
├── import_dataset.py               # Dataset import utility
├── requirements.txt
├── gbdt_example/                   # GBDT fraud risk training
├── mslr_example/                   # LambdaMART ranking
└── criteo_example/                 # CTR prediction
docker-compose.yml                  # PostgreSQL 16 (5433) + Redis 7 (6379)
```

## Ports Reference

| Service | Port | Notes |
|---------|------|-------|
| REST API | 8085 | Feature store REST endpoints |
| gRPC | 9090 | Online serving + scalar writes |
| Vaadin UI | 8086 | Admin UI at `/ui/` |
| PostgreSQL | 5433 | Metadata registry |
| Redis | 6379 | Feature vector cache |
