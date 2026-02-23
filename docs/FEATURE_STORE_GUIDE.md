# Feature Store — Complete Guide

A production-grade ML Feature Store built with Java Spring Boot, providing
low-latency online serving via gRPC, offline storage via Apache Iceberg,
and a Vaadin admin UI.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Feature Store                          │
│                                                             │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐  │
│  │ REST API │  │ gRPC API  │  │ Vaadin UI│  │ Flyway   │  │
│  │ :8085    │  │ :9090     │  │ :8086    │  │ Migrate  │  │
│  └────┬─────┘  └────┬──────┘  └────┬─────┘  └────┬─────┘  │
│       │              │              │              │        │
│  ┌────┴──────────────┴──────────────┴──────────────┴────┐  │
│  │              Service Layer                            │  │
│  │  FeatureRegistryService  │  OnlineServingService      │  │
│  │  OfflineStoreService     │  ValidationService         │  │
│  │  TrainingExecutionService                             │  │
│  └──────┬──────────┬────────────┬──────────┬────────────┘  │
│         │          │            │          │                │
│  ┌──────┴───┐ ┌────┴────┐ ┌────┴───┐ ┌───┴──────────┐    │
│  │PostgreSQL│ │ RocksDB │ │ Redis  │ │Apache Iceberg│    │
│  │ Metadata │ │ Online  │ │ Cache  │ │  Offline     │    │
│  │ :5433    │ │ Store   │ │ :6379  │ │  Store       │    │
│  └──────────┘ └─────────┘ └────────┘ └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**Core components:**

- **PostgreSQL** — Metadata store for entities, features, feature views, events, and statistics
- **RocksDB** — Embedded key-value store for low-latency online feature serving (<1ms)
- **Redis** — Feature cache layer for frequently accessed vectors
- **Apache Iceberg** — Columnar offline store for historical feature data and batch training
- **gRPC** — High-throughput protocol for feature read/write (Protobuf serialization)
- **REST API** — Management API for entity/feature/view CRUD operations
- **Vaadin UI** — Admin dashboard for monitoring and management

## Quick Start

### Prerequisites

- Java 21 (OpenJDK or Corretto)
- Maven 3.9+
- Docker & Docker Compose
- Python 3.11+ (for examples)

### 1. Start Infrastructure

```bash
cd feature_store
docker compose up -d   # PostgreSQL on 5433, Redis on 6379
```

### 2. Build and Run

```bash
mvn clean compile
mvn spring-boot:run
```

The server starts on three ports:
- **REST API**: http://localhost:8085
- **Vaadin UI**: http://localhost:8086/ui/
- **gRPC**: localhost:9090

### 3. Install Python Dependencies

```bash
cd python
pip install -r requirements.txt

# Generate gRPC stubs
python -m grpc_tools.protoc \
  -I../src/main/proto \
  --python_out=. \
  --grpc_python_out=. \
  feature_store.proto
```

### 4. Run an Example

```bash
# Merchant fraud risk (simplest — runs in ~30s)
cd python/gbdt_example
python run_gbdt_example.py --standalone

# Learning to Rank (MSLR-WEB10K)
cd python/mslr_example
python run_mslr_example.py --standalone

# Click-Through Rate (Criteo)
cd python/criteo_example
python run_criteo_example.py --standalone
```

## Concepts

### Entity

An **entity** is the primary key dimension for features. Examples: `merchant`, `user`,
`query_doc_pair`, `impression`. Each entity has a join key type (STRING, INT64, etc.).

### Feature

A **feature** is a single typed value attached to an entity. Features have:
- **Name** — unique within an entity (e.g., `chargeback_rate_90d`)
- **Data type** — FLOAT64, INT64, STRING, etc.
- **Update frequency** — REALTIME, HOURLY, DAILY, WEEKLY
- **Max age** — staleness threshold in seconds
- **Default value** — fallback when the feature is missing

### Feature View

A **feature view** is an ordered set of features that together form a fixed-length
feature vector for model training and serving. Feature views have:
- **Name + version** — e.g., `merchant_fraud_gbdt_v1`
- **Schema hash** — deterministic hash ensuring client and server agree on feature order
- **Vector length** — number of features (must match the model's expected input)

### Feature Vector

A materialized feature vector is a fixed-length `float[]` keyed by
`(view_name, view_version, entity_id)`. Written via gRPC `PutFeatureVector`,
read via `GetOnlineFeatures`.

## REST API Reference

Base URL: `http://localhost:8085`

### Entities

| Method | Path | Description |
|--------|------|-------------|
| POST   | `/api/v1/entities` | Create entity |
| GET    | `/api/v1/entities` | List all entities |
| GET    | `/api/v1/entities/{id}` | Get entity by ID |
| GET    | `/api/v1/entities/by-name/{name}` | Get entity by name |
| PUT    | `/api/v1/entities/{id}` | Update entity |
| DELETE | `/api/v1/entities/{id}` | Delete entity |

**Create entity:**
```json
POST /api/v1/entities
{
  "name": "merchant",
  "description": "Merchant entity for fraud risk scoring",
  "joinKey": "merchant_id",
  "joinKeyType": "STRING"
}
```

### Features

| Method | Path | Description |
|--------|------|-------------|
| POST   | `/api/v1/features` | Create feature |
| GET    | `/api/v1/features` | List features (optional `?entityId=`) |
| GET    | `/api/v1/features/{id}` | Get feature by ID |
| PUT    | `/api/v1/features/{id}` | Update feature |
| DELETE | `/api/v1/features/{id}` | Delete feature |

**Create feature:**
```json
POST /api/v1/features
{
  "name": "chargeback_rate_90d",
  "entityId": 1,
  "dtype": "FLOAT64",
  "description": "Chargeback rate over 90 days",
  "owner": "risk-team",
  "sourcePipeline": "merchant_risk_daily",
  "updateFrequency": "DAILY",
  "maxAgeSeconds": 86400,
  "defaultValue": "0.0"
}
```

### Feature Views

| Method | Path | Description |
|--------|------|-------------|
| POST   | `/api/v1/feature-views` | Create feature view |
| GET    | `/api/v1/feature-views` | List all feature views |
| GET    | `/api/v1/feature-views/{id}` | Get feature view by ID |
| DELETE | `/api/v1/feature-views/{id}` | Delete feature view |

**Create feature view:**
```json
POST /api/v1/feature-views
{
  "name": "merchant_fraud_gbdt_v1",
  "version": 1,
  "entityId": 1,
  "description": "Merchant fraud risk GBDT features",
  "modelName": "merchant_fraud_xgb",
  "mlFramework": "XGBOOST",
  "featureIds": [1, 2, 3, 4, 5]
}
```

### Validation

```
POST /api/v1/validate/model
{
  "aucRoc": 0.95,
  "aucPr": 0.80,
  "scoreStd": 0.15,
  "baselineAucRoc": null
}
```

Returns validation gate results (pass/fail for each metric threshold).

## gRPC API Reference

Service: `FeatureStoreService` on port 9090

### GetOnlineFeatures

Fetch feature vectors for a batch of entity IDs.

```protobuf
rpc GetOnlineFeatures(GetFeaturesRequest) returns (GetFeaturesResponse);

message GetFeaturesRequest {
  string view_name = 1;
  int32 view_version = 2;
  repeated string entity_ids = 3;
  bool include_metadata = 4;
  string request_id = 5;
}
```

**Python usage:**
```python
from grpc_client import FeatureStoreClient

client = FeatureStoreClient("localhost:9090")
result = client.get_online_features("merchant_fraud_gbdt_v1", 1, ["m_001", "m_002"])
for v in result["vectors"]:
    print(f"{v['entity_id']}: {v['values'][:5]}...")
client.close()
```

### PutFeatureVector / PutFeatureVectorBatch

Write pre-materialized feature vectors.

```python
# Single vector
client.put_feature_vector(
    view_name="merchant_fraud_gbdt_v1",
    view_version=1,
    entity_type="merchant",
    entity_id="m_001",
    values=[0.008, 45230.0, 423.0, 106.7, 28.0, ...],
    schema_hash=356846339,
)

# Batch (more efficient)
vectors = [
    {"view_name": "...", "view_version": 1, "entity_type": "merchant",
     "entity_id": f"m_{i:03d}", "values": [...], "schema_hash": 356846339}
    for i in range(100)
]
written = client.put_feature_vector_batch(vectors)
```

### PutScalarFeatures

Write individual scalar features (not part of a pre-materialized vector).

```python
client.put_scalar_features("merchant", "m_001", {
    "chargeback_rate_90d": 0.008,
    "gmv_30d": 45230.0,
})
```

### GetViewSchema

Retrieve the schema for a feature view.

```python
schema = client.get_view_schema("merchant_fraud_gbdt_v1", version=1)
print(schema["feature_names"])  # ["gmv_30d", "gmv_90d", ...]
print(schema["schema_hash"])    # 356846339
```

## Integration Guide

### Typical ML Pipeline

```
1. Define entity + features → REST API (one-time setup)
2. Create feature view → REST API (one-time, or on schema change)
3. Materialize feature vectors → gRPC PutFeatureVectorBatch (ETL pipeline)
4. Train model → Pull features from DataFrames or feature store
5. Validate model → REST /api/v1/validate/model
6. Serve predictions → gRPC GetOnlineFeatures → model.predict()
```

### Python Integration Pattern

```python
import numpy as np
import xgboost as xgb
from grpc_client import FeatureStoreClient

# 1. Load trained model
model = xgb.XGBClassifier()
model.load_model("models/model.ubj")

# 2. Connect to feature store
client = FeatureStoreClient("localhost:9090")

# 3. Fetch features for a batch of entities
result = client.get_online_features(
    view_name="merchant_fraud_gbdt_v1",
    view_version=1,
    entity_ids=["m_001", "m_002", "m_003"],
)

# 4. Build feature matrix and predict
X = np.array([v["values"] for v in result["vectors"]])
predictions = model.predict_proba(X)[:, 1]

# 5. Use predictions in your application
for entity_id, prob in zip(entity_ids, predictions):
    print(f"{entity_id}: risk={prob:.4f}")

client.close()
```

### Java/Spring Integration

The feature store is a Spring Boot application. To embed it or call it from
another Java service:

```java
// gRPC client (from any JVM service)
ManagedChannel channel = ManagedChannelBuilder
    .forAddress("localhost", 9090)
    .usePlaintext()
    .build();
FeatureStoreServiceGrpc.FeatureStoreServiceBlockingStub stub =
    FeatureStoreServiceGrpc.newBlockingStub(channel);

GetFeaturesRequest request = GetFeaturesRequest.newBuilder()
    .setViewName("merchant_fraud_gbdt_v1")
    .setViewVersion(1)
    .addEntityIds("m_001")
    .build();
GetFeaturesResponse response = stub.getOnlineFeatures(request);
```

### Schema Hash Verification

The schema hash ensures feature vector compatibility. Both the client (Python)
and server (Java) compute the same hash:

```python
import hashlib

def compute_schema_hash(feature_names: list[str]) -> int:
    key = ",".join(feature_names)
    digest = hashlib.md5(key.encode()).hexdigest()[:8]
    return int(digest, 16) % (2**31)
```

If you get a schema hash mismatch error, it means the feature order in your
client doesn't match the feature view definition on the server.

## Dataset Examples

### Example 1: Merchant Fraud Risk (GBDT)

**Location:** `python/gbdt_example/`

Binary classification predicting merchant fraud risk using 15 hand-crafted
features (GMV, chargeback rates, account age, etc.).

```bash
# Standalone (no server needed)
cd python/gbdt_example
python run_gbdt_example.py --standalone

# Full pipeline (requires running server)
python run_gbdt_example.py
```

**Features:** 15 (gmv_30d, gmv_90d, txn_count_30d, avg_txn_value, ...)
**Label:** is_high_risk (binary, ~8% positive rate)
**Model:** XGBoost GBDT classifier
**Expected AUC-ROC:** ~0.997

### Example 2: MSLR-WEB10K Learning to Rank

**Location:** `python/mslr_example/`

Learning to Rank on Microsoft's MSLR-WEB10K dataset. 136-dimensional feature
vectors with relevance labels 0-4. Trains a LambdaMART ranker.

```bash
# Standalone with synthetic data
cd python/mslr_example
python run_mslr_example.py --standalone

# With real data (downloads ~1.2 GB from Microsoft Research)
python run_mslr_example.py --standalone --no-synthetic

# Full pipeline with feature store
python run_mslr_example.py

# Customize synthetic data size
python run_mslr_example.py --standalone --n-queries 500 --docs-per-query 100
```

**Dataset:** Microsoft Learning to Rank (MSLR-WEB10K)
- 136 features per query-document pair (TF, IDF, BM25, PageRank, link stats)
- Relevance labels: 0 (irrelevant) to 4 (perfectly relevant)
- 5 pre-defined train/validation/test folds
- ~10K queries per fold

**Features:** 136 (feature_1 through feature_136)
**Label:** relevance (0-4)
**Model:** XGBoost LambdaMART ranker (rank:ndcg objective)
**Metrics:** NDCG@1, NDCG@5, NDCG@10, MAP
**Entity:** query_doc_pair

**Data formats:**
- Real data: SVM-light format (`relevance qid:N 1:val 2:val ... 136:val`)
- Synthetic: Generated with correlated features and realistic label distribution

### Example 3: Criteo CTR Prediction

**Location:** `python/criteo_example/`

Click-Through Rate prediction on display advertising data. 13 numeric features
and 26 categorical features (hashed to float buckets). Binary click/no-click
classification.

```bash
# Standalone with synthetic data
cd python/criteo_example
python run_criteo_example.py --standalone

# With real data (requires Kaggle account)
python run_criteo_example.py --standalone --no-synthetic --max-rows 1000000

# Full pipeline with feature store
python run_criteo_example.py

# Smaller synthetic dataset for testing
python run_criteo_example.py --standalone --n-samples 50000
```

**Dataset:** Criteo Display Advertising Challenge
- 13 integer/count features (I1-I13)
- 26 categorical features (C1-C26), hashed to float buckets
- Binary label: click (1) or no-click (0)
- ~45M rows in full dataset (~4.3 GB)

**Features:** 39 (I1-I13 numeric + C1-C26 hashed categorical)
**Label:** click (binary, ~3.4% positive rate)
**Model:** XGBoost binary classifier
**Metrics:** AUC-ROC, LogLoss, AUC-PR
**Entity:** impression

**Real data download:** Requires Kaggle CLI and accepted competition rules:
```bash
pip install kaggle
# Accept rules at: https://www.kaggle.com/c/criteo-display-ad-challenge/rules
kaggle competitions download -c criteo-display-ad-challenge
```

## Vaadin Admin UI

Access the admin dashboard at http://localhost:8086/ui/

**Views available:**
- **Dashboard** — Overview of entities, features, views, and recent events
- **Entities** — CRUD management for entity types
- **Features** — Browse and manage feature definitions
- **Feature Views** — View and manage feature view schemas
- **Feature Lookup** — Look up specific feature vectors by entity ID
- **Statistics** — Feature distribution statistics and monitoring
- **Training** — Execute and monitor Python training scripts

## Configuration

Key settings in `src/main/resources/application.yml`:

```yaml
server:
  port: 8085               # REST API port

grpc:
  server:
    port: 9090             # gRPC port

spring:
  datasource:
    url: jdbc:postgresql://localhost:5433/featurestore
    username: featurestore
    password: featurestore

feature-store:
  rocksdb:
    path: ./data/rocksdb   # RocksDB data directory
  redis:
    host: localhost
    port: 6379
  iceberg:
    warehouse: ./data/iceberg  # Iceberg warehouse path
```

## Project Structure

```
feature_store/
├── src/main/java/com/platform/featurestore/
│   ├── FeatureStoreApplication.java
│   ├── config/                # Spring/RocksDB/Redis config
│   ├── controller/            # REST controllers
│   ├── dto/                   # Request/response DTOs
│   ├── entity/                # JPA entities
│   ├── grpc/                  # gRPC service implementation
│   ├── repository/            # Spring Data JPA repositories
│   ├── service/               # Business logic services
│   └── store/                 # RocksDB, Redis, Iceberg stores
├── src/main/proto/
│   └── feature_store.proto    # gRPC/Protobuf definitions
├── src/main/resources/
│   ├── application.yml
│   └── db/migration/          # Flyway SQL migrations
├── src/test/                  # Unit tests (43 tests)
├── python/
│   ├── requirements.txt
│   ├── grpc_client.py         # Python gRPC client
│   ├── gbdt_example/          # Merchant fraud risk example
│   ├── mslr_example/          # Learning to Rank example
│   └── criteo_example/        # CTR prediction example
├── docs/
│   └── FEATURE_STORE_GUIDE.md # This file
├── docker-compose.yml
└── pom.xml
```

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Check what's using the port
lsof -i :8085
lsof -i :9090

# Kill the process
kill -9 <PID>
```

**PostgreSQL connection refused:**
```bash
docker compose up -d
docker compose ps  # Verify containers are running
```

**gRPC stubs not found:**
```bash
cd python
python -m grpc_tools.protoc \
  -I../src/main/proto \
  --python_out=. \
  --grpc_python_out=. \
  feature_store.proto
```

**Schema hash mismatch:**
This means your Python feature order doesn't match the server's feature view.
Re-register the features or verify the feature list order matches.

**LazyInitializationException in Vaadin UI:**
The entity graphs are configured in the JPA repositories. If you add new
relationships, add `@EntityGraph` annotations to the repository methods.

### Performance Tuning

- **gRPC batch size:** Use `PutFeatureVectorBatch` with 500-1000 vectors per call
- **RocksDB:** Tune block cache size via `feature-store.rocksdb.block-cache-size`
- **Redis:** Enable Redis cache for hot entities via `feature-store.redis.enabled=true`
- **Training:** Use `--no-search` to skip hyperparameter search for faster iterations
