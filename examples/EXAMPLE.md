# Java Client Example — Merchant Fraud Risk

A complete end-to-end Java example demonstrating the ML Feature Store lifecycle: synthetic data generation, feature registration, batch + scalar materialization, online serving, offline storage, Parquet export, XGBoost training, evaluation, and live inference.

## Overview

This example generates 50,000 synthetic merchant records with 15 fraud-risk features, pushes them through every layer of the feature store, trains an XGBoost binary classifier using XGBoost4J, and runs inference by fetching features from the online store.

**Key technologies**: Java 21, XGBoost4J, gRPC, Apache Parquet (via parquet-avro), java.net.http.HttpClient

## Prerequisites

- **Feature Store server running** (REST on port 8085, gRPC on 9090)
  ```bash
  # From the project root
  docker compose up -d          # PostgreSQL 16 (5433) + Redis 7 (6379)
  mvn spring-boot:run           # Starts server
  ```
- **Java 21+** and **Maven 3.9+**
- ~500 MB free memory (50K rows × 15 features + XGBoost training)

## Build

```bash
cd examples
mvn compile
```

This compiles the example and generates gRPC/protobuf stubs from `../src/main/proto/feature_store.proto`.

## Run

```bash
cd examples
mvn exec:java -Dexec.mainClass="com.platform.featurestore.examples.MerchantFraudExample"
```

Or, after building the shaded JAR:

```bash
mvn package -DskipTests
java -jar target/feature-store-examples-1.0-SNAPSHOT.jar
```

## What Each Step Does

### Step 1: Generate Dataset

Generates 50,000 synthetic merchants using `MerchantFraudDataGenerator`. Each merchant has 15 features:

| Feature | Distribution | Description |
|---------|-------------|-------------|
| `gmv_30d` | Lognormal(10, 1.5) | Gross merchandise volume, 30 days |
| `gmv_90d` | 2.5–3.5× gmv_30d | GMV, 90 days |
| `txn_count_30d` | Poisson(200) | Transaction count |
| `avg_txn_value` | gmv_30d / txn_count | Average transaction size |
| `active_days_30d` | Uniform(3, 30) | Active trading days |
| `chargeback_rate_90d` | Beta, boosted for new accounts | Chargeback rate |
| `refund_rate_30d` | Beta(2, 30) | Refund rate |
| `dispute_count_90d` | Poisson, boosted for new | Dispute count |
| `fraud_reports_30d` | Poisson, boosted for new | Fraud reports |
| `account_age_days` | Uniform(7, 1825) | Account age |
| `days_since_last_payout` | Uniform(0, 29) | Payout recency |
| `gmv_velocity_pct` | Normal(0.05, 0.30) | GMV growth rate |
| `txn_velocity_pct` | Normal(0.03, 0.25) | Txn count growth |
| `mcc_risk_score` | Uniform(0, 1) | Category risk |
| `country_risk_score` | Uniform(0, 1) | Country risk |

Label `is_high_risk` is 1 for the top ~8% by composite risk score.

### Step 2: Register Entity, Features, Feature View

Creates (or finds existing):
- **Entity** `merchant` with join key `merchant_id` (STRING)
- **15 Features** with types, descriptions, update frequencies, and max-age SLAs
- **Feature View** `merchant_fraud_gbdt_v1` (version 1) binding features to model `fraud_xgb_java_v1`

All via REST API (`/api/v1/entities`, `/api/v1/features`, `/api/v1/feature-views`).

### Step 3: Materialize Feature Vectors (Batch)

Writes all 50K merchants as pre-materialized feature vectors via `POST /api/v1/materialize/batch` in chunks of 500. These go into **RocksDB** (feature_vectors column family) and **Redis** (protobuf byte array cache).

This is the primary write path for bulk ingestion — each vector contains all 15 feature values in schema order.

### Step 4: Write Individual Scalar Features (gRPC)

Demonstrates per-feature updates via `PutScalarFeatures` gRPC call for 10 merchants. Unlike batch vector materialization, scalar writes store each feature independently in the `feature_scalars` column family.

Use case: when only a subset of features changes (e.g., an hourly pipeline updates `days_since_last_payout` without rewriting the full vector).

### Step 5: Fetch Online Features & Verify

Fetches vectors via `GetOnlineFeatures` gRPC call and verifies values match the generated data. Also retrieves the view schema to confirm feature names and schema hash.

Read path: Redis cache → RocksDB vectors → RocksDB scalar assembly (fallback).

### Step 6: Write to Offline Store (Iceberg)

Writes 5,000 merchants (75K feature records in attribute form) to the Iceberg offline store via `POST /api/v1/offline/write-records`. Each record is one (entity_id, feature_name, value_float, event_time) tuple.

This populates the offline store for point-in-time training data retrieval and audit trails.

### Step 7: Export to Parquet

Writes two local Parquet files using `parquet-avro`:

1. **Attribute form** (`merchant_features_attribute.parquet`) — tall/narrow format matching the Iceberg schema. 50K × 15 = 750K rows. Useful for data lake ingestion.

2. **Materialized form** (`merchant_features_materialized.parquet`) — wide format with one row per merchant and all 15 features as named columns plus `is_high_risk`. 50K rows. Ready for direct training use.

Files are written to `examples/exports/`.

### Step 8: Train XGBoost Model

Trains a binary classifier using XGBoost4J (`ml.dmlc:xgboost4j_2.12:2.1.3`):

- **Objective**: `binary:logistic`
- **Metric**: AUC-ROC
- **Rounds**: 200
- **Key params**: max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8
- **Eval**: Prints train/test AUC after each round

Model is saved to `examples/models/fraud_model.ubj` (Universal Binary JSON format).

### Step 9: Evaluate Model

Computes on the test set (20% stratified split):
- **AUC-ROC** (trapezoidal approximation)
- **Precision, Recall, F1** at threshold 0.5
- **Confusion matrix** (TP, FP, TN, FN)
- **Risk label distribution** (HIGH ≥ 0.7, MEDIUM 0.3–0.7, LOW < 0.3)

Expected AUC-ROC: ~0.95+ (synthetic data with clear signal).

### Step 10: Inference (Fetch → Predict)

The full production inference loop:
1. Select 20 test merchants by entity ID
2. Fetch their feature vectors from the online store via gRPC `GetOnlineFeatures`
3. Build a `DMatrix` from the fetched vectors
4. Run XGBoost prediction
5. Assign risk labels: HIGH (≥0.7), MEDIUM (0.3–0.7), LOW (<0.3)

Prints entity ID, risk label, probability, and select feature values.

## Expected Output

```
======================================================================
ML Feature Store — Java Client Example
Merchant Fraud Risk (50K merchants, 15 features, XGBoost)
======================================================================

Step 1: Generate Dataset
  Generated 50,000 merchants in ~120ms
  Positive rate: ~4000 / 50000 (8.0%)

Step 2: Register Entity, Features, Feature View
  Created entity: <uuid>
  Registered 15 features
  Created feature view: <uuid>

Step 3: Materialize Feature Vectors (batch)
  Materialized 50,000 / 50,000 vectors
  Done: 50,000 vectors in ~8s (~6,250 vectors/sec)

Step 4: Write Individual Scalar Features (gRPC)
  Wrote 150 scalar features for 10 merchants

Step 5: Fetch Online Features & Verify Round-trip
  Fetched 5 vectors (latency: ~200μs)
  Verification for m_000000: ✓ MATCH

Step 6: Write to Offline Store (Iceberg attribute form)
  Wrote 75,000 feature records (5000 merchants × 15 features)

Step 7: Export to Parquet (attribute + materialized forms)
  Attribute form:      examples/exports/merchant_features_attribute.parquet (~15 MB)
  Materialized form:   examples/exports/merchant_features_materialized.parquet (~3 MB)

Step 8: Train XGBoost Model (XGBoost4J)
  Trained 200 rounds in ~3s
  Model saved: examples/models/fraud_model.ubj

Step 9: Evaluate Model
  AUC-ROC:     0.96xx
  Precision:   0.8xxx
  Recall:      0.7xxx
  F1:          0.8xxx

Step 10: Inference (fetch from store → predict)
  Fetched 20 vectors in ~5ms
  Scored 20 entities in ~1ms
  Results:
    m_012345: risk=LOW    (prob=0.0312) gmv_30d=45230 chargeback=0.0023
    m_034567: risk=HIGH   (prob=0.8921) gmv_30d=1230  chargeback=0.0890
    ...
```

## Project Files

```
examples/
├── pom.xml                         # Standalone Maven project
├── EXAMPLE.md                      # This file
├── exports/                        # Generated Parquet files (gitignored)
├── models/                         # Trained XGBoost model (gitignored)
└── src/main/java/.../examples/
    ├── MerchantFraudExample.java   # Main entry point (10-step lifecycle)
    ├── FeatureStoreClient.java     # REST + gRPC client library
    ├── MerchantFraudDataGenerator.java  # 50K synthetic dataset
    └── ParquetExportHelper.java    # Parquet export (attribute + materialized)
```

## API Coverage

This example exercises the following feature store APIs:

**REST**:
- `POST /api/v1/entities` — create entity
- `GET /api/v1/entities/by-name/{name}` — find entity
- `POST /api/v1/features` — create feature
- `GET /api/v1/features?entityId=...` — list features
- `POST /api/v1/feature-views` — create feature view
- `POST /api/v1/materialize/batch` — batch vector write
- `POST /api/v1/offline/write-records` — Iceberg offline write

**gRPC** (`FeatureStoreService`):
- `PutScalarFeatures` — individual feature writes
- `GetOnlineFeatures` — batch vector retrieval
- `GetViewSchema` — schema metadata

## Customization

Edit constants in `MerchantFraudExample.java`:

```java
private static final int N_MERCHANTS = 50_000;  // Dataset size
private static final long SEED = 42;             // RNG seed
private static final String REST_URL = "http://localhost:8085";
private static final String GRPC_TARGET = "localhost:9090";
```

XGBoost hyperparameters are in `stepTrainXGBoost()` — adjust `max_depth`, `learning_rate`, `nRounds`, etc.
