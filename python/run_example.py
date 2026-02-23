#!/usr/bin/env python3
"""
End-to-end demo:
  1. Generate merchant fraud risk data
  2. Materialize feature vectors via gRPC
  3. Retrieve features via gRPC
  4. Train models (XGBoost, SVM, Neural Net)
  5. Validate model performance

Prerequisites:
  - Feature Store Java server running (REST on :8080, gRPC on :9090)
  - pip install -r requirements.txt
  - Generate protobuf stubs:
    cd python && python -m grpc_tools.protoc -I../src/main/proto \
        --python_out=. --grpc_python_out=. feature_store.proto
"""

import sys
import os
import time
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

# Add python dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# -------------------------------------------------------
# 1. Generate merchant fraud risk dataset
# -------------------------------------------------------

FEATURE_NAMES = [
    "gmv_30d", "gmv_90d", "transaction_count_30d", "avg_transaction_value",
    "active_days_30d", "chargeback_rate_90d", "refund_rate_30d",
    "dispute_count_90d", "fraud_rate_30d", "account_age_days",
    "days_since_last_payout", "gmv_trend_pct", "txn_count_trend_pct",
    "mcc_risk_score", "country_risk_score"
]
LABEL_COL = "is_high_risk"


def generate_merchant_features(n: int = 5000) -> pd.DataFrame:
    np.random.seed(42)
    merchant_ids = [f"m_{i:06d}" for i in range(n)]

    df = pd.DataFrame({
        "entity_id": merchant_ids,
        "entity_type": "merchant",
        "gmv_30d": np.random.lognormal(10, 1.5, n),
        "gmv_90d": np.random.lognormal(11, 1.5, n),
        "transaction_count_30d": np.random.poisson(200, n).astype(float),
        "avg_transaction_value": np.random.lognormal(4, 1, n),
        "active_days_30d": np.random.randint(5, 30, n).astype(float),
        "chargeback_rate_90d": np.clip(np.random.beta(1, 50, n), 0, 0.2),
        "refund_rate_30d": np.clip(np.random.beta(2, 30, n), 0, 0.3),
        "dispute_count_90d": np.random.poisson(1.5, n).astype(float),
        "fraud_rate_30d": np.clip(np.random.beta(1, 100, n), 0, 0.1),
        "account_age_days": np.random.randint(30, 1825, n).astype(float),
        "days_since_last_payout": np.random.randint(0, 30, n).astype(float),
        "gmv_trend_pct": np.random.normal(0.05, 0.3, n),
        "txn_count_trend_pct": np.random.normal(0.03, 0.25, n),
        "mcc_risk_score": np.random.uniform(0, 1, n),
        "country_risk_score": np.random.uniform(0, 1, n),
    })

    risk_score = (
        df["chargeback_rate_90d"] * 5
        + df["fraud_rate_30d"] * 8
        + (1 / (df["account_age_days"] + 1)) * 100
        - np.log1p(df["gmv_90d"]) * 0.05
        + np.random.normal(0, 0.1, n)
    )
    df["is_high_risk"] = (risk_score > risk_score.quantile(0.92)).astype(int)
    return df


def compute_schema_hash(feature_names):
    key = ",".join(feature_names)
    digest = hashlib.md5(key.encode()).hexdigest()[:8]
    return int(digest, 16) % (2**31)


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    VIEW_NAME = "merchant_fraud_risk_v3"
    VIEW_VERSION = 3

    print("=" * 60)
    print("Feature Store — End-to-End Demo")
    print("=" * 60)

    # 1. Generate data
    print("\n[1] Generating dataset...")
    df = generate_merchant_features(5000)
    print(f"    {len(df)} merchants, {df['is_high_risk'].mean()*100:.1f}% high-risk")
    schema_hash = compute_schema_hash(FEATURE_NAMES)
    print(f"    Schema hash: {schema_hash}")

    # 2. Materialize via gRPC
    print("\n[2] Materializing features via gRPC...")
    try:
        from grpc_client import FeatureStoreClient
        client = FeatureStoreClient("localhost:9090")

        batch_size = 500
        total_written = 0
        t0 = time.time()

        for start in range(0, len(df), batch_size):
            chunk = df.iloc[start:start + batch_size]
            vectors = []
            for _, row in chunk.iterrows():
                values = [float(row[f]) if not np.isnan(row[f]) else 0.0
                          for f in FEATURE_NAMES]
                vectors.append({
                    "view_name": VIEW_NAME,
                    "view_version": VIEW_VERSION,
                    "entity_type": "merchant",
                    "entity_id": row["entity_id"],
                    "values": values,
                    "schema_hash": schema_hash,
                })
            written = client.put_feature_vector_batch(vectors)
            total_written += written

        elapsed = time.time() - t0
        print(f"    Materialized {total_written} vectors in {elapsed:.1f}s "
              f"({total_written/elapsed:.0f} vectors/sec)")

        # 3. Retrieve features
        print("\n[3] Retrieving features via gRPC...")
        sample_ids = df["entity_id"].sample(10).tolist()
        t0 = time.time()
        result = client.get_online_features(VIEW_NAME, VIEW_VERSION, sample_ids)
        elapsed_ms = (time.time() - t0) * 1000

        print(f"    Retrieved {len(result['vectors'])} vectors in {elapsed_ms:.1f}ms")
        if result["vectors"]:
            v = result["vectors"][0]
            print(f"    Example: entity={v['entity_id']}, "
                  f"features={len(v['values'])}, hash={v['schema_hash']}")

        client.close()

    except ImportError:
        print("    [SKIP] gRPC stubs not generated. Run:")
        print("    cd python && python -m grpc_tools.protoc "
              "-I../src/main/proto --python_out=. --grpc_python_out=. feature_store.proto")
    except Exception as e:
        print(f"    [SKIP] gRPC connection failed: {e}")
        print("    Make sure the Feature Store server is running.")

    # 4. Train models
    print("\n[4] Training models...")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import roc_auc_score, classification_report

    X = df[FEATURE_NAMES].fillna(0).values
    y = df[LABEL_COL].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # XGBoost
    try:
        import xgboost as xgb
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            random_state=42, tree_method="hist", device="cpu",
            verbosity=0)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"    [XGBoost] AUC-ROC: {auc:.4f}")
    except ImportError:
        print("    [SKIP] xgboost not installed")

    # SVM
    from sklearn.svm import SVC
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    svm = SVC(probability=True, class_weight="balanced", random_state=42)
    svm.fit(X_train_s, y_train)
    y_prob_svm = svm.predict_proba(X_test_s)[:, 1]
    auc_svm = roc_auc_score(y_test, y_prob_svm)
    print(f"    [SVM]     AUC-ROC: {auc_svm:.4f}")

    # 5. Validate
    print("\n[5] Validation gates...")
    score_std = float(np.std(y_prob)) if 'y_prob' in dir() else float(np.std(y_prob_svm))
    best_auc = auc if 'auc' in dir() else auc_svm
    print(f"    AUC-ROC:  {best_auc:.4f} (threshold: >= 0.75) {'PASS' if best_auc >= 0.75 else 'FAIL'}")
    print(f"    Score std: {score_std:.4f} (threshold: > 0.05) {'PASS' if score_std > 0.05 else 'FAIL'}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
