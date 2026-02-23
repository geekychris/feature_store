"""
Merchant fraud risk dataset for GBDT training.

Generates a realistic synthetic dataset with 15 features designed to predict
merchant fraud risk. Features span GMV, transaction patterns, chargeback/refund
rates, account tenure, and risk scores.

Can be used standalone (returns DataFrames) or integrated with the feature store
(registers entities/features/views via REST, materializes via gRPC).
"""

import hashlib
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Feature schema — single source of truth
# ---------------------------------------------------------------------------

FEATURE_SCHEMA = [
    # (name, dtype, description, update_frequency, max_age_seconds)
    ("gmv_30d",               "FLOAT64", "Gross merchandise volume last 30 days",      "DAILY",  86400),
    ("gmv_90d",               "FLOAT64", "Gross merchandise volume last 90 days",      "DAILY",  86400),
    ("txn_count_30d",         "FLOAT64", "Transaction count last 30 days",             "DAILY",  86400),
    ("avg_txn_value",         "FLOAT64", "Average transaction value",                  "DAILY",  86400),
    ("active_days_30d",       "FLOAT64", "Days with at least one transaction (30d)",   "DAILY",  86400),
    ("chargeback_rate_90d",   "FLOAT64", "Chargeback rate over 90 days",              "DAILY",  86400),
    ("refund_rate_30d",       "FLOAT64", "Refund rate over 30 days",                  "DAILY",  86400),
    ("dispute_count_90d",     "FLOAT64", "Number of disputes in 90 days",             "DAILY",  86400),
    ("fraud_reports_30d",     "FLOAT64", "Fraud reports received in 30 days",         "DAILY",  86400),
    ("account_age_days",      "FLOAT64", "Days since merchant account creation",      "DAILY",  86400),
    ("days_since_last_payout","FLOAT64", "Days since last payout settlement",         "HOURLY", 3600),
    ("gmv_velocity_pct",      "FLOAT64", "GMV change rate (30d vs prior 30d)",        "DAILY",  86400),
    ("txn_velocity_pct",      "FLOAT64", "Transaction count change rate",             "DAILY",  86400),
    ("mcc_risk_score",        "FLOAT64", "Merchant category code risk (0-1)",         "WEEKLY", 604800),
    ("country_risk_score",    "FLOAT64", "Country-level risk index (0-1)",            "WEEKLY", 604800),
]

FEATURE_NAMES = [f[0] for f in FEATURE_SCHEMA]
LABEL_COL = "is_high_risk"
VIEW_NAME = "merchant_fraud_gbdt_v1"
VIEW_VERSION = 1
ENTITY_NAME = "merchant"


def compute_schema_hash(feature_names: list[str]) -> int:
    """Same hash algorithm as the Java side (FeatureRegistryService.computeSchemaHash)."""
    key = ",".join(feature_names)
    digest = hashlib.md5(key.encode()).hexdigest()[:8]
    return int(digest, 16) % (2**31)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(n_merchants: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic merchant fraud risk dataset.

    The label (is_high_risk) is derived from a latent risk score that combines
    chargeback rate, fraud reports, account age, and GMV velocity — producing
    roughly 8% positive rate (class imbalance typical of fraud problems).
    """
    rng = np.random.default_rng(seed)

    merchant_ids = [f"m_{i:06d}" for i in range(n_merchants)]

    # --- Base features ---
    gmv_30d = rng.lognormal(10, 1.5, n_merchants)
    gmv_90d = gmv_30d * rng.uniform(2.5, 3.5, n_merchants)  # ~3x of 30d
    txn_count_30d = rng.poisson(200, n_merchants).astype(float)
    avg_txn_value = gmv_30d / np.maximum(txn_count_30d, 1)

    # --- Activity ---
    active_days_30d = rng.integers(3, 31, n_merchants).astype(float)
    account_age_days = rng.integers(7, 1825, n_merchants).astype(float)
    days_since_last_payout = rng.integers(0, 30, n_merchants).astype(float)

    # --- Risk indicators ---
    # New/risky merchants: higher chargeback, fraud, dispute rates
    is_new = (account_age_days < 90).astype(float)
    chargeback_rate_90d = np.clip(
        rng.beta(1 + is_new * 2, 50, n_merchants), 0, 0.2
    )
    refund_rate_30d = np.clip(rng.beta(2, 30, n_merchants), 0, 0.3)
    dispute_count_90d = rng.poisson(1.5 + is_new * 3, n_merchants).astype(float)
    fraud_reports_30d = rng.poisson(0.3 + is_new * 1.5, n_merchants).astype(float)

    # --- Trends ---
    gmv_velocity_pct = rng.normal(0.05, 0.30, n_merchants)
    txn_velocity_pct = rng.normal(0.03, 0.25, n_merchants)

    # --- Categorical risk scores ---
    mcc_risk_score = rng.uniform(0, 1, n_merchants)
    country_risk_score = rng.uniform(0, 1, n_merchants)

    # --- Latent risk score → label ---
    risk_score = (
        chargeback_rate_90d * 5.0
        + fraud_reports_30d * 1.2
        + dispute_count_90d * 0.3
        + (1.0 / (account_age_days + 1)) * 200
        + np.clip(gmv_velocity_pct, 0, None) * 0.8  # rapid growth is suspicious
        + mcc_risk_score * 0.5
        + country_risk_score * 0.3
        - np.log1p(gmv_90d) * 0.03  # established volume reduces risk slightly
        + rng.normal(0, 0.15, n_merchants)  # noise
    )
    is_high_risk = (risk_score > np.quantile(risk_score, 0.92)).astype(int)

    df = pd.DataFrame({
        "entity_id": merchant_ids,
        "gmv_30d": gmv_30d,
        "gmv_90d": gmv_90d,
        "txn_count_30d": txn_count_30d,
        "avg_txn_value": avg_txn_value,
        "active_days_30d": active_days_30d,
        "chargeback_rate_90d": chargeback_rate_90d,
        "refund_rate_30d": refund_rate_30d,
        "dispute_count_90d": dispute_count_90d,
        "fraud_reports_30d": fraud_reports_30d,
        "account_age_days": account_age_days,
        "days_since_last_payout": days_since_last_payout,
        "gmv_velocity_pct": gmv_velocity_pct,
        "txn_velocity_pct": txn_velocity_pct,
        "mcc_risk_score": mcc_risk_score,
        "country_risk_score": country_risk_score,
        LABEL_COL: is_high_risk,
    })

    return df


def train_test_split_df(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split preserving class balance."""
    from sklearn.model_selection import train_test_split
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df[LABEL_COL])


# ---------------------------------------------------------------------------
# Feature store integration (REST + gRPC)
# ---------------------------------------------------------------------------

def register_with_feature_store(
    base_url: str = "http://localhost:8080",
) -> dict:
    """
    Register entity, features, and feature view via REST API.
    Returns dict with entity_id, feature_ids, and view_id.
    """
    headers = {"Content-Type": "application/json"}

    # 1. Create entity
    resp = requests.post(f"{base_url}/api/v1/entities", json={
        "name": ENTITY_NAME,
        "description": "Merchant entity for fraud risk scoring",
        "joinKey": "merchant_id",
        "joinKeyType": "STRING",
    }, headers=headers)
    if resp.status_code == 201:
        entity_id = resp.json()["id"]
        print(f"  Created entity: {entity_id}")
    elif resp.status_code == 409 or "already exists" in resp.text.lower():
        # Entity already exists — look it up
        resp2 = requests.get(f"{base_url}/api/v1/entities/by-name/{ENTITY_NAME}")
        entity_id = resp2.json()["id"]
        print(f"  Entity already exists: {entity_id}")
    else:
        resp.raise_for_status()
        entity_id = resp.json()["id"]

    # 2. Create features
    feature_ids = []
    for name, dtype, desc, freq, max_age in FEATURE_SCHEMA:
        resp = requests.post(f"{base_url}/api/v1/features", json={
            "name": name,
            "entityId": entity_id,
            "dtype": dtype,
            "description": desc,
            "owner": "fraud-risk-team",
            "sourcePipeline": "merchant_risk_daily",
            "updateFrequency": freq,
            "maxAgeSeconds": max_age,
            "defaultValue": "0.0",
        }, headers=headers)
        if resp.status_code == 201:
            feature_ids.append(resp.json()["id"])
        else:
            print(f"  Warning: feature '{name}' may already exist ({resp.status_code})")
            # Try to look it up
            features_resp = requests.get(
                f"{base_url}/api/v1/features", params={"entityId": entity_id}
            )
            if features_resp.ok:
                for f in features_resp.json():
                    if f["name"] == name:
                        feature_ids.append(f["id"])
                        break

    print(f"  Registered {len(feature_ids)} features")

    # 3. Create feature view
    resp = requests.post(f"{base_url}/api/v1/feature-views", json={
        "name": VIEW_NAME,
        "version": VIEW_VERSION,
        "entityId": entity_id,
        "description": "Merchant fraud risk GBDT features",
        "modelName": "merchant_fraud_xgb",
        "mlFramework": "XGBOOST",
        "featureIds": feature_ids,
    }, headers=headers)
    if resp.status_code == 201:
        view = resp.json()
        print(f"  Created view: {view['name']} v{view['version']} "
              f"(hash={view['schemaHash']}, len={view['vectorLength']})")
    else:
        print(f"  View may already exist ({resp.status_code})")

    return {
        "entity_id": entity_id,
        "feature_ids": feature_ids,
    }


def materialize_to_feature_store(
    df: pd.DataFrame,
    grpc_target: str = "localhost:9090",
    batch_size: int = 500,
) -> int:
    """
    Materialize feature vectors to the feature store via gRPC.
    Returns total vectors written.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from grpc_client import FeatureStoreClient

    schema_hash = compute_schema_hash(FEATURE_NAMES)
    client = FeatureStoreClient(grpc_target)
    total = 0

    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        vectors = []
        for _, row in chunk.iterrows():
            values = [float(row[f]) if pd.notna(row[f]) else 0.0 for f in FEATURE_NAMES]
            vectors.append({
                "view_name": VIEW_NAME,
                "view_version": VIEW_VERSION,
                "entity_type": ENTITY_NAME,
                "entity_id": row["entity_id"],
                "values": values,
                "schema_hash": schema_hash,
            })
        written = client.put_feature_vector_batch(vectors)
        total += written

    client.close()
    return total


if __name__ == "__main__":
    print("Generating dataset...")
    df = generate_dataset(10_000)
    print(f"  {len(df)} merchants")
    print(f"  {df[LABEL_COL].sum()} high-risk ({df[LABEL_COL].mean()*100:.1f}%)")
    print(f"  Features: {FEATURE_NAMES}")
    print(f"\nSample:\n{df[FEATURE_NAMES + [LABEL_COL]].describe()}")
