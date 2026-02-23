"""
Criteo Display Advertising Challenge dataset for the Feature Store.

Binary click prediction (CTR) dataset with 13 numeric features and
26 categorical features. One of the standard benchmarks for CTR models.

Two modes:
  - Real data:  Downloads from Criteo/Kaggle (~4.3 GB compressed)
  - Synthetic:  Generates realistic fake data for testing (default)

References:
  https://www.kaggle.com/c/criteo-display-ad-challenge
  https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/
"""

import hashlib
import os
import gzip
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Feature schema — 13 numeric + 26 categorical
# ---------------------------------------------------------------------------

NUM_NUMERIC = 13
NUM_CATEGORICAL = 26

NUMERIC_FEATURE_NAMES = [f"I{i}" for i in range(1, NUM_NUMERIC + 1)]
CATEGORICAL_FEATURE_NAMES = [f"C{i}" for i in range(1, NUM_CATEGORICAL + 1)]

# For the feature store, categorical features are hashed to float values
# The combined feature vector is: [13 numeric] + [26 hashed categorical]
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
NUM_FEATURES = len(FEATURE_NAMES)  # 39

LABEL_COL = "click"
ENTITY_NAME = "impression"
VIEW_NAME = "criteo_ctr_v1"
VIEW_VERSION = 1

FEATURE_DESCRIPTIONS = {
    "I1": "Integer feature 1 (quantized count)",
    "I2": "Integer feature 2 (quantized count)",
    "I3": "Integer feature 3 (quantized count)",
    "I4": "Integer feature 4 (quantized count)",
    "I5": "Integer feature 5 (quantized count)",
    "I6": "Integer feature 6 (quantized count)",
    "I7": "Integer feature 7 (quantized count)",
    "I8": "Integer feature 8 (quantized count)",
    "I9": "Integer feature 9 (quantized count)",
    "I10": "Integer feature 10 (quantized count)",
    "I11": "Integer feature 11 (quantized count)",
    "I12": "Integer feature 12 (quantized count)",
    "I13": "Integer feature 13 (quantized count)",
    "C1": "Categorical feature 1 (hashed)",
    "C2": "Categorical feature 2 (hashed)",
    "C3": "Categorical feature 3 (hashed)",
    "C4": "Categorical feature 4 (hashed)",
    "C5": "Categorical feature 5 (hashed)",
    "C6": "Categorical feature 6 (hashed)",
    "C7": "Categorical feature 7 (hashed)",
    "C8": "Categorical feature 8 (hashed)",
    "C9": "Categorical feature 9 (hashed)",
    "C10": "Categorical feature 10 (hashed)",
    "C11": "Categorical feature 11 (hashed)",
    "C12": "Categorical feature 12 (hashed)",
    "C13": "Categorical feature 13 (hashed)",
    "C14": "Categorical feature 14 (hashed)",
    "C15": "Categorical feature 15 (hashed)",
    "C16": "Categorical feature 16 (hashed)",
    "C17": "Categorical feature 17 (hashed)",
    "C18": "Categorical feature 18 (hashed)",
    "C19": "Categorical feature 19 (hashed)",
    "C20": "Categorical feature 20 (hashed)",
    "C21": "Categorical feature 21 (hashed)",
    "C22": "Categorical feature 22 (hashed)",
    "C23": "Categorical feature 23 (hashed)",
    "C24": "Categorical feature 24 (hashed)",
    "C25": "Categorical feature 25 (hashed)",
    "C26": "Categorical feature 26 (hashed)",
}

DATA_DIR = Path(__file__).parent / "data"


def compute_schema_hash(feature_names: list[str]) -> int:
    """Same hash algorithm as the Java side."""
    key = ",".join(feature_names)
    digest = hashlib.md5(key.encode()).hexdigest()[:8]
    return int(digest, 16) % (2**31)


# ---------------------------------------------------------------------------
# Categorical hashing — convert hex strings to float
# ---------------------------------------------------------------------------

def hash_categorical(value: str, n_buckets: int = 10000) -> float:
    """Hash a categorical string value to a float bucket index."""
    if not value or value == "":
        return 0.0
    h = int(hashlib.md5(value.encode()).hexdigest()[:8], 16)
    return float(h % n_buckets)


# ---------------------------------------------------------------------------
# TSV parser for Criteo data
# ---------------------------------------------------------------------------

def parse_criteo_tsv(
    filepath: str,
    max_rows: Optional[int] = None,
    has_label: bool = True,
) -> pd.DataFrame:
    """
    Parse Criteo TSV format:
      label \\t I1 \\t I2 \\t ... \\t I13 \\t C1 \\t C2 \\t ... \\t C26

    Categorical features are hex strings — hashed to float buckets.
    Returns DataFrame with columns: [click, I1..I13, C1..C26, entity_id]
    """
    rows = []
    open_fn = gzip.open if filepath.endswith(".gz") else open

    with open_fn(filepath, "rt") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            parts = line.strip().split("\t")

            if has_label:
                label = int(parts[0])
                feat_start = 1
            else:
                label = -1  # unknown
                feat_start = 0

            # Numeric features (may be empty)
            numeric = []
            for j in range(NUM_NUMERIC):
                idx = feat_start + j
                if idx < len(parts) and parts[idx] != "":
                    try:
                        numeric.append(float(parts[idx]))
                    except ValueError:
                        numeric.append(0.0)
                else:
                    numeric.append(0.0)

            # Categorical features — hash to float
            categorical = []
            for j in range(NUM_CATEGORICAL):
                idx = feat_start + NUM_NUMERIC + j
                if idx < len(parts) and parts[idx] != "":
                    categorical.append(hash_categorical(parts[idx]))
                else:
                    categorical.append(0.0)

            rows.append([label] + numeric + categorical)

    columns = [LABEL_COL] + NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
    df = pd.DataFrame(rows, columns=columns)
    df["entity_id"] = [f"imp_{i:08d}" for i in range(len(df))]
    return df


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_criteo(data_dir: Optional[str] = None) -> str:
    """
    Download Criteo Display Advertising Challenge dataset.

    NOTE: This dataset must be downloaded from Kaggle:
      https://www.kaggle.com/c/criteo-display-ad-challenge/data

    Requires:
      1. Kaggle account and API key (~/.kaggle/kaggle.json)
      2. Accept competition rules on the Kaggle website
      3. Install kaggle CLI: pip install kaggle

    Alternatively, download manually and place train.txt in the data directory.

    Returns path to the data directory.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / "train.txt"

    if train_file.exists():
        print(f"  Dataset already exists at {train_file}")
        return str(data_dir)

    # Try Kaggle CLI download
    print("  Attempting Kaggle CLI download...")
    print("  NOTE: Requires kaggle CLI and accepted competition rules.")
    print("  Install: pip install kaggle")
    print("  Accept rules at: https://www.kaggle.com/c/criteo-display-ad-challenge/rules")
    print()

    try:
        import subprocess
        result = subprocess.run(
            ["kaggle", "competitions", "download",
             "-c", "criteo-display-ad-challenge",
             "-p", str(data_dir)],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            print(f"  Downloaded to {data_dir}")
            # Extract if it's a zip
            import zipfile
            for zf_path in data_dir.glob("*.zip"):
                with zipfile.ZipFile(zf_path, "r") as zf:
                    zf.extractall(data_dir)
            if train_file.exists():
                return str(data_dir)
        else:
            print(f"  Kaggle CLI failed: {result.stderr.strip()}")
    except FileNotFoundError:
        print("  Kaggle CLI not found.")
    except Exception as e:
        print(f"  Kaggle download failed: {e}")

    raise FileNotFoundError(
        f"Could not download Criteo dataset.\n"
        f"Please download manually from:\n"
        f"  https://www.kaggle.com/c/criteo-display-ad-challenge/data\n"
        f"and place train.txt at: {train_file}"
    )


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_samples: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic CTR dataset that mimics Criteo data.

    Produces n_samples ad impressions with realistic:
    - ~3.4% click-through rate (typical for display ads)
    - 13 numeric features (count-like distributions)
    - 26 categorical features (hashed to float buckets)
    - Label correlated with features to make the problem learnable
    """
    rng = np.random.default_rng(seed)

    # --- Numeric features (13) ---
    # Mimic count/frequency distributions from Criteo
    numeric = np.zeros((n_samples, NUM_NUMERIC))

    # Different distributions for different feature groups
    # Features 1-4: small counts (0-100)
    for i in range(4):
        numeric[:, i] = rng.poisson(5 + i * 3, n_samples).astype(float)

    # Features 5-8: medium counts (0-1000)
    for i in range(4, 8):
        numeric[:, i] = np.clip(rng.lognormal(3, 1.5, n_samples), 0, 5000)

    # Features 9-13: large counts / continuous (0-10000)
    for i in range(8, NUM_NUMERIC):
        numeric[:, i] = np.clip(rng.lognormal(5, 2, n_samples), 0, 100000)

    # --- Categorical features (26) ---
    # Simulated as hashed bucket indices (like real Criteo)
    n_buckets = 10000
    categorical = np.zeros((n_samples, NUM_CATEGORICAL))

    for i in range(NUM_CATEGORICAL):
        # Different cardinalities for different categories
        if i < 5:   # low cardinality (e.g., device type, browser)
            n_cats = 20 + i * 10
        elif i < 15: # medium cardinality (e.g., ad category, site)
            n_cats = 200 + i * 50
        else:        # high cardinality (e.g., user ID hashes)
            n_cats = 1000 + i * 100

        # Zipf-like distribution (few popular categories)
        cats = rng.zipf(1.5, n_samples) % n_cats
        categorical[:, i] = cats.astype(float)

    # --- Click label ---
    # Latent CTR score combines numeric and categorical signals
    latent = np.zeros(n_samples)

    # Numeric contributions (strong signal features)
    latent += np.log1p(numeric[:, 0]) * 0.8   # I1 is the most important
    latent += np.log1p(numeric[:, 2]) * 0.6    # I3
    latent -= np.log1p(numeric[:, 4]) * 0.3    # I5 negatively correlated
    latent += (numeric[:, 7] > np.median(numeric[:, 7])).astype(float) * 1.0  # I8
    latent += np.log1p(numeric[:, 9]) * 0.4    # I10
    latent -= (numeric[:, 11] > np.percentile(numeric[:, 11], 75)).astype(float) * 0.5  # I12

    # Categorical contributions (certain categories boost CTR)
    for i in [0, 2, 5]:
        latent += (categorical[:, i] < 3).astype(float) * 0.8
    for i in [10, 14, 20]:
        latent += (categorical[:, i] < 10).astype(float) * 0.5

    latent += rng.normal(0, 0.8, n_samples)  # moderate noise

    # Convert to click probability via sigmoid
    # Offset calibrated to produce ~3-5% CTR
    click_prob = 1 / (1 + np.exp(-(latent - 10)))
    click = rng.binomial(1, click_prob)

    # Build DataFrame
    df = pd.DataFrame(numeric, columns=NUMERIC_FEATURE_NAMES)
    for i, name in enumerate(CATEGORICAL_FEATURE_NAMES):
        df[name] = categorical[:, i]
    df.insert(0, LABEL_COL, click)
    df["entity_id"] = [f"imp_{i:08d}" for i in range(n_samples)]

    return df


# ---------------------------------------------------------------------------
# Load dataset (real or synthetic)
# ---------------------------------------------------------------------------

def load_dataset(
    synthetic: bool = True,
    data_dir: Optional[str] = None,
    max_rows: Optional[int] = None,
    n_samples: int = 100_000,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the Criteo CTR dataset (real or synthetic).

    Returns (train_df, test_df).
    Each has columns: [click, I1..I13, C1..C26, entity_id]
    """
    if synthetic:
        print("  Generating synthetic Criteo CTR dataset...")
        df = generate_synthetic_dataset(n_samples, seed)

        # Time-based split (chronological — last N% is test)
        split_idx = int(len(df) * (1 - test_fraction))
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)

        ctr_train = train_df[LABEL_COL].mean() * 100
        ctr_test = test_df[LABEL_COL].mean() * 100
        print(f"  Train: {len(train_df):,} rows (CTR: {ctr_train:.2f}%)")
        print(f"  Test:  {len(test_df):,} rows (CTR: {ctr_test:.2f}%)")
        return train_df, test_df

    # Real data
    data_dir_path = Path(data_dir) if data_dir else DATA_DIR
    train_file = data_dir_path / "train.txt"

    if not train_file.exists():
        download_criteo(data_dir)

    print(f"  Parsing {train_file}...")
    df = parse_criteo_tsv(str(train_file), max_rows=max_rows)
    print(f"    {len(df):,} rows, CTR: {df[LABEL_COL].mean()*100:.2f}%")

    # Split
    split_idx = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    print(f"  Train: {len(train_df):,} rows")
    print(f"  Test:  {len(test_df):,} rows")
    return train_df, test_df


# ---------------------------------------------------------------------------
# Feature store integration
# ---------------------------------------------------------------------------

def register_with_feature_store(
    base_url: str = "http://localhost:8085",
) -> dict:
    """Register entity, features, and feature view via REST API."""
    headers = {"Content-Type": "application/json"}

    # 1. Create entity
    resp = requests.post(f"{base_url}/api/v1/entities", json={
        "name": ENTITY_NAME,
        "description": "Ad impression for CTR prediction",
        "joinKey": "impression_id",
        "joinKeyType": "STRING",
    }, headers=headers)
    if resp.status_code == 201:
        entity_id = resp.json()["id"]
        print(f"  Created entity: {entity_id}")
    elif resp.status_code == 409 or "already exists" in resp.text.lower():
        resp2 = requests.get(f"{base_url}/api/v1/entities/by-name/{ENTITY_NAME}")
        entity_id = resp2.json()["id"]
        print(f"  Entity already exists: {entity_id}")
    else:
        resp.raise_for_status()
        entity_id = resp.json()["id"]

    # 2. Create features
    feature_ids = []
    for name in FEATURE_NAMES:
        desc = FEATURE_DESCRIPTIONS.get(name, f"Criteo feature {name}")
        dtype = "FLOAT64"  # All stored as float (categoricals are hashed)
        resp = requests.post(f"{base_url}/api/v1/features", json={
            "name": name,
            "entityId": entity_id,
            "dtype": dtype,
            "description": desc,
            "owner": "ads-ml-team",
            "sourcePipeline": "criteo_ctr",
            "updateFrequency": "REALTIME",
            "maxAgeSeconds": 3600,
            "defaultValue": "0.0",
        }, headers=headers)
        if resp.status_code == 201:
            feature_ids.append(resp.json()["id"])
        else:
            features_resp = requests.get(
                f"{base_url}/api/v1/features", params={"entityId": entity_id}
            )
            if features_resp.ok:
                for f in features_resp.json():
                    if f["name"] == name:
                        feature_ids.append(f["id"])
                        break

    print(f"  Registered {len(feature_ids)} / {NUM_FEATURES} features")

    # 3. Create feature view
    resp = requests.post(f"{base_url}/api/v1/feature-views", json={
        "name": VIEW_NAME,
        "version": VIEW_VERSION,
        "entityId": entity_id,
        "description": "Criteo CTR features (13 numeric + 26 categorical hashed)",
        "modelName": "criteo_ctr_xgb",
        "mlFramework": "XGBOOST",
        "featureIds": feature_ids,
    }, headers=headers)
    if resp.status_code == 201:
        view = resp.json()
        print(f"  Created view: {view['name']} v{view['version']} "
              f"(hash={view['schemaHash']}, len={view['vectorLength']})")
    else:
        print(f"  View may already exist ({resp.status_code})")

    return {"entity_id": entity_id, "feature_ids": feature_ids}


def materialize_to_feature_store(
    df: pd.DataFrame,
    grpc_target: str = "localhost:9090",
    batch_size: int = 500,
    max_vectors: Optional[int] = None,
) -> int:
    """Materialize feature vectors to the feature store via gRPC."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from grpc_client import FeatureStoreClient

    schema_hash = compute_schema_hash(FEATURE_NAMES)
    client = FeatureStoreClient(grpc_target)
    total = 0

    rows = df if max_vectors is None else df.head(max_vectors)

    for start in range(0, len(rows), batch_size):
        chunk = rows.iloc[start:start + batch_size]
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

        if (start // batch_size) % 20 == 0 and start > 0:
            print(f"    Materialized {total:,} / {len(rows):,} vectors...")

    client.close()
    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Criteo CTR dataset loader")
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--no-synthetic", dest="synthetic", action="store_false")
    parser.add_argument("--n-samples", type=int, default=100_000)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    train_df, test_df = load_dataset(
        synthetic=args.synthetic,
        n_samples=args.n_samples,
        max_rows=args.max_rows,
    )

    print(f"\nTrain: {len(train_df):,} rows, CTR: {train_df[LABEL_COL].mean()*100:.2f}%")
    print(f"Test:  {len(test_df):,} rows, CTR: {test_df[LABEL_COL].mean()*100:.2f}%")
    print(f"\nNumeric feature stats:\n{train_df[NUMERIC_FEATURE_NAMES].describe()}")
    print(f"\nCategorical feature cardinalities:")
    for name in CATEGORICAL_FEATURE_NAMES[:5]:
        print(f"  {name}: {train_df[name].nunique()} unique values")
