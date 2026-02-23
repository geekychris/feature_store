"""
MSLR-WEB10K Learning to Rank dataset for the Feature Store.

Microsoft Learning to Rank dataset with 136 features per query-document pair
and relevance labels 0-4. Features include TF, IDF, BM25, PageRank, URL/body
statistics across single terms, term pairs, and whole queries.

Two modes:
  - Real data:  Downloads MSLR-WEB10K from Microsoft Research (~1.2 GB)
  - Synthetic:  Generates realistic fake data for testing (default)

References:
  https://www.microsoft.com/en-us/research/project/mslr/
"""

import hashlib
import os
import time
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Feature schema — 136 features from MSLR-WEB10K
# ---------------------------------------------------------------------------

# The 136 features are grouped into categories (per Microsoft documentation):
#   1-5:   covered query term number / ratio (stream: body, anchor, title, URL, doc)
#   6-10:  TF (body, anchor, title, URL, doc)
#   11-15: TF*IDF (body, anchor, title, URL, doc)
#   16-20: BM25 (body, anchor, title, URL, doc)
#   21-25: LMIR.ABS (body, anchor, title, URL, doc)
#   26-30: LMIR.DIR (body, anchor, title, URL, doc)
#   31-35: LMIR.JM (body, anchor, title, URL, doc)
#   36-40: same for min of term pairs
#   41-45: same for max of term pairs
#   ... repeated for different aggregation types (sum, min, max, mean, var)
#   131:   PageRank
#   132:   Inlink number
#   133:   Outlink number
#   134:   Number of slash in URL
#   135:   URL length
#   136:   Number of child pages

NUM_FEATURES = 136
FEATURE_NAMES = [f"feature_{i}" for i in range(1, NUM_FEATURES + 1)]
LABEL_COL = "relevance"
ENTITY_NAME = "query_doc_pair"
VIEW_NAME = "mslr_ranking_v1"
VIEW_VERSION = 1

# Feature descriptions for the important ones (too many for all 136)
FEATURE_DESCRIPTIONS = {
    "feature_1": "Covered query term number (body)",
    "feature_6": "Term frequency (body)",
    "feature_11": "TF*IDF (body)",
    "feature_16": "BM25 (body)",
    "feature_21": "LMIR.ABS (body)",
    "feature_26": "LMIR.DIR (body)",
    "feature_31": "LMIR.JM (body)",
    "feature_131": "PageRank",
    "feature_132": "Inlink number",
    "feature_133": "Outlink number",
    "feature_134": "Number of slashes in URL",
    "feature_135": "URL length",
    "feature_136": "Number of child pages",
}

DATA_DIR = Path(__file__).parent / "data"
MSLR_URL = "https://api.onedrive.com/v1.0/shares/s!AtGnbS_VBJnGkxMEg5UZCxPIU3N4/root/content"


def compute_schema_hash(feature_names: list[str]) -> int:
    """Same hash algorithm as the Java side."""
    key = ",".join(feature_names)
    digest = hashlib.md5(key.encode()).hexdigest()[:8]
    return int(digest, 16) % (2**31)


# ---------------------------------------------------------------------------
# SVM-light parser
# ---------------------------------------------------------------------------

def parse_svmlight_file(filepath: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Parse MSLR SVM-light format:
      relevance qid:N 1:val 2:val ... 136:val

    Returns DataFrame with columns: [qid, relevance, feature_1..feature_136]
    """
    rows = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            relevance = int(parts[0])
            qid = int(parts[1].split(":")[1])

            features = [0.0] * NUM_FEATURES
            for token in parts[2:]:
                if ":" in token:
                    idx_str, val_str = token.split(":", 1)
                    idx = int(idx_str) - 1  # 0-indexed
                    if 0 <= idx < NUM_FEATURES:
                        features[idx] = float(val_str)

            rows.append([qid, relevance] + features)

    df = pd.DataFrame(rows, columns=["qid", LABEL_COL] + FEATURE_NAMES)
    df["qid"] = df["qid"].astype(int)
    df["entity_id"] = df.apply(lambda r: f"q{int(r['qid'])}_d{r.name}", axis=1)
    return df


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_mslr(data_dir: Optional[str] = None, fold: int = 1) -> str:
    """
    Download MSLR-WEB10K Fold1 from Microsoft Research.

    NOTE: This dataset requires accepting the Microsoft Research license.
    If automatic download fails, manually download from:
      https://www.microsoft.com/en-us/research/project/mslr/

    Returns path to the extracted fold directory.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    fold_dir = data_dir / f"Fold{fold}"
    train_file = fold_dir / "train.txt"

    if train_file.exists():
        print(f"  Dataset already exists at {fold_dir}")
        return str(fold_dir)

    zip_path = data_dir / "MSLR-WEB10K.zip"

    if not zip_path.exists():
        print(f"  Downloading MSLR-WEB10K (~1.2 GB)...")
        print(f"  URL: {MSLR_URL}")
        print(f"  NOTE: If download fails, download manually from:")
        print(f"    https://www.microsoft.com/en-us/research/project/mslr/")
        print(f"  and place the zip at: {zip_path}")

        try:
            resp = requests.get(MSLR_URL, stream=True, timeout=30)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            downloaded = 0

            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192 * 16):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        print(f"\r  Downloaded: {downloaded / 1e6:.0f} MB / "
                              f"{total / 1e6:.0f} MB ({pct:.0f}%)", end="", flush=True)
            print()
        except Exception as e:
            if zip_path.exists():
                zip_path.unlink()
            raise RuntimeError(
                f"Failed to download MSLR-WEB10K: {e}\n"
                f"Download manually from https://www.microsoft.com/en-us/research/project/mslr/\n"
                f"and place the zip at: {zip_path}"
            )

    # Extract
    print(f"  Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    if not train_file.exists():
        raise FileNotFoundError(
            f"Expected {train_file} after extraction. "
            f"Check the zip contents and folder structure."
        )

    print(f"  Extracted to {fold_dir}")
    return str(fold_dir)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_queries: int = 200,
    docs_per_query: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic Learning to Rank dataset that mimics MSLR-WEB10K.

    Produces n_queries queries, each with docs_per_query candidate documents.
    Relevance labels are 0-4 with a realistic distribution skewed toward 0.
    Feature values are correlated with relevance to make the problem learnable.
    """
    rng = np.random.default_rng(seed)
    n_total = n_queries * docs_per_query

    # Query IDs
    qids = np.repeat(np.arange(1, n_queries + 1), docs_per_query)

    # Latent relevance (continuous) — used to generate correlated features
    latent = rng.normal(0, 1, n_total)

    # Generate 136 features with varying correlation to latent relevance
    features = np.zeros((n_total, NUM_FEATURES))

    for i in range(NUM_FEATURES):
        # Different feature groups have different correlation strengths
        if i < 5:      # covered query terms — strong signal
            corr = 0.5
            noise_scale = 1.0
        elif i < 35:    # TF, TF*IDF, BM25, LMIR variants — moderate signal
            corr = 0.3
            noise_scale = 2.0
        elif i < 130:   # aggregated variants — weaker signal
            corr = 0.15
            noise_scale = 3.0
        else:           # PageRank, link features — moderate signal
            corr = 0.35
            noise_scale = 1.5

        features[:, i] = latent * corr + rng.normal(0, noise_scale, n_total)

    # Make some features non-negative (like real TF, PageRank)
    for i in list(range(0, 5)) + list(range(130, 136)):
        features[:, i] = np.abs(features[:, i])

    # PageRank (feature 131, idx 130) — log-normal distribution
    features[:, 130] = np.exp(features[:, 130] * 0.5)

    # Link counts — Poisson-like
    for i in [131, 132, 135]:
        features[:, i] = np.abs(features[:, i]).round()

    # Generate relevance labels 0-4 from latent score
    # Distribution: 0 (60%), 1 (20%), 2 (12%), 3 (5%), 4 (3%)
    thresholds = np.array([
        np.quantile(latent, 0.60),
        np.quantile(latent, 0.80),
        np.quantile(latent, 0.92),
        np.quantile(latent, 0.97),
    ])
    relevance = np.digitize(latent, thresholds)

    # Entity IDs
    entity_ids = [f"q{qid}_d{j}" for qid, j in zip(qids, range(n_total))]

    df = pd.DataFrame(features, columns=FEATURE_NAMES)
    df.insert(0, "qid", qids)
    df.insert(1, LABEL_COL, relevance)
    df["entity_id"] = entity_ids

    return df


# ---------------------------------------------------------------------------
# Load dataset (real or synthetic)
# ---------------------------------------------------------------------------

def load_dataset(
    synthetic: bool = True,
    data_dir: Optional[str] = None,
    fold: int = 1,
    max_rows: Optional[int] = None,
    n_queries: int = 200,
    docs_per_query: int = 50,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the MSLR-WEB10K dataset (real or synthetic).

    Returns (train_df, test_df) where test_df is the vali.txt split.
    Each has columns: [qid, relevance, feature_1..feature_136, entity_id]
    """
    if synthetic:
        print("  Generating synthetic MSLR-WEB10K dataset...")
        df = generate_synthetic_dataset(n_queries, docs_per_query, seed)

        # Split by query ID
        all_qids = df["qid"].unique()
        rng = np.random.default_rng(seed)
        rng.shuffle(all_qids)
        split = int(len(all_qids) * 0.8)
        train_qids = set(all_qids[:split])
        test_qids = set(all_qids[split:])

        train_df = df[df["qid"].isin(train_qids)].reset_index(drop=True)
        test_df = df[df["qid"].isin(test_qids)].reset_index(drop=True)

        print(f"  Train: {len(train_df):,} rows, {len(train_qids)} queries")
        print(f"  Test:  {len(test_df):,} rows, {len(test_qids)} queries")
        return train_df, test_df

    # Real data
    fold_dir = download_mslr(data_dir, fold)
    fold_path = Path(fold_dir)

    print(f"  Parsing train.txt...")
    train_df = parse_svmlight_file(str(fold_path / "train.txt"), max_rows)
    print(f"    {len(train_df):,} rows, {train_df['qid'].nunique()} queries")

    print(f"  Parsing vali.txt...")
    test_df = parse_svmlight_file(str(fold_path / "vali.txt"), max_rows)
    print(f"    {len(test_df):,} rows, {test_df['qid'].nunique()} queries")

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
        "description": "Query-document pair for Learning to Rank",
        "joinKey": "query_doc_id",
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

    # 2. Create features (all 136 are FLOAT64)
    feature_ids = []
    for i, name in enumerate(FEATURE_NAMES):
        desc = FEATURE_DESCRIPTIONS.get(name, f"MSLR-WEB10K feature {i+1}")
        resp = requests.post(f"{base_url}/api/v1/features", json={
            "name": name,
            "entityId": entity_id,
            "dtype": "FLOAT64",
            "description": desc,
            "owner": "ranking-team",
            "sourcePipeline": "mslr_ranking",
            "updateFrequency": "DAILY",
            "maxAgeSeconds": 86400,
            "defaultValue": "0.0",
        }, headers=headers)
        if resp.status_code == 201:
            feature_ids.append(resp.json()["id"])
        else:
            # Try to look up existing feature
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
        "description": "MSLR-WEB10K Learning to Rank features (136-dim)",
        "modelName": "mslr_lambdamart",
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

        if (start // batch_size) % 10 == 0 and start > 0:
            print(f"    Materialized {total:,} / {len(rows):,} vectors...")

    client.close()
    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MSLR-WEB10K dataset loader")
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--no-synthetic", dest="synthetic", action="store_false")
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--docs-per-query", type=int, default=50)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    train_df, test_df = load_dataset(
        synthetic=args.synthetic,
        n_queries=args.n_queries,
        docs_per_query=args.docs_per_query,
        max_rows=args.max_rows,
    )

    print(f"\nTrain: {len(train_df):,} rows, {train_df['qid'].nunique()} queries")
    print(f"Test:  {len(test_df):,} rows, {test_df['qid'].nunique()} queries")
    print(f"\nRelevance distribution (train):")
    for label in sorted(train_df[LABEL_COL].unique()):
        count = (train_df[LABEL_COL] == label).sum()
        pct = count / len(train_df) * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")
    print(f"\nSample features:\n{train_df[FEATURE_NAMES[:5]].describe()}")
