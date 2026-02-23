"""
Real-time inference using the feature store + trained GBDT model.

Demonstrates two inference patterns:
  1. Online scoring — fetch features from the feature store via gRPC, score with GBDT
  2. Batch scoring  — fetch a batch of merchants, score all at once

Also supports offline inference directly from a DataFrame (no feature store needed).
"""

import sys
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb

from dataset import FEATURE_NAMES, VIEW_NAME, VIEW_VERSION, ENTITY_NAME, compute_schema_hash

# Ensure grpc_client is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ScoringResult:
    """Result of scoring a single entity."""
    entity_id: str
    risk_probability: float
    risk_label: str          # HIGH / MEDIUM / LOW
    feature_values: list
    feature_names: list
    latency_ms: float
    source: str              # FEATURE_STORE or DATAFRAME


@dataclass
class BatchScoringResult:
    """Result of scoring a batch of entities."""
    results: list[ScoringResult]
    total_latency_ms: float
    avg_latency_ms: float
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int


# ---------------------------------------------------------------------------
# Risk classification thresholds
# ---------------------------------------------------------------------------

HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.3


def classify_risk(prob: float) -> str:
    if prob >= HIGH_RISK_THRESHOLD:
        return "HIGH"
    elif prob >= MEDIUM_RISK_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_dir: str = "models") -> xgb.XGBClassifier:
    """Load trained GBDT model from disk."""
    model_path = Path(model_dir) / "model.ubj"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py first."
        )
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))
    return clf


# ---------------------------------------------------------------------------
# Online inference via feature store (gRPC)
# ---------------------------------------------------------------------------

def score_entity(
    model: xgb.XGBClassifier,
    entity_id: str,
    grpc_target: str = "localhost:9090",
) -> ScoringResult:
    """
    Score a single merchant by fetching features from the feature store.

    Flow:
      1. gRPC GetOnlineFeatures → feature vector
      2. XGBoost predict_proba → risk probability
      3. Classify into HIGH / MEDIUM / LOW
    """
    from grpc_client import FeatureStoreClient

    t0 = time.time()

    client = FeatureStoreClient(grpc_target)
    result = client.get_online_features(VIEW_NAME, VIEW_VERSION, [entity_id])
    client.close()

    if not result["vectors"]:
        return ScoringResult(
            entity_id=entity_id,
            risk_probability=0.0,
            risk_label="UNKNOWN",
            feature_values=[],
            feature_names=FEATURE_NAMES,
            latency_ms=(time.time() - t0) * 1000,
            source="FEATURE_STORE",
        )

    vector = result["vectors"][0]
    values = np.array(vector["values"]).reshape(1, -1)

    prob = float(model.predict_proba(values)[0, 1])
    latency_ms = (time.time() - t0) * 1000

    return ScoringResult(
        entity_id=entity_id,
        risk_probability=prob,
        risk_label=classify_risk(prob),
        feature_values=vector["values"],
        feature_names=FEATURE_NAMES,
        latency_ms=latency_ms,
        source="FEATURE_STORE",
    )


def score_batch(
    model: xgb.XGBClassifier,
    entity_ids: list[str],
    grpc_target: str = "localhost:9090",
) -> BatchScoringResult:
    """
    Score a batch of merchants using the feature store.

    Fetches all feature vectors in a single gRPC batch call, then runs
    XGBoost prediction on the full batch at once for efficiency.
    """
    from grpc_client import FeatureStoreClient

    t0 = time.time()

    client = FeatureStoreClient(grpc_target)
    result = client.get_online_features(VIEW_NAME, VIEW_VERSION, entity_ids)
    client.close()

    # Build feature matrix from returned vectors
    entity_to_values = {}
    for v in result["vectors"]:
        entity_to_values[v["entity_id"]] = v["values"]

    # Batch predict
    scoring_results = []
    found_ids = [eid for eid in entity_ids if eid in entity_to_values]

    if found_ids:
        X = np.array([entity_to_values[eid] for eid in found_ids])
        probs = model.predict_proba(X)[:, 1]

        for eid, prob in zip(found_ids, probs):
            scoring_results.append(ScoringResult(
                entity_id=eid,
                risk_probability=float(prob),
                risk_label=classify_risk(float(prob)),
                feature_values=entity_to_values[eid],
                feature_names=FEATURE_NAMES,
                latency_ms=0,  # set at batch level
                source="FEATURE_STORE",
            ))

    # Mark missing entities
    for eid in entity_ids:
        if eid not in entity_to_values:
            scoring_results.append(ScoringResult(
                entity_id=eid,
                risk_probability=0.0,
                risk_label="UNKNOWN",
                feature_values=[],
                feature_names=FEATURE_NAMES,
                latency_ms=0,
                source="FEATURE_STORE",
            ))

    total_ms = (time.time() - t0) * 1000

    return BatchScoringResult(
        results=scoring_results,
        total_latency_ms=total_ms,
        avg_latency_ms=total_ms / max(len(entity_ids), 1),
        high_risk_count=sum(1 for r in scoring_results if r.risk_label == "HIGH"),
        medium_risk_count=sum(1 for r in scoring_results if r.risk_label == "MEDIUM"),
        low_risk_count=sum(1 for r in scoring_results if r.risk_label == "LOW"),
    )


# ---------------------------------------------------------------------------
# Offline inference from DataFrame (no feature store needed)
# ---------------------------------------------------------------------------

def score_dataframe(
    model: xgb.XGBClassifier,
    df,  # pd.DataFrame
) -> "pd.DataFrame":
    """
    Score all rows in a DataFrame. Adds risk_probability and risk_label columns.
    Useful for batch scoring or backtesting without a running feature store.
    """
    import pandas as pd

    X = df[FEATURE_NAMES].fillna(0).values
    probs = model.predict_proba(X)[:, 1]

    result_df = df.copy()
    result_df["risk_probability"] = probs
    result_df["risk_label"] = [classify_risk(p) for p in probs]

    return result_df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_scoring_result(result: ScoringResult):
    """Pretty-print a single scoring result."""
    print(f"  Entity:      {result.entity_id}")
    print(f"  Risk:        {result.risk_label} ({result.risk_probability:.4f})")
    print(f"  Source:      {result.source}")
    print(f"  Latency:     {result.latency_ms:.1f}ms")
    if result.feature_values:
        print(f"  Features:    {len(result.feature_values)} values")
        # Show top contributing features
        top_features = sorted(
            zip(result.feature_names, result.feature_values),
            key=lambda x: abs(x[1]), reverse=True
        )[:5]
        for name, val in top_features:
            print(f"    {name}: {val:.4f}")


def print_batch_result(result: BatchScoringResult):
    """Pretty-print batch scoring summary."""
    print(f"  Total scored:    {len(result.results)}")
    print(f"  Total latency:   {result.total_latency_ms:.1f}ms")
    print(f"  Avg latency:     {result.avg_latency_ms:.1f}ms per entity")
    print(f"  Risk breakdown:")
    print(f"    HIGH:   {result.high_risk_count}")
    print(f"    MEDIUM: {result.medium_risk_count}")
    print(f"    LOW:    {result.low_risk_count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score merchants with trained GBDT")
    parser.add_argument("--model-dir", default="models", help="Model directory")
    parser.add_argument("--entity-ids", nargs="+", help="Merchant IDs to score")
    parser.add_argument("--grpc-target", default="localhost:9090", help="gRPC target")
    parser.add_argument("--offline", action="store_true",
                        help="Score from generated data instead of feature store")
    parser.add_argument("--n-merchants", type=int, default=100,
                        help="Number of merchants for offline scoring")
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.model_dir)

    if args.offline:
        from dataset import generate_dataset
        print(f"\nOffline scoring {args.n_merchants} merchants...")
        df = generate_dataset(args.n_merchants)
        scored = score_dataframe(model, df)

        high = (scored["risk_label"] == "HIGH").sum()
        med = (scored["risk_label"] == "MEDIUM").sum()
        low = (scored["risk_label"] == "LOW").sum()
        print(f"  Results: HIGH={high}, MEDIUM={med}, LOW={low}")
        print(f"\n  Top 5 highest risk:")
        top5 = scored.nlargest(5, "risk_probability")
        for _, row in top5.iterrows():
            print(f"    {row['entity_id']}: {row['risk_label']} "
                  f"(p={row['risk_probability']:.4f})")
    else:
        entity_ids = args.entity_ids or ["m_000001", "m_000010", "m_000100"]
        print(f"\nScoring {len(entity_ids)} merchants via feature store...")
        batch_result = score_batch(model, entity_ids, args.grpc_target)
        print_batch_result(batch_result)
        print("\n  Individual results:")
        for r in batch_result.results:
            print(f"    {r.entity_id}: {r.risk_label} (p={r.risk_probability:.4f})")
