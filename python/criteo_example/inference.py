"""
Real-time CTR inference using the feature store + trained XGBoost model.

Demonstrates two inference patterns:
  1. Online scoring — fetch features from feature store, predict CTR
  2. Batch scoring  — score a batch of impressions at once

Also supports offline inference directly from a DataFrame.
"""

import sys
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from dataset import FEATURE_NAMES, LABEL_COL, VIEW_NAME, VIEW_VERSION, ENTITY_NAME, compute_schema_hash

# Ensure grpc_client is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class CTRScoringResult:
    """Result of scoring a single impression."""
    entity_id: str
    click_probability: float
    bid_recommendation: str   # HIGH_BID / MEDIUM_BID / LOW_BID / NO_BID
    feature_values: list
    feature_names: list
    latency_ms: float
    source: str


@dataclass
class BatchCTRResult:
    """Result of scoring a batch of impressions."""
    results: list[CTRScoringResult]
    total_latency_ms: float
    avg_latency_ms: float
    mean_ctr: float
    high_bid_count: int
    medium_bid_count: int
    low_bid_count: int
    no_bid_count: int


# ---------------------------------------------------------------------------
# Bid recommendation thresholds
# ---------------------------------------------------------------------------

HIGH_BID_THRESHOLD = 0.10    # >10% predicted CTR → aggressive bid
MEDIUM_BID_THRESHOLD = 0.05  # >5% CTR → normal bid
LOW_BID_THRESHOLD = 0.02     # >2% CTR → conservative bid
                              # <2% → don't bid


def classify_bid(ctr: float) -> str:
    if ctr >= HIGH_BID_THRESHOLD:
        return "HIGH_BID"
    elif ctr >= MEDIUM_BID_THRESHOLD:
        return "MEDIUM_BID"
    elif ctr >= LOW_BID_THRESHOLD:
        return "LOW_BID"
    else:
        return "NO_BID"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_dir: str = "models") -> xgb.XGBClassifier:
    """Load trained CTR model from disk."""
    model_path = Path(model_dir) / "model.ubj"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py first."
        )
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))
    return clf


# ---------------------------------------------------------------------------
# Online scoring via feature store
# ---------------------------------------------------------------------------

def score_impression(
    model: xgb.XGBClassifier,
    entity_id: str,
    grpc_target: str = "localhost:9090",
) -> CTRScoringResult:
    """Score a single impression by fetching features from the feature store."""
    from grpc_client import FeatureStoreClient

    t0 = time.time()

    client = FeatureStoreClient(grpc_target)
    result = client.get_online_features(VIEW_NAME, VIEW_VERSION, [entity_id])
    client.close()

    if not result["vectors"]:
        return CTRScoringResult(
            entity_id=entity_id,
            click_probability=0.0,
            bid_recommendation="NO_BID",
            feature_values=[],
            feature_names=FEATURE_NAMES,
            latency_ms=(time.time() - t0) * 1000,
            source="FEATURE_STORE",
        )

    vector = result["vectors"][0]
    values = np.array(vector["values"]).reshape(1, -1)
    ctr = float(model.predict_proba(values)[0, 1])

    return CTRScoringResult(
        entity_id=entity_id,
        click_probability=ctr,
        bid_recommendation=classify_bid(ctr),
        feature_values=vector["values"],
        feature_names=FEATURE_NAMES,
        latency_ms=(time.time() - t0) * 1000,
        source="FEATURE_STORE",
    )


def score_batch(
    model: xgb.XGBClassifier,
    entity_ids: list[str],
    grpc_target: str = "localhost:9090",
) -> BatchCTRResult:
    """Score a batch of impressions using the feature store."""
    from grpc_client import FeatureStoreClient

    t0 = time.time()

    client = FeatureStoreClient(grpc_target)
    result = client.get_online_features(VIEW_NAME, VIEW_VERSION, entity_ids)
    client.close()

    entity_to_values = {}
    for v in result["vectors"]:
        entity_to_values[v["entity_id"]] = v["values"]

    scoring_results = []
    found_ids = [eid for eid in entity_ids if eid in entity_to_values]

    if found_ids:
        X = np.array([entity_to_values[eid] for eid in found_ids])
        probs = model.predict_proba(X)[:, 1]

        for eid, prob in zip(found_ids, probs):
            scoring_results.append(CTRScoringResult(
                entity_id=eid,
                click_probability=float(prob),
                bid_recommendation=classify_bid(float(prob)),
                feature_values=entity_to_values[eid],
                feature_names=FEATURE_NAMES,
                latency_ms=0,
                source="FEATURE_STORE",
            ))

    for eid in entity_ids:
        if eid not in entity_to_values:
            scoring_results.append(CTRScoringResult(
                entity_id=eid,
                click_probability=0.0,
                bid_recommendation="NO_BID",
                feature_values=[],
                feature_names=FEATURE_NAMES,
                latency_ms=0,
                source="FEATURE_STORE",
            ))

    total_ms = (time.time() - t0) * 1000
    ctrs = [r.click_probability for r in scoring_results]

    return BatchCTRResult(
        results=scoring_results,
        total_latency_ms=total_ms,
        avg_latency_ms=total_ms / max(len(entity_ids), 1),
        mean_ctr=float(np.mean(ctrs)) if ctrs else 0.0,
        high_bid_count=sum(1 for r in scoring_results if r.bid_recommendation == "HIGH_BID"),
        medium_bid_count=sum(1 for r in scoring_results if r.bid_recommendation == "MEDIUM_BID"),
        low_bid_count=sum(1 for r in scoring_results if r.bid_recommendation == "LOW_BID"),
        no_bid_count=sum(1 for r in scoring_results if r.bid_recommendation == "NO_BID"),
    )


# ---------------------------------------------------------------------------
# Offline scoring from DataFrame
# ---------------------------------------------------------------------------

def score_dataframe(
    model: xgb.XGBClassifier,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Score all impressions in a DataFrame. Adds click_probability and bid_recommendation."""
    X = df[FEATURE_NAMES].fillna(0).values
    probs = model.predict_proba(X)[:, 1]

    result_df = df.copy()
    result_df["click_probability"] = probs
    result_df["bid_recommendation"] = [classify_bid(p) for p in probs]
    return result_df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_scoring_result(result: CTRScoringResult):
    """Pretty-print a single scoring result."""
    print(f"  Impression: {result.entity_id}")
    print(f"  CTR:        {result.click_probability:.4f} ({result.click_probability*100:.2f}%)")
    print(f"  Bid:        {result.bid_recommendation}")
    print(f"  Source:     {result.source}")
    print(f"  Latency:    {result.latency_ms:.1f}ms")


def print_batch_result(result: BatchCTRResult):
    """Pretty-print batch scoring summary."""
    print(f"  Impressions scored: {len(result.results)}")
    print(f"  Mean CTR:           {result.mean_ctr*100:.2f}%")
    print(f"  Total latency:      {result.total_latency_ms:.1f}ms")
    print(f"  Avg latency:        {result.avg_latency_ms:.1f}ms per impression")
    print(f"  Bid recommendations:")
    print(f"    HIGH_BID:   {result.high_bid_count}")
    print(f"    MEDIUM_BID: {result.medium_bid_count}")
    print(f"    LOW_BID:    {result.low_bid_count}")
    print(f"    NO_BID:     {result.no_bid_count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score impressions with trained CTR model")
    parser.add_argument("--model-dir", default="models", help="Model directory")
    parser.add_argument("--entity-ids", nargs="+", help="Impression IDs to score")
    parser.add_argument("--grpc-target", default="localhost:9090", help="gRPC target")
    parser.add_argument("--offline", action="store_true",
                        help="Score from generated data instead of feature store")
    parser.add_argument("--n-samples", type=int, default=10_000,
                        help="Number of samples for offline scoring")
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.model_dir)

    if args.offline:
        from dataset import load_dataset
        print(f"\nOffline scoring {args.n_samples} impressions...")
        _, test_df = load_dataset(synthetic=True, n_samples=args.n_samples)
        scored = score_dataframe(model, test_df)

        print(f"  Mean CTR: {scored['click_probability'].mean()*100:.2f}%")
        print(f"  Bid breakdown:")
        for bid in ["HIGH_BID", "MEDIUM_BID", "LOW_BID", "NO_BID"]:
            count = (scored["bid_recommendation"] == bid).sum()
            pct = count / len(scored) * 100
            print(f"    {bid}: {count:,} ({pct:.1f}%)")

        print(f"\n  Top 5 highest CTR:")
        top5 = scored.nlargest(5, "click_probability")
        for _, row in top5.iterrows():
            print(f"    {row['entity_id']}: {row['bid_recommendation']} "
                  f"(CTR={row['click_probability']*100:.2f}%)")
    else:
        entity_ids = args.entity_ids or ["imp_00000001", "imp_00000010", "imp_00000100"]
        print(f"\nScoring {len(entity_ids)} impressions via feature store...")
        batch_result = score_batch(model, entity_ids, args.grpc_target)
        print_batch_result(batch_result)
        print("\n  Individual results:")
        for r in batch_result.results:
            print(f"    {r.entity_id}: {r.bid_recommendation} (CTR={r.click_probability*100:.2f}%)")
