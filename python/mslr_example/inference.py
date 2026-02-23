"""
Ranking inference using the feature store + trained LambdaMART model.

Demonstrates two inference patterns:
  1. Online re-ranking — fetch features for query-doc pairs, rank by score
  2. Batch evaluation  — score and evaluate a full test set

Also supports offline inference directly from a DataFrame (no feature store needed).
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
class RankingResult:
    """Result of ranking a set of documents for a query."""
    query_id: int
    ranked_entity_ids: list[str]
    scores: list[float]
    relevances: list[int]  # if known
    ndcg_5: Optional[float]
    latency_ms: float
    source: str  # FEATURE_STORE or DATAFRAME


@dataclass
class BatchRankingResult:
    """Result of evaluating ranking across multiple queries."""
    results: list[RankingResult]
    mean_ndcg_5: float
    mean_ndcg_10: float
    total_docs_scored: int
    total_latency_ms: float
    avg_query_latency_ms: float


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_dir: str = "models") -> xgb.XGBRanker:
    """Load trained LambdaMART ranker from disk."""
    model_path = Path(model_dir) / "ranker.ubj"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py first."
        )
    ranker = xgb.XGBRanker()
    ranker.load_model(str(model_path))
    return ranker


# ---------------------------------------------------------------------------
# NDCG computation
# ---------------------------------------------------------------------------

def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Normalized DCG at position k."""
    rels = relevances[:k]
    if len(rels) == 0:
        return 0.0
    discounts = np.log2(np.arange(len(rels)) + 2)
    dcg = float(np.sum((2**rels - 1) / discounts))
    ideal_rels = np.sort(relevances)[::-1][:k]
    ideal_dcg = float(np.sum((2**ideal_rels - 1) / np.log2(np.arange(len(ideal_rels)) + 2)))
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


# ---------------------------------------------------------------------------
# Online ranking via feature store
# ---------------------------------------------------------------------------

def rank_query(
    model: xgb.XGBRanker,
    entity_ids: list[str],
    grpc_target: str = "localhost:9090",
    relevances: Optional[list[int]] = None,
) -> RankingResult:
    """
    Rank a set of candidate documents for a query.

    Fetches features from the feature store, scores with LambdaMART,
    returns documents sorted by predicted relevance.
    """
    from grpc_client import FeatureStoreClient

    t0 = time.time()

    client = FeatureStoreClient(grpc_target)
    result = client.get_online_features(VIEW_NAME, VIEW_VERSION, entity_ids)
    client.close()

    entity_to_values = {}
    for v in result["vectors"]:
        entity_to_values[v["entity_id"]] = v["values"]

    found_ids = [eid for eid in entity_ids if eid in entity_to_values]

    if not found_ids:
        return RankingResult(
            query_id=0,
            ranked_entity_ids=[],
            scores=[],
            relevances=[],
            ndcg_5=None,
            latency_ms=(time.time() - t0) * 1000,
            source="FEATURE_STORE",
        )

    X = np.array([entity_to_values[eid] for eid in found_ids])
    scores = model.predict(X)

    # Sort by score descending
    order = np.argsort(-scores)
    ranked_ids = [found_ids[i] for i in order]
    ranked_scores = [float(scores[i]) for i in order]

    # Compute NDCG if relevances provided
    rels_map = {}
    if relevances:
        for eid, rel in zip(entity_ids, relevances):
            rels_map[eid] = rel
    ranked_rels = [rels_map.get(eid, 0) for eid in ranked_ids]
    ndcg_5 = ndcg_at_k(np.array(ranked_rels, dtype=float), 5) if rels_map else None

    latency_ms = (time.time() - t0) * 1000

    return RankingResult(
        query_id=0,
        ranked_entity_ids=ranked_ids,
        scores=ranked_scores,
        relevances=ranked_rels,
        ndcg_5=ndcg_5,
        latency_ms=latency_ms,
        source="FEATURE_STORE",
    )


# ---------------------------------------------------------------------------
# Offline ranking from DataFrame
# ---------------------------------------------------------------------------

def rank_dataframe(
    model: xgb.XGBRanker,
    df: pd.DataFrame,
) -> BatchRankingResult:
    """
    Score and rank all queries in a DataFrame.
    Returns NDCG metrics and per-query ranking results.
    """
    t0 = time.time()

    results = []
    all_ndcg5 = []
    all_ndcg10 = []
    total_docs = 0

    for qid, group in df.groupby("qid"):
        X = group[FEATURE_NAMES].fillna(0).values
        scores = model.predict(X)

        order = np.argsort(-scores)
        sorted_group = group.iloc[order]

        relevances = sorted_group[LABEL_COL].values.astype(float)
        n5 = ndcg_at_k(relevances, 5)
        n10 = ndcg_at_k(relevances, 10)
        all_ndcg5.append(n5)
        all_ndcg10.append(n10)

        entity_ids = sorted_group["entity_id"].tolist() if "entity_id" in sorted_group else []

        results.append(RankingResult(
            query_id=int(qid),
            ranked_entity_ids=entity_ids,
            scores=[float(s) for s in scores[order]],
            relevances=[int(r) for r in relevances],
            ndcg_5=n5,
            latency_ms=0,
            source="DATAFRAME",
        ))
        total_docs += len(group)

    total_ms = (time.time() - t0) * 1000

    return BatchRankingResult(
        results=results,
        mean_ndcg_5=float(np.mean(all_ndcg5)),
        mean_ndcg_10=float(np.mean(all_ndcg10)),
        total_docs_scored=total_docs,
        total_latency_ms=total_ms,
        avg_query_latency_ms=total_ms / max(len(results), 1),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_ranking_result(result: RankingResult):
    """Pretty-print a single query ranking result."""
    print(f"  Query {result.query_id}:")
    print(f"    Docs ranked: {len(result.ranked_entity_ids)}")
    if result.ndcg_5 is not None:
        print(f"    NDCG@5:      {result.ndcg_5:.4f}")
    print(f"    Latency:     {result.latency_ms:.1f}ms")
    print(f"    Source:       {result.source}")
    if result.ranked_entity_ids:
        print(f"    Top 5:")
        for i, (eid, score, rel) in enumerate(
            zip(result.ranked_entity_ids[:5], result.scores[:5], result.relevances[:5])
        ):
            print(f"      {i+1}. {eid}  score={score:.4f}  rel={rel}")


def print_batch_ranking_result(result: BatchRankingResult):
    """Pretty-print batch ranking summary."""
    print(f"  Queries evaluated:  {len(result.results)}")
    print(f"  Total docs scored:  {result.total_docs_scored:,}")
    print(f"  Mean NDCG@5:        {result.mean_ndcg_5:.4f}")
    print(f"  Mean NDCG@10:       {result.mean_ndcg_10:.4f}")
    print(f"  Total latency:      {result.total_latency_ms:.1f}ms")
    print(f"  Avg query latency:  {result.avg_query_latency_ms:.1f}ms")

    # Distribution of per-query NDCG@5
    ndcg5s = [r.ndcg_5 for r in result.results if r.ndcg_5 is not None]
    if ndcg5s:
        arr = np.array(ndcg5s)
        print(f"\n  NDCG@5 distribution:")
        print(f"    Mean:   {arr.mean():.4f}")
        print(f"    Median: {np.median(arr):.4f}")
        print(f"    P25:    {np.percentile(arr, 25):.4f}")
        print(f"    P75:    {np.percentile(arr, 75):.4f}")
        print(f"    Min:    {arr.min():.4f}")
        print(f"    Max:    {arr.max():.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rank documents with trained LambdaMART")
    parser.add_argument("--model-dir", default="models", help="Model directory")
    parser.add_argument("--entity-ids", nargs="+", help="Entity IDs to rank")
    parser.add_argument("--grpc-target", default="localhost:9090", help="gRPC target")
    parser.add_argument("--offline", action="store_true",
                        help="Rank from generated data instead of feature store")
    parser.add_argument("--n-queries", type=int, default=50,
                        help="Number of queries for offline ranking")
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.model_dir)

    if args.offline:
        from dataset import load_dataset
        print(f"\nOffline ranking with {args.n_queries} queries...")
        _, test_df = load_dataset(synthetic=True, n_queries=args.n_queries, docs_per_query=50)
        batch_result = rank_dataframe(model, test_df)
        print_batch_ranking_result(batch_result)

        print("\n  Sample query rankings:")
        for r in batch_result.results[:3]:
            print_ranking_result(r)
            print()
    else:
        entity_ids = args.entity_ids or ["q1_d0", "q1_d1", "q1_d2", "q1_d3", "q1_d4"]
        print(f"\nRanking {len(entity_ids)} documents via feature store...")
        result = rank_query(model, entity_ids, args.grpc_target)
        print_ranking_result(result)
