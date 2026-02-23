#!/usr/bin/env python3
"""
End-to-end Learning to Rank example using MSLR-WEB10K.

This script demonstrates the full ML lifecycle with the feature store:

  1. Load      — MSLR-WEB10K dataset (real or synthetic, 136 features)
  2. Register  — entity, features, feature view via REST API
  3. Materialize — feature vectors to the store via gRPC
  4. Train     — XGBoost LambdaMART ranker with hyperparameter search
  5. Validate  — check ranking metrics against validation gates
  6. Infer     — re-rank documents by fetching features from the store

Running modes:
  - Full (with feature store):
      Requires the Java server running on localhost:8085 (REST) / :9090 (gRPC)
  - Standalone (no feature store):
      Trains and evaluates using in-memory DataFrames only

Usage:
  # Full pipeline (server must be running)
  python run_mslr_example.py

  # Standalone — train and evaluate without feature store
  python run_mslr_example.py --standalone

  # Use real MSLR-WEB10K data (requires download)
  python run_mslr_example.py --no-synthetic --max-rows 50000
"""

import argparse
import sys
import time

from dataset import (
    load_dataset,
    register_with_feature_store,
    materialize_to_feature_store,
    FEATURE_NAMES,
    LABEL_COL,
    VIEW_NAME,
    VIEW_VERSION,
    NUM_FEATURES,
)
from train import (
    train_from_dataframes,
    print_training_report,
)
from inference import (
    load_model,
    rank_query,
    rank_dataframe,
    print_ranking_result,
    print_batch_ranking_result,
)


def run_full_pipeline(args):
    """Full pipeline with feature store integration."""
    t_start = time.time()

    # ---------------------------------------------------------------
    # Step 1: Load dataset
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 1: Load Dataset")
    print("=" * 60)

    train_df, test_df = load_dataset(
        synthetic=args.synthetic,
        fold=args.fold,
        max_rows=args.max_rows,
        n_queries=args.n_queries,
        docs_per_query=args.docs_per_query,
    )
    print(f"  Train: {len(train_df):,} rows, {train_df['qid'].nunique()} queries")
    print(f"  Test:  {len(test_df):,} rows, {test_df['qid'].nunique()} queries")
    print(f"  Features: {NUM_FEATURES}")
    print(f"  Relevance levels: {sorted(train_df[LABEL_COL].unique())}")

    # ---------------------------------------------------------------
    # Step 2: Register with feature store
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Register Entity / Features / View")
    print("=" * 60)

    try:
        reg_result = register_with_feature_store(args.rest_url)
    except Exception as e:
        print(f"  [SKIP] Registration failed: {e}")
        reg_result = None

    # ---------------------------------------------------------------
    # Step 3: Materialize features
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Materialize Features")
    print("=" * 60)

    try:
        t0 = time.time()
        # Materialize both train and test sets (limit for large datasets)
        max_vecs = args.max_materialize or 10000
        total = materialize_to_feature_store(train_df, args.grpc_target, max_vectors=max_vecs)
        total += materialize_to_feature_store(test_df, args.grpc_target, max_vectors=max_vecs)
        elapsed = time.time() - t0
        print(f"  Materialized {total:,} vectors in {elapsed:.1f}s "
              f"({total/max(elapsed,0.01):,.0f} vectors/sec)")
    except Exception as e:
        print(f"  [SKIP] Materialization failed: {e}")

    # ---------------------------------------------------------------
    # Step 4: Train LambdaMART
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Train LambdaMART Ranker (XGBoost)")
    print("=" * 60)

    result = train_from_dataframes(
        train_df,
        test_df,
        search_hyperparams=not args.no_search,
        save_dir=args.model_dir,
    )

    # ---------------------------------------------------------------
    # Step 5: Validate
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Validate Model")
    print("=" * 60)

    print(f"  Gates: {'PASSED' if result.gates_passed else 'FAILED'}")
    for gate in result.gate_details:
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"    [{status}] {gate['gate']}: {gate['value']:.4f} "
              f"(threshold: {gate['threshold']})")

    # ---------------------------------------------------------------
    # Step 6: Inference
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 6: Ranking Inference")
    print("=" * 60)

    model = load_model(args.model_dir)

    # Try feature store inference
    sample_qid = test_df["qid"].unique()[0]
    sample_docs = test_df[test_df["qid"] == sample_qid]
    sample_ids = sample_docs["entity_id"].tolist()[:20]
    sample_rels = sample_docs[LABEL_COL].tolist()[:20]

    try:
        print(f"\n  Re-ranking {len(sample_ids)} docs for query {sample_qid} via feature store...")
        ranking = rank_query(model, sample_ids, args.grpc_target, relevances=sample_rels)
        ranking.query_id = sample_qid
        print_ranking_result(ranking)
    except Exception as e:
        print(f"  [SKIP] Feature store ranking failed: {e}")

    # Offline batch evaluation
    print(f"\n  Offline batch evaluation on test set...")
    batch_result = rank_dataframe(model, test_df)
    print_batch_ranking_result(batch_result)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    elapsed_total = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"Pipeline Complete ({elapsed_total:.1f}s)")
    print("=" * 60)
    print(f"  Model:     {result.model_name}")
    print(f"  NDCG@5:    {result.ndcg_5:.4f}")
    print(f"  NDCG@10:   {result.ndcg_10:.4f}")
    print(f"  MAP:       {result.map_score:.4f}")
    print(f"  Gates:     {'PASSED' if result.gates_passed else 'FAILED'}")
    print(f"  Model saved: {result.model_path}")


def run_standalone(args):
    """Train and evaluate without feature store."""
    print("\n" + "=" * 60)
    print("Standalone Mode (no feature store)")
    print("=" * 60)

    # Load
    print("\n[1/3] Loading dataset...")
    train_df, test_df = load_dataset(
        synthetic=args.synthetic,
        fold=args.fold,
        max_rows=args.max_rows,
        n_queries=args.n_queries,
        docs_per_query=args.docs_per_query,
    )

    # Train
    print("\n[2/3] Training LambdaMART ranker...")
    result = train_from_dataframes(
        train_df,
        test_df,
        search_hyperparams=not args.no_search,
        save_dir=args.model_dir,
    )

    # Evaluate
    print("\n[3/3] Evaluating on test set...")
    model = load_model(args.model_dir)
    batch_result = rank_dataframe(model, test_df)
    print_batch_ranking_result(batch_result)

    print(f"\n  Best query (NDCG@5):")
    best = max(batch_result.results, key=lambda r: r.ndcg_5 or 0)
    print_ranking_result(best)

    print(f"\n  Worst query (NDCG@5):")
    worst = min(batch_result.results, key=lambda r: r.ndcg_5 if r.ndcg_5 is not None else 1)
    print_ranking_result(worst)


def run_infer_only(args):
    """Run inference on specific entities."""
    model = load_model(args.model_dir)

    if args.entity_ids:
        print(f"\nRe-ranking {len(args.entity_ids)} documents...")
        result = rank_query(model, args.entity_ids, args.grpc_target)
        print_ranking_result(result)
    else:
        print("\nNo entity IDs provided, running offline evaluation...")
        _, test_df = load_dataset(
            synthetic=args.synthetic,
            n_queries=args.n_queries,
            docs_per_query=args.docs_per_query,
        )
        batch_result = rank_dataframe(model, test_df)
        print_batch_ranking_result(batch_result)


def main():
    parser = argparse.ArgumentParser(
        description="MSLR-WEB10K Learning to Rank Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mslr_example.py --standalone                  # Quick standalone test
  python run_mslr_example.py                               # Full pipeline with feature store
  python run_mslr_example.py --no-synthetic --max-rows 50000  # Real data (partial)
  python run_mslr_example.py --infer-only --entity-ids q1_d0 q1_d1 q1_d2
        """,
    )

    parser.add_argument("--standalone", action="store_true",
                        help="Run without feature store (train + evaluate only)")
    parser.add_argument("--infer-only", action="store_true",
                        help="Skip training, just run inference")

    # Data options
    parser.add_argument("--synthetic", action="store_true", default=True,
                        help="Use synthetic data (default)")
    parser.add_argument("--no-synthetic", dest="synthetic", action="store_false",
                        help="Use real MSLR-WEB10K data (requires download)")
    parser.add_argument("--fold", type=int, default=1, help="MSLR fold number (1-5)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Max rows to load from real data")
    parser.add_argument("--n-queries", type=int, default=200,
                        help="Number of synthetic queries")
    parser.add_argument("--docs-per-query", type=int, default=50,
                        help="Docs per query for synthetic data")

    # Training options
    parser.add_argument("--no-search", action="store_true",
                        help="Skip hyperparameter search (use defaults)")
    parser.add_argument("--model-dir", default="models",
                        help="Directory to save/load models")

    # Feature store options
    parser.add_argument("--rest-url", default="http://localhost:8085",
                        help="Feature store REST URL")
    parser.add_argument("--grpc-target", default="localhost:9090",
                        help="Feature store gRPC target")
    parser.add_argument("--max-materialize", type=int, default=None,
                        help="Max vectors to materialize (for large datasets)")

    # Inference options
    parser.add_argument("--entity-ids", nargs="+",
                        help="Entity IDs for inference")

    args = parser.parse_args()

    if args.infer_only:
        run_infer_only(args)
    elif args.standalone:
        run_standalone(args)
    else:
        run_full_pipeline(args)


if __name__ == "__main__":
    main()
