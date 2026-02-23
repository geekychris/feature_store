#!/usr/bin/env python3
"""
End-to-end CTR prediction example using Criteo Display Advertising data.

This script demonstrates the full ML lifecycle with the feature store:

  1. Load      — Criteo CTR dataset (real or synthetic, 39 features)
  2. Register  — entity, features, feature view via REST API
  3. Materialize — feature vectors to the store via gRPC
  4. Train     — XGBoost binary classifier with hyperparameter search
  5. Validate  — check CTR metrics against validation gates
  6. Infer     — score impressions by fetching features from the store

Running modes:
  - Full (with feature store):
      Requires the Java server running on localhost:8085 (REST) / :9090 (gRPC)
  - Standalone (no feature store):
      Trains and evaluates using in-memory DataFrames only

Usage:
  # Full pipeline (server must be running)
  python run_criteo_example.py

  # Standalone — train and evaluate without feature store
  python run_criteo_example.py --standalone

  # Use real Criteo data (requires Kaggle download)
  python run_criteo_example.py --no-synthetic --max-rows 1000000
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
    NUM_NUMERIC,
    NUM_CATEGORICAL,
)
from train import (
    train_from_dataframes,
    print_training_report,
)
from inference import (
    load_model,
    score_batch,
    score_dataframe,
    print_scoring_result,
    print_batch_result,
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
        max_rows=args.max_rows,
        n_samples=args.n_samples,
    )
    ctr_train = train_df[LABEL_COL].mean() * 100
    ctr_test = test_df[LABEL_COL].mean() * 100
    print(f"  Train: {len(train_df):,} rows (CTR: {ctr_train:.2f}%)")
    print(f"  Test:  {len(test_df):,} rows (CTR: {ctr_test:.2f}%)")
    print(f"  Features: {NUM_FEATURES} ({NUM_NUMERIC} numeric + {NUM_CATEGORICAL} categorical)")

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
        max_vecs = args.max_materialize or 10000
        total = materialize_to_feature_store(train_df, args.grpc_target, max_vectors=max_vecs)
        total += materialize_to_feature_store(test_df, args.grpc_target, max_vectors=max_vecs)
        elapsed = time.time() - t0
        print(f"  Materialized {total:,} vectors in {elapsed:.1f}s "
              f"({total/max(elapsed,0.01):,.0f} vectors/sec)")
    except Exception as e:
        print(f"  [SKIP] Materialization failed: {e}")

    # ---------------------------------------------------------------
    # Step 4: Train CTR model
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Train CTR Model (XGBoost)")
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
        direction = "<=" if gate["gate"] == "LogLoss" else ">="
        print(f"    [{status}] {gate['gate']}: {gate['value']:.4f} "
              f"({direction} {gate['threshold']})")

    # ---------------------------------------------------------------
    # Step 6: Inference
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 6: CTR Inference")
    print("=" * 60)

    model = load_model(args.model_dir)

    # Try feature store inference
    sample_ids = test_df["entity_id"].sample(min(20, len(test_df)), random_state=42).tolist()
    try:
        print(f"\n  Scoring {len(sample_ids)} impressions via feature store...")
        batch_result = score_batch(model, sample_ids, args.grpc_target)
        print_batch_result(batch_result)

        print("\n  Sample scores:")
        for r in batch_result.results[:5]:
            print(f"    {r.entity_id}: {r.bid_recommendation} "
                  f"(CTR={r.click_probability*100:.2f}%)")
    except Exception as e:
        print(f"  [SKIP] Feature store inference failed: {e}")

    # Offline batch scoring
    print(f"\n  Offline batch evaluation on test set...")
    scored = score_dataframe(model, test_df)
    mean_ctr = scored["click_probability"].mean() * 100
    print(f"  Mean predicted CTR: {mean_ctr:.2f}%")
    for bid in ["HIGH_BID", "MEDIUM_BID", "LOW_BID", "NO_BID"]:
        count = (scored["bid_recommendation"] == bid).sum()
        pct = count / len(scored) * 100
        print(f"    {bid}: {count:,} ({pct:.1f}%)")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    elapsed_total = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"Pipeline Complete ({elapsed_total:.1f}s)")
    print("=" * 60)
    print(f"  Model:       {result.model_name}")
    print(f"  AUC-ROC:     {result.auc_roc:.4f}")
    print(f"  LogLoss:     {result.logloss:.4f}")
    print(f"  AUC-PR:      {result.auc_pr:.4f}")
    print(f"  Gates:       {'PASSED' if result.gates_passed else 'FAILED'}")
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
        max_rows=args.max_rows,
        n_samples=args.n_samples,
    )

    # Train
    print("\n[2/3] Training CTR model...")
    result = train_from_dataframes(
        train_df,
        test_df,
        search_hyperparams=not args.no_search,
        save_dir=args.model_dir,
    )

    # Evaluate
    print("\n[3/3] Scoring test set...")
    model = load_model(args.model_dir)
    scored = score_dataframe(model, test_df)

    print(f"\n  Test set scoring results:")
    print(f"  Mean predicted CTR: {scored['click_probability'].mean()*100:.2f}%")
    print(f"  Actual CTR:         {test_df[LABEL_COL].mean()*100:.2f}%")

    for bid in ["HIGH_BID", "MEDIUM_BID", "LOW_BID", "NO_BID"]:
        count = (scored["bid_recommendation"] == bid).sum()
        pct = count / len(scored) * 100
        print(f"    {bid}: {count:,} ({pct:.1f}%)")

    # Click rate by bid tier
    print(f"\n  Actual CTR by bid tier:")
    for bid in ["HIGH_BID", "MEDIUM_BID", "LOW_BID", "NO_BID"]:
        mask = scored["bid_recommendation"] == bid
        if mask.sum() > 0:
            actual_ctr = scored.loc[mask, LABEL_COL].mean() * 100
            print(f"    {bid}: {actual_ctr:.2f}% actual CTR ({mask.sum():,} impressions)")


def run_infer_only(args):
    """Run inference on specific entities."""
    model = load_model(args.model_dir)

    if args.entity_ids:
        print(f"\nScoring {len(args.entity_ids)} impressions...")
        batch_result = score_batch(model, args.entity_ids, args.grpc_target)
        print_batch_result(batch_result)
        for r in batch_result.results:
            print_scoring_result(r)
    else:
        print("\nNo entity IDs provided, running offline evaluation...")
        _, test_df = load_dataset(synthetic=args.synthetic, n_samples=args.n_samples)
        scored = score_dataframe(model, test_df)
        print(f"  Mean CTR: {scored['click_probability'].mean()*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Criteo CTR Prediction Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_criteo_example.py --standalone                  # Quick standalone test
  python run_criteo_example.py                               # Full pipeline with feature store
  python run_criteo_example.py --no-synthetic --max-rows 1000000  # Real data (partial)
  python run_criteo_example.py --infer-only --entity-ids imp_00000001 imp_00000002
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
                        help="Use real Criteo data (requires Kaggle download)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Max rows to load from real data")
    parser.add_argument("--n-samples", type=int, default=100_000,
                        help="Number of synthetic samples")

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
                        help="Max vectors to materialize")

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
