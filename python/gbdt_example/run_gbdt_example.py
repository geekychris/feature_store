#!/usr/bin/env python3
"""
End-to-end GBDT example for merchant fraud risk scoring.

This script demonstrates the full ML lifecycle with the feature store:

  1. Generate   — synthetic merchant fraud dataset (10K merchants, 15 features)
  2. Register   — entity, features, feature view via REST API
  3. Materialize — feature vectors to the store via gRPC
  4. Train      — XGBoost GBDT with hyperparameter search + cross-validation
  5. Validate   — check model metrics against feature store validation gates
  6. Infer      — score merchants by fetching features from the store

Running modes:
  - Full (with feature store):
      Requires the Java server running on localhost:8080 (REST) / :9090 (gRPC)
  - Standalone (no feature store):
      Trains and infers using in-memory DataFrames only

Usage:
  # Full pipeline (server must be running)
  python run_gbdt_example.py

  # Standalone — train and infer without feature store
  python run_gbdt_example.py --standalone

  # Skip training, just infer with existing model
  python run_gbdt_example.py --infer-only --entity-ids m_000001 m_000050
"""

import argparse
import sys
import time

from dataset import (
    generate_dataset,
    register_with_feature_store,
    materialize_to_feature_store,
    FEATURE_NAMES,
    LABEL_COL,
    VIEW_NAME,
    VIEW_VERSION,
)
from train import (
    train_from_dataframe,
    validate_with_feature_store,
    load_model as load_trained_model,
    print_training_report,
)
from inference import (
    load_model,
    score_batch,
    score_dataframe,
    print_batch_result,
)


def run_full_pipeline(args):
    """Full pipeline with feature store integration."""
    t_start = time.time()

    # ---------------------------------------------------------------
    # Step 1: Generate dataset
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 1: Generate Dataset")
    print("=" * 60)

    df = generate_dataset(args.n_merchants)
    pos_rate = df[LABEL_COL].mean() * 100
    print(f"  Merchants:     {len(df):,}")
    print(f"  Features:      {len(FEATURE_NAMES)}")
    print(f"  High-risk:     {df[LABEL_COL].sum():,} ({pos_rate:.1f}%)")

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
        print("  Continuing with training only...")
        reg_result = None

    # ---------------------------------------------------------------
    # Step 3: Materialize features
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Materialize Features")
    print("=" * 60)

    try:
        t0 = time.time()
        total = materialize_to_feature_store(df, args.grpc_target)
        elapsed = time.time() - t0
        print(f"  Materialized {total:,} vectors in {elapsed:.1f}s "
              f"({total/elapsed:,.0f} vectors/sec)")
    except Exception as e:
        print(f"  [SKIP] Materialization failed: {e}")

    # ---------------------------------------------------------------
    # Step 4: Train GBDT
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Train GBDT (XGBoost)")
    print("=" * 60)

    result = train_from_dataframe(
        df,
        search_hyperparams=not args.no_search,
        save_dir=args.model_dir,
    )

    # ---------------------------------------------------------------
    # Step 5: Validate with feature store
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Validate Model")
    print("=" * 60)

    try:
        validation = validate_with_feature_store(result, args.rest_url)
        print(f"  Server validation: {'PASSED' if validation['passed'] else 'FAILED'}")
        for gate in validation.get("gates", []):
            status = "PASS" if gate["passed"] else "FAIL"
            print(f"    [{status}] {gate['name']}: {gate['metric']:.4f} "
                  f"(threshold: {gate['threshold']})")
    except Exception as e:
        print(f"  [SKIP] Server validation failed: {e}")
        print(f"  Local validation: {'PASSED' if result.gates_passed else 'FAILED'}")

    # ---------------------------------------------------------------
    # Step 6: Inference
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 6: Real-time Inference")
    print("=" * 60)

    model = load_model(args.model_dir)

    # Try feature store inference first
    sample_ids = df["entity_id"].sample(min(10, len(df)), random_state=42).tolist()
    try:
        print(f"\n  Scoring {len(sample_ids)} merchants via feature store...")
        batch_result = score_batch(model, sample_ids, args.grpc_target)
        print_batch_result(batch_result)

        print("\n  Sample scores:")
        for r in batch_result.results[:5]:
            print(f"    {r.entity_id}: {r.risk_label} (p={r.risk_probability:.4f})")
    except Exception as e:
        print(f"  [SKIP] Feature store inference failed: {e}")
        print(f"\n  Falling back to offline inference...")
        scored = score_dataframe(model, df.sample(min(100, len(df)), random_state=42))
        high = (scored["risk_label"] == "HIGH").sum()
        med = (scored["risk_label"] == "MEDIUM").sum()
        low = (scored["risk_label"] == "LOW").sum()
        print(f"  Results: HIGH={high}, MEDIUM={med}, LOW={low}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    elapsed_total = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"Pipeline Complete ({elapsed_total:.1f}s)")
    print("=" * 60)
    print(f"  Model:       {result.model_name}")
    print(f"  AUC-ROC:     {result.auc_roc:.4f}")
    print(f"  AUC-PR:      {result.auc_pr:.4f}")
    print(f"  Gates:       {'PASSED' if result.gates_passed else 'FAILED'}")
    print(f"  Model saved: {result.model_path}")


def run_standalone(args):
    """Train and infer without feature store."""
    print("\n" + "=" * 60)
    print("Standalone Mode (no feature store)")
    print("=" * 60)

    # Generate
    print("\n[1/3] Generating dataset...")
    df = generate_dataset(args.n_merchants)
    print(f"  {len(df):,} merchants, {df[LABEL_COL].mean()*100:.1f}% high-risk")

    # Train
    print("\n[2/3] Training GBDT...")
    result = train_from_dataframe(
        df,
        search_hyperparams=not args.no_search,
        save_dir=args.model_dir,
    )

    # Infer
    print("\n[3/3] Scoring merchants...")
    model = load_model(args.model_dir)
    scored = score_dataframe(model, df)

    high = (scored["risk_label"] == "HIGH").sum()
    med = (scored["risk_label"] == "MEDIUM").sum()
    low = (scored["risk_label"] == "LOW").sum()
    print(f"\n  Risk Distribution:")
    print(f"    HIGH:   {high:,} ({high/len(scored)*100:.1f}%)")
    print(f"    MEDIUM: {med:,} ({med/len(scored)*100:.1f}%)")
    print(f"    LOW:    {low:,} ({low/len(scored)*100:.1f}%)")

    print(f"\n  Top 10 Riskiest Merchants:")
    top10 = scored.nlargest(10, "risk_probability")
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"    {i:2d}. {row['entity_id']}: p={row['risk_probability']:.4f}"
              f"  chargeback={row['chargeback_rate_90d']:.4f}"
              f"  fraud_reports={row['fraud_reports_30d']:.0f}"
              f"  age={row['account_age_days']:.0f}d")

    print(f"\n  Model: {result.model_path}")
    print(f"  AUC-ROC: {result.auc_roc:.4f}  AUC-PR: {result.auc_pr:.4f}")


def run_infer_only(args):
    """Infer with an existing model."""
    model = load_model(args.model_dir)
    entity_ids = args.entity_ids or ["m_000001", "m_000010", "m_000100"]

    if args.standalone:
        from dataset import generate_dataset
        df = generate_dataset(args.n_merchants)
        scored = score_dataframe(model, df)
        for eid in entity_ids:
            row = scored[scored["entity_id"] == eid]
            if not row.empty:
                r = row.iloc[0]
                print(f"  {eid}: {r['risk_label']} (p={r['risk_probability']:.4f})")
            else:
                print(f"  {eid}: not found in dataset")
    else:
        batch_result = score_batch(model, entity_ids, args.grpc_target)
        print_batch_result(batch_result)
        for r in batch_result.results:
            print(f"  {r.entity_id}: {r.risk_label} (p={r.risk_probability:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GBDT Merchant Fraud Risk — End-to-End Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--standalone", action="store_true",
                        help="Run without feature store (train + infer from DataFrames)")
    parser.add_argument("--infer-only", action="store_true",
                        help="Skip training, infer with existing model")
    parser.add_argument("--n-merchants", type=int, default=10_000,
                        help="Number of merchants to generate (default: 10000)")
    parser.add_argument("--model-dir", default="models",
                        help="Directory for model artifacts (default: models)")
    parser.add_argument("--rest-url", default="http://localhost:8080",
                        help="Feature store REST URL")
    parser.add_argument("--grpc-target", default="localhost:9090",
                        help="Feature store gRPC target")
    parser.add_argument("--no-search", action="store_true",
                        help="Skip hyperparameter search, use defaults")
    parser.add_argument("--entity-ids", nargs="+",
                        help="Merchant IDs for inference")
    args = parser.parse_args()

    if args.infer_only:
        run_infer_only(args)
    elif args.standalone:
        run_standalone(args)
    else:
        run_full_pipeline(args)
