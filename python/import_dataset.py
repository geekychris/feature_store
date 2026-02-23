#!/usr/bin/env python3
"""
Import datasets into the Feature Store via REST (register) and gRPC (materialize).

Used by the Vaadin admin UI to load datasets without writing code.
Also usable from the command line.

Datasets are discovered automatically from python/*_example/dataset_manifest.json files.

Usage:
  python import_dataset.py --list                              # List available datasets
  python import_dataset.py --dataset gbdt
  python import_dataset.py --dataset mslr --n-queries 100 --docs-per-query 50
  python import_dataset.py --dataset criteo --n-samples 50000
  python import_dataset.py --dataset gbdt --register-only
  python import_dataset.py --dataset mslr --materialize-only --max-vectors 5000
"""

import argparse
import glob
import json
import os
import sys
import time


def discover_datasets():
    """Scan python/*_example/dataset_manifest.json for available datasets."""
    manifests = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for path in sorted(glob.glob(os.path.join(script_dir, "*_example", "dataset_manifest.json"))):
        try:
            with open(path) as f:
                manifest = json.load(f)
            manifests[manifest["id"]] = manifest
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: could not parse {path}: {e}", file=sys.stderr)
    return manifests


def list_datasets():
    """Print discovered datasets."""
    manifests = discover_datasets()
    if not manifests:
        print("No datasets found. Ensure python/*_example/dataset_manifest.json files exist.")
        return
    print(f"\nAvailable datasets ({len(manifests)}):")
    print("-" * 60)
    for ds_id, m in manifests.items():
        print(f"  {ds_id:10s}  {m['name']}")
        print(f"             {m['description']}")
        params = ", ".join(p["name"] for p in m.get("params", []))
        if params:
            print(f"             params: {params}")
        print()


def import_gbdt(args):
    """Import the Merchant Fraud GBDT dataset."""
    sys.path.insert(0, "python/gbdt_example")
    from dataset import (
        generate_dataset,
        register_with_feature_store,
        materialize_to_feature_store,
        FEATURE_NAMES,
    )

    n_merchants = getattr(args, "n_merchants", 1000) or 1000
    print(f"[GBDT] Generating {n_merchants:,} merchant dataset...")
    df = generate_dataset(n_merchants)
    print(f"  {len(df):,} merchants, {len(FEATURE_NAMES)} features")
    print(f"  High-risk: {df['is_high_risk'].sum():,} ({df['is_high_risk'].mean()*100:.1f}%)")

    if not args.materialize_only:
        print(f"\n[GBDT] Registering with feature store at {args.rest_url}...")
        try:
            register_with_feature_store(args.rest_url)
            print("  Registration complete.")
        except Exception as e:
            print(f"  Registration failed: {e}")
            if args.register_only:
                return

    if not args.register_only:
        print(f"\n[GBDT] Materializing {len(df):,} vectors to {args.grpc_target}...")
        try:
            t0 = time.time()
            total = materialize_to_feature_store(df, args.grpc_target)
            elapsed = time.time() - t0
            print(f"  Materialized {total:,} vectors in {elapsed:.1f}s")
        except Exception as e:
            print(f"  Materialization failed: {e}")

    print("\n[GBDT] Import complete.")


def import_mslr(args):
    """Import the MSLR-WEB10K Learning to Rank dataset."""
    sys.path.insert(0, "python/mslr_example")
    from dataset import (
        load_dataset,
        register_with_feature_store,
        materialize_to_feature_store,
        FEATURE_NAMES,
        NUM_FEATURES,
    )

    n_queries = getattr(args, "n_queries", 200) or 200
    docs_per_query = getattr(args, "docs_per_query", 50) or 50
    print(f"[MSLR] Loading dataset ({n_queries} queries, {docs_per_query} docs/query)...")
    train_df, test_df = load_dataset(
        synthetic=True, n_queries=n_queries, docs_per_query=docs_per_query,
    )
    print(f"  Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")
    print(f"  Features: {NUM_FEATURES}")

    if not args.materialize_only:
        print(f"\n[MSLR] Registering {NUM_FEATURES} features at {args.rest_url}...")
        try:
            register_with_feature_store(args.rest_url)
            print("  Registration complete.")
        except Exception as e:
            print(f"  Registration failed: {e}")
            if args.register_only:
                return

    if not args.register_only:
        import pandas as pd
        combined = pd.concat([train_df, test_df], ignore_index=True)
        max_vecs = getattr(args, "max_vectors", None) or len(combined)
        print(f"\n[MSLR] Materializing up to {max_vecs:,} vectors to {args.grpc_target}...")
        try:
            t0 = time.time()
            total = materialize_to_feature_store(combined, args.grpc_target, max_vectors=max_vecs)
            elapsed = time.time() - t0
            print(f"  Materialized {total:,} vectors in {elapsed:.1f}s")
        except Exception as e:
            print(f"  Materialization failed: {e}")

    print("\n[MSLR] Import complete.")


def import_criteo(args):
    """Import the Criteo CTR dataset."""
    sys.path.insert(0, "python/criteo_example")
    from dataset import (
        load_dataset,
        register_with_feature_store,
        materialize_to_feature_store,
        FEATURE_NAMES,
        NUM_FEATURES,
        LABEL_COL,
    )

    n_samples = getattr(args, "n_samples", 100_000) or 100_000
    print(f"[Criteo] Generating {n_samples:,} impression dataset...")
    train_df, test_df = load_dataset(synthetic=True, n_samples=n_samples)
    print(f"  Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")
    print(f"  Features: {NUM_FEATURES}")
    print(f"  CTR: {train_df[LABEL_COL].mean()*100:.2f}%")

    if not args.materialize_only:
        print(f"\n[Criteo] Registering {NUM_FEATURES} features at {args.rest_url}...")
        try:
            register_with_feature_store(args.rest_url)
            print("  Registration complete.")
        except Exception as e:
            print(f"  Registration failed: {e}")
            if args.register_only:
                return

    if not args.register_only:
        import pandas as pd
        combined = pd.concat([train_df, test_df], ignore_index=True)
        max_vecs = getattr(args, "max_vectors", None) or len(combined)
        print(f"\n[Criteo] Materializing up to {max_vecs:,} vectors to {args.grpc_target}...")
        try:
            t0 = time.time()
            total = materialize_to_feature_store(combined, args.grpc_target, max_vectors=max_vecs)
            elapsed = time.time() - t0
            print(f"  Materialized {total:,} vectors in {elapsed:.1f}s")
        except Exception as e:
            print(f"  Materialization failed: {e}")

    print("\n[Criteo] Import complete.")


# Mapping from manifest import_id to handler function
IMPORT_HANDLERS = {
    "gbdt": import_gbdt,
    "mslr": import_mslr,
    "criteo": import_criteo,
}


def main():
    # Discover available datasets from manifests
    available = discover_datasets()
    known_ids = sorted(set(list(IMPORT_HANDLERS.keys()) + list(available.keys())))

    parser = argparse.ArgumentParser(
        description="Import datasets into the Feature Store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true",
                        help="List available datasets and exit")
    parser.add_argument("--dataset", choices=known_ids,
                        help="Dataset to import")
    parser.add_argument("--rest-url", default="http://localhost:8085",
                        help="Feature store REST URL")
    parser.add_argument("--grpc-target", default="localhost:9090",
                        help="Feature store gRPC target")
    parser.add_argument("--register-only", action="store_true",
                        help="Only register entities/features/views, skip materialization")
    parser.add_argument("--materialize-only", action="store_true",
                        help="Only materialize vectors, skip registration")
    parser.add_argument("--max-vectors", type=int, default=None,
                        help="Max vectors to materialize")

    # Dataset-specific params (kept for backward compat; new datasets use manifests)
    parser.add_argument("--n-merchants", type=int, default=1000,
                        help="Number of merchants (gbdt)")
    parser.add_argument("--n-queries", type=int, default=200,
                        help="Number of queries (mslr)")
    parser.add_argument("--docs-per-query", type=int, default=50,
                        help="Docs per query (mslr)")
    parser.add_argument("--n-samples", type=int, default=100_000,
                        help="Number of samples (criteo)")

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if not args.dataset:
        parser.error("--dataset is required (or use --list to see available datasets)")

    handler = IMPORT_HANDLERS.get(args.dataset)
    if handler:
        handler(args)
    else:
        print(f"Error: no import handler for dataset '{args.dataset}'.", file=sys.stderr)
        print(f"Known datasets: {', '.join(known_ids)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
