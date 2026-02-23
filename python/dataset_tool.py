#!/usr/bin/env python3
"""
Dataset tool for the Feature Store UI.

Provides three subcommands:
  preview  — Generate dataset and output first N rows as JSON
  export   — Generate dataset and write to Parquet or Iceberg table
  infer    — Load trained model and run inference on sample data

All subcommands discover datasets from python/*_example/dataset_manifest.json.

Output conventions:
  - Normal log lines go to stdout
  - Structured JSON results are on a single line prefixed with a magic marker:
      __PREVIEW_JSON__:{...}
      __INFER_JSON__:{...}
    The Java UI captures these lines and parses them.
"""

import argparse
import glob
import json
import os
import sys
import time


# ---------------------------------------------------------------------------
# Dataset discovery (same as import_dataset.py)
# ---------------------------------------------------------------------------

def discover_datasets():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    manifests = {}
    for path in sorted(glob.glob(os.path.join(script_dir, "*_example", "dataset_manifest.json"))):
        try:
            with open(path) as f:
                manifest = json.load(f)
            manifest["_dir"] = os.path.dirname(path)
            manifests[manifest["id"]] = manifest
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: could not parse {path}: {e}", file=sys.stderr)
    return manifests


def get_dataset_module(manifest):
    """Import and return the dataset module for a manifest."""
    ds_dir = manifest["_dir"]
    sys.path.insert(0, ds_dir)
    import importlib
    # All examples have a dataset.py
    spec = importlib.util.spec_from_file_location("dataset", os.path.join(ds_dir, "dataset.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_inference_module(manifest):
    """Import and return the inference module for a manifest."""
    ds_dir = manifest["_dir"]
    sys.path.insert(0, ds_dir)
    # Also need parent for grpc_client
    sys.path.insert(0, os.path.dirname(ds_dir))
    import importlib
    spec = importlib.util.spec_from_file_location("inference", os.path.join(ds_dir, "inference.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_models_dir(manifest):
    """Resolve models_dir to an absolute path, trying multiple strategies."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw = manifest.get("models_dir")
    if raw:
        # Already absolute?
        if os.path.isabs(raw) and os.path.isdir(raw):
            return raw
        # Relative to CWD?
        if os.path.isdir(raw):
            return os.path.abspath(raw)
        # Relative to project root (parent of python/ dir)
        candidate = os.path.join(project_root, raw)
        if os.path.isdir(candidate):
            return candidate
    # Fallback: models/ under project root (training saves here by default)
    project_models = os.path.join(project_root, "models")
    if os.path.isdir(project_models):
        return project_models
    # Final fallback: models/ under manifest dir
    return os.path.join(manifest["_dir"], "models")


def generate_dataframe(manifest, args):
    """Generate / load the dataset DataFrame based on dataset id."""
    mod = get_dataset_module(manifest)
    ds_id = manifest["id"]

    if ds_id == "gbdt":
        n = getattr(args, "n_merchants", None) or 1000
        df = mod.generate_dataset(n)
        return df
    elif ds_id == "mslr":
        nq = getattr(args, "n_queries", None) or 200
        dpq = getattr(args, "docs_per_query", None) or 40
        train_df, test_df = mod.load_dataset(synthetic=True, n_queries=nq, docs_per_query=dpq)
        import pandas as pd
        return pd.concat([train_df, test_df], ignore_index=True)
    elif ds_id == "criteo":
        ns = getattr(args, "n_samples", None) or 100_000
        train_df, test_df = mod.load_dataset(synthetic=True, n_samples=ns)
        import pandas as pd
        return pd.concat([train_df, test_df], ignore_index=True)
    else:
        raise ValueError(f"Unknown dataset: {ds_id}")


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def cmd_preview(args, manifest):
    """Generate dataset and output first N rows as JSON."""
    print(f"[Preview] Generating {manifest['name']}...")
    df = generate_dataframe(manifest, args)

    n_rows = min(args.rows, len(df))
    preview_df = df.head(n_rows)

    print(f"[Preview] Dataset: {len(df):,} rows x {len(df.columns)} columns")
    print(f"[Preview] Showing first {n_rows} rows")

    # Build JSON payload
    columns = list(preview_df.columns)
    data = []
    for _, row in preview_df.iterrows():
        data.append([_safe_value(v) for v in row.values])

    payload = {"columns": columns, "data": data, "total_rows": len(df)}
    print(f"__PREVIEW_JSON__:{json.dumps(payload)}")


def _safe_value(v):
    """Convert numpy/pandas types to JSON-safe Python types."""
    import numpy as np
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 6) if not np.isnan(v) else None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float):
        return round(v, 6) if v == v else None  # NaN check
    return v


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def cmd_export(args, manifest):
    """Generate dataset and export to Parquet or Iceberg."""
    print(f"[Export] Generating {manifest['name']}...")
    df = generate_dataframe(manifest, args)
    print(f"[Export] Dataset: {len(df):,} rows x {len(df.columns)} columns")

    output_dir = args.output_dir or os.path.join(manifest["_dir"], "exports")
    os.makedirs(output_dir, exist_ok=True)

    fmt = args.format
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ds_id = manifest["id"]

    if fmt == "parquet":
        output_path = os.path.join(output_dir, f"{ds_id}_{timestamp}.parquet")
        print(f"[Export] Writing Parquet: {output_path}")
        df.to_parquet(output_path, index=False, engine="pyarrow")
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[Export] Done: {output_path} ({size_mb:.1f} MB, {len(df):,} rows)")

    elif fmt == "iceberg":
        # Write as Parquet first, then register with Iceberg catalog if available
        parquet_path = os.path.join(output_dir, f"{ds_id}_{timestamp}.parquet")
        print(f"[Export] Writing Parquet (Iceberg-compatible): {parquet_path}")
        df.to_parquet(parquet_path, index=False, engine="pyarrow")

        # Write Iceberg metadata sidecar
        metadata = {
            "format": "iceberg",
            "table_name": f"{ds_id}_dataset",
            "schema": {col: str(df[col].dtype) for col in df.columns},
            "num_rows": len(df),
            "parquet_file": os.path.basename(parquet_path),
            "created_at": timestamp,
        }
        meta_path = os.path.join(output_dir, f"{ds_id}_{timestamp}_iceberg_meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
        print(f"[Export] Done: {parquet_path} ({size_mb:.1f} MB)")
        print(f"[Export] Iceberg metadata: {meta_path}")

    else:
        print(f"[Export] ERROR: Unknown format '{fmt}'", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _df_to_table(df, max_rows=None):
    """Convert a DataFrame to {columns, data} table format for JSON output."""
    if max_rows:
        df = df.head(max_rows)
    columns = list(df.columns)
    data = []
    for _, row in df.iterrows():
        data.append([_safe_value(v) for v in row.values])
    return {"columns": columns, "data": data}


def _load_input_file(path):
    """Load sample data from a JSON file (same format as __SAMPLE_JSON__)."""
    import pandas as pd
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data["data"], columns=data["columns"])
    # Convert numeric columns (values arrive as strings from the UI)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    return df


def cmd_infer(args, manifest):
    """Load model, generate sample data, run inference, output JSON."""
    ds_id = manifest["id"]
    models_dir = resolve_models_dir(manifest)
    infer_type = manifest.get("infer_type", "classification")
    input_file = getattr(args, "input_file", None)

    print(f"[Infer] Dataset: {manifest['name']}")
    print(f"[Infer] Type: {infer_type}")
    print(f"[Infer] Models dir: {models_dir}")
    if input_file:
        print(f"[Infer] Input file: {input_file}")

    # Check model exists
    import pathlib
    model_files = list(pathlib.Path(models_dir).glob("*.ubj"))
    if not model_files:
        model_files = list(pathlib.Path(models_dir).glob("*.joblib"))
    if not model_files:
        print(f"[Infer] ERROR: No model found in {models_dir}. Train first.")
        print(f"[Infer] Searched: {os.path.abspath(models_dir)}")
        return

    print(f"[Infer] Found model: {model_files[0].name}")

    n_samples = args.n_samples or 50
    inf_mod = get_inference_module(manifest)
    ds_mod = get_dataset_module(manifest)

    if infer_type == "classification":
        model = inf_mod.load_model(models_dir)

        if input_file:
            sample = _load_input_file(input_file)
            print(f"[Infer] Loaded {len(sample)} samples from file")
        else:
            df = ds_mod.generate_dataset(max(n_samples, 100))
            sample = df.sample(min(n_samples, len(df)), random_state=42).reset_index(drop=True)
            print(f"[Infer] Generated {len(sample)} samples")
            print(f"__SAMPLE_JSON__:{json.dumps(_df_to_table(sample))}")

        scored = inf_mod.score_dataframe(model, sample)

        summary = {
            "total": len(scored),
            "high": int((scored["risk_label"] == "HIGH").sum()),
            "medium": int((scored["risk_label"] == "MEDIUM").sum()),
            "low": int((scored["risk_label"] == "LOW").sum()),
        }
        print(f"[Infer] Scored {len(scored)} entities")
        print(f"[Infer] HIGH={summary['high']} MEDIUM={summary['medium']} LOW={summary['low']}")
        print(f"__INFER_JSON__:{json.dumps({'type': infer_type, 'scored': _df_to_table(scored), 'summary': summary})}")

    elif infer_type == "ranking":
        model = inf_mod.load_model(models_dir)

        if input_file:
            test_df = _load_input_file(input_file)
            print(f"[Infer] Loaded {len(test_df)} docs from file")
        else:
            nq = min(getattr(args, "n_queries", 10) or 10, 50)
            train_df, test_df = ds_mod.load_dataset(synthetic=True, n_queries=nq, docs_per_query=20)
            preview_rows = min(n_samples, len(test_df))
            print(f"[Infer] Loaded {len(test_df)} test docs across {test_df['qid'].nunique()} queries")
            print(f"__SAMPLE_JSON__:{json.dumps(_df_to_table(test_df, max_rows=preview_rows))}")

        batch = inf_mod.rank_dataframe(model, test_df)

        results = []
        for rr in batch.results[:20]:
            results.append({
                "query_id": rr.query_id,
                "ndcg_5": round(rr.ndcg_5, 4) if rr.ndcg_5 is not None else None,
                "num_docs": len(rr.ranked_entity_ids),
                "top_doc": rr.ranked_entity_ids[0] if rr.ranked_entity_ids else None,
                "top_score": round(rr.scores[0], 4) if rr.scores else None,
            })

        summary = {
            "total_queries": len(batch.results),
            "mean_ndcg_5": round(batch.mean_ndcg_5, 4),
            "mean_ndcg_10": round(batch.mean_ndcg_10, 4),
            "total_docs": batch.total_docs_scored,
        }
        print(f"[Infer] Ranked {summary['total_queries']} queries, {summary['total_docs']} docs")
        print(f"[Infer] Mean NDCG@5={summary['mean_ndcg_5']:.4f} NDCG@10={summary['mean_ndcg_10']:.4f}")
        print(f"__INFER_JSON__:{json.dumps({'type': infer_type, 'results': results, 'summary': summary})}")

    elif infer_type == "ctr":
        model = inf_mod.load_model(models_dir)

        if input_file:
            sample = _load_input_file(input_file)
            print(f"[Infer] Loaded {len(sample)} samples from file")
        else:
            ns = max(n_samples * 10, 1000)
            _, test_df = ds_mod.load_dataset(synthetic=True, n_samples=min(ns, 100_000))
            sample = test_df.sample(min(n_samples, len(test_df)), random_state=42).reset_index(drop=True)
            print(f"[Infer] Generated {len(sample)} samples")
            print(f"__SAMPLE_JSON__:{json.dumps(_df_to_table(sample))}")

        scored = inf_mod.score_dataframe(model, sample)

        summary = {
            "total": len(scored),
            "mean_ctr": round(float(scored["click_probability"].mean()), 6),
            "high_bid": int((scored["bid_recommendation"] == "HIGH_BID").sum()),
            "medium_bid": int((scored["bid_recommendation"] == "MEDIUM_BID").sum()),
            "low_bid": int((scored["bid_recommendation"] == "LOW_BID").sum()),
            "no_bid": int((scored["bid_recommendation"] == "NO_BID").sum()),
        }
        print(f"[Infer] Scored {len(scored)} impressions, mean CTR={summary['mean_ctr']*100:.2f}%")
        print(f"__INFER_JSON__:{json.dumps({'type': infer_type, 'scored': _df_to_table(scored), 'summary': summary})}")

    else:
        print(f"[Infer] ERROR: Unknown infer_type '{infer_type}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    manifests = discover_datasets()
    known_ids = sorted(manifests.keys())

    parser = argparse.ArgumentParser(
        description="Dataset tool for the Feature Store UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- preview --
    p_preview = sub.add_parser("preview", help="Preview dataset as a table")
    p_preview.add_argument("--dataset", required=True, choices=known_ids)
    p_preview.add_argument("--rows", type=int, default=50, help="Number of rows to show")
    p_preview.add_argument("--n-merchants", type=int, default=1000)
    p_preview.add_argument("--n-queries", type=int, default=200)
    p_preview.add_argument("--docs-per-query", type=int, default=40)
    p_preview.add_argument("--n-samples", type=int, default=100_000)

    # -- export --
    p_export = sub.add_parser("export", help="Export dataset to Parquet or Iceberg")
    p_export.add_argument("--dataset", required=True, choices=known_ids)
    p_export.add_argument("--format", required=True, choices=["parquet", "iceberg"])
    p_export.add_argument("--output-dir", default=None)
    p_export.add_argument("--n-merchants", type=int, default=1000)
    p_export.add_argument("--n-queries", type=int, default=200)
    p_export.add_argument("--docs-per-query", type=int, default=40)
    p_export.add_argument("--n-samples", type=int, default=100_000)

    # -- infer --
    p_infer = sub.add_parser("infer", help="Run inference with trained model")
    p_infer.add_argument("--dataset", required=True, choices=known_ids)
    p_infer.add_argument("--n-samples", type=int, default=50, help="Sample size for inference")
    p_infer.add_argument("--n-queries", type=int, default=10)
    p_infer.add_argument("--docs-per-query", type=int, default=20)
    p_infer.add_argument("--n-merchants", type=int, default=1000)
    p_infer.add_argument("--input-file", default=None, help="JSON file with sample data to score")

    args = parser.parse_args()
    manifest = manifests.get(args.dataset)
    if not manifest:
        print(f"Error: dataset '{args.dataset}' not found", file=sys.stderr)
        sys.exit(1)

    if args.command == "preview":
        cmd_preview(args, manifest)
    elif args.command == "export":
        cmd_export(args, manifest)
    elif args.command == "infer":
        cmd_infer(args, manifest)


if __name__ == "__main__":
    main()
