"""
CLI for CUDA/C code generation from XGBoost models.

Usage:
    python -m cuda_codegen generate --model models/model.ubj --output generated/scoring_kernel.cu
    python -m cuda_codegen generate --model models/model.ubj --output generated/scoring_kernel.c --cpu
    python -m cuda_codegen verify   --model models/model.ubj --n-samples 1000
"""

import argparse
import sys

from .generator import CudaCodeGenerator
from .verify import verify


def cmd_generate(args):
    """Generate CUDA or C code from an XGBoost model."""
    gen = CudaCodeGenerator(
        args.model,
        metadata_path=args.metadata,
    )

    user_fc = getattr(args, "user_features", None)
    library = getattr(args, "library", False)

    if library and user_fc:
        # Library mode: separate core + drivers + benchmark + Makefile
        files = gen.generate_split_library(args.output, user_fc)
        item_fc = gen.model_info.num_features - user_fc
        print(f"Generated split-feature library in: {args.output}")
        print(f"  Trees:          {gen.model_info.num_trees}")
        print(f"  User features:  {user_fc}  "
              f"[{', '.join(gen.model_info.feature_names[:user_fc])}]")
        print(f"  Item features:  {item_fc}  "
              f"[{', '.join(gen.model_info.feature_names[user_fc:user_fc+8])}"
              + (", ..." if item_fc > 8 else "") + "]")
        print(f"  Files:")
        for role, path in sorted(files.items()):
            print(f"    {role:12s}  {path}")
        print(f"\n  Build:  cd {args.output} && make cpu")
        print(f"  Bench:  cd {args.output} && make bench_cpu")
    elif user_fc:
        # Split-feature mode (single file)
        if args.cpu:
            path = gen.generate_cpu_split(args.output, user_fc)
        else:
            path = gen.generate_cuda_split(args.output, user_fc)
        item_fc = gen.model_info.num_features - user_fc
        print(f"Generated (split-feature): {path}")
        print(f"  Trees:          {gen.model_info.num_trees}")
        print(f"  User features:  {user_fc}  "
              f"[{', '.join(gen.model_info.feature_names[:user_fc])}]")
        print(f"  Item features:  {item_fc}  "
              f"[{', '.join(gen.model_info.feature_names[user_fc:user_fc+8])}"
              + (", ..." if item_fc > 8 else "") + "]")
    else:
        # Standard mode
        if args.cpu:
            path = gen.generate_cpu(args.output)
        else:
            path = gen.generate_cuda(args.output)
        print(f"Generated: {path}")
        print(f"  Trees:      {gen.model_info.num_trees}")
        print(f"  Features:   {gen.model_info.num_features}")

    print(f"  Objective:  {gen.model_info.objective}")
    print(f"  Base score: {gen.model_info.base_score:.8f}")


def cmd_verify(args):
    """Verify generated code matches XGBoost Python predictions."""
    user_fc = getattr(args, "user_features", None)
    topk = getattr(args, "topk", None)
    print(f"Verifying model: {args.model}")
    if user_fc:
        print(f"  Mode:        split-feature (user={user_fc})")
    if topk:
        print(f"  Top-K:       {topk}")
    print(f"  Samples:     {args.n_samples}")
    print(f"  NaN fraction: {args.nan_fraction:.0%}")
    print(f"  Tolerance:   {args.tolerance:.0e}")
    print()

    result = verify(
        args.model,
        n_samples=args.n_samples,
        metadata_path=args.metadata,
        tolerance=args.tolerance,
        nan_fraction=args.nan_fraction,
        user_feature_count=user_fc,
        topk=topk,
    )

    if result["success"]:
        print("  PASSED")
    else:
        if "error" in result:
            print(f"  FAILED: {result['error']}")
            sys.exit(1)
        print("  FAILED")

    if "max_diff" in result:
        print(f"  Max diff:        {result['max_diff']:.2e}")
        print(f"  Mean diff:       {result['mean_diff']:.2e}")
        print(f"  Expected range:  [{result['expected_range'][0]:.6f}, "
              f"{result['expected_range'][1]:.6f}]")
        print(f"  Actual range:    [{result['actual_range'][0]:.6f}, "
              f"{result['actual_range'][1]:.6f}]")

    if "index_agreement" in result:
        print(f"  Index agreement: {result['index_agreement']:.1%}")

    if "worst_samples" in result and not result["success"]:
        print("\n  Worst mismatches:")
        for s in result["worst_samples"]:
            print(f"    sample[{s['index']}]: expected={s['expected']:.8f}"
                  f" actual={s['actual']:.8f} diff={s['diff']:.2e}")

    if not result["success"]:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="cuda_codegen",
        description="Generate CUDA/C scoring code from XGBoost models",
    )
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    # --- generate ---
    gen_p = sub.add_parser(
        "generate",
        help="Generate CUDA or C source code from a trained XGBoost model",
    )
    gen_p.add_argument(
        "--model", required=True,
        help="Path to XGBoost model file (.ubj, .json, .bin)",
    )
    gen_p.add_argument(
        "--output", required=True,
        help="Output file path (.cu for CUDA, .c for CPU)",
    )
    gen_p.add_argument(
        "--metadata",
        help="Path to training_result.json (provides feature names)",
    )
    gen_p.add_argument(
        "--cpu", action="store_true",
        help="Generate CPU-only C code instead of CUDA",
    )
    gen_p.add_argument(
        "--user-features", type=int, default=None,
        help="Split-feature mode: number of user features (0..N-1). "
             "Remaining features are item features kept resident on GPU.",
    )
    gen_p.add_argument(
        "--library", action="store_true",
        help="Generate as a library: separate core header, main driver, "
             "benchmark driver, and Makefile. --output is a directory. "
             "Requires --user-features.",
    )
    gen_p.set_defaults(func=cmd_generate)

    # --- verify ---
    ver_p = sub.add_parser(
        "verify",
        help="Verify generated C code matches XGBoost Python predictions",
    )
    ver_p.add_argument(
        "--model", required=True,
        help="Path to XGBoost model file",
    )
    ver_p.add_argument(
        "--metadata",
        help="Path to training_result.json",
    )
    ver_p.add_argument(
        "--n-samples", type=int, default=1000,
        help="Number of random test samples (default: 1000)",
    )
    ver_p.add_argument(
        "--tolerance", type=float, default=1e-5,
        help="Max allowed absolute prediction difference (default: 1e-5)",
    )
    ver_p.add_argument(
        "--nan-fraction", type=float, default=0.05,
        help="Fraction of feature values to set to NaN (default: 0.05)",
    )
    ver_p.add_argument(
        "--user-features", type=int, default=None,
        help="Split-feature mode: number of user features (0..N-1)",
    )
    ver_p.add_argument(
        "--topk", type=int, default=None,
        help="Verify top-K ranking mode (requires --user-features)",
    )
    ver_p.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
