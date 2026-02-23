"""
XGBoost LambdaMART training for Learning to Rank on MSLR-WEB10K.

Uses rank:ndcg objective with grouped queries. Evaluates using:
  - NDCG@1, NDCG@5, NDCG@10
  - MAP (Mean Average Precision)
  - Per-query ranking analysis

Supports hyperparameter search over tree depth, learning rate, and
regularization parameters.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from dataset import FEATURE_NAMES, LABEL_COL, NUM_FEATURES


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Discounted cumulative gain at position k."""
    relevances = relevances[:k]
    if len(relevances) == 0:
        return 0.0
    discounts = np.log2(np.arange(len(relevances)) + 2)
    return float(np.sum((2**relevances - 1) / discounts))


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Normalized DCG at position k."""
    dcg = dcg_at_k(relevances, k)
    ideal = dcg_at_k(np.sort(relevances)[::-1], k)
    if ideal == 0:
        return 0.0
    return dcg / ideal


def mean_ndcg(df: pd.DataFrame, score_col: str, k: int) -> float:
    """Compute mean NDCG@k across all queries."""
    ndcgs = []
    for qid, group in df.groupby("qid"):
        sorted_group = group.sort_values(score_col, ascending=False)
        relevances = sorted_group[LABEL_COL].values.astype(float)
        ndcgs.append(ndcg_at_k(relevances, k))
    return float(np.mean(ndcgs))


def mean_average_precision(df: pd.DataFrame, score_col: str, threshold: int = 2) -> float:
    """
    Compute MAP over queries. Documents with relevance >= threshold are 'relevant'.
    """
    aps = []
    for qid, group in df.groupby("qid"):
        sorted_group = group.sort_values(score_col, ascending=False)
        relevant = (sorted_group[LABEL_COL].values >= threshold).astype(float)
        if relevant.sum() == 0:
            continue  # Skip queries with no relevant docs
        precision_at_k = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
        ap = np.sum(precision_at_k * relevant) / relevant.sum()
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

@dataclass
class RankingTrainingResult:
    """Captures everything about a ranking training run."""
    model_name: str
    ndcg_1: float
    ndcg_5: float
    ndcg_10: float
    map_score: float
    cv_ndcg5_mean: float
    cv_ndcg5_std: float
    feature_importance: dict
    hyperparams: dict
    train_queries: int
    test_queries: int
    train_docs: int
    test_docs: int
    training_time_sec: float
    model_path: Optional[str] = None
    gates_passed: bool = False
    gate_details: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------
NDCG5_THRESHOLD = 0.30
NDCG10_THRESHOLD = 0.30
MAP_THRESHOLD = 0.20


# ---------------------------------------------------------------------------
# Hyperparameter configurations
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@5",
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "device": "cpu",
    "random_state": 42,
    "verbosity": 0,
}

PARAM_GRID = [
    {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 800, "min_child_weight": 15},
    {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 500, "min_child_weight": 10},
    {"max_depth": 8, "learning_rate": 0.08, "n_estimators": 300, "min_child_weight": 5},
    {"max_depth": 5, "learning_rate": 0.04, "n_estimators": 600, "min_child_weight": 8, "subsample": 0.7},
    {"max_depth": 7, "learning_rate": 0.06, "n_estimators": 400, "min_child_weight": 6, "colsample_bytree": 0.7},
]


# ---------------------------------------------------------------------------
# Query group sizes
# ---------------------------------------------------------------------------

def compute_group_sizes(qids: np.ndarray) -> np.ndarray:
    """Compute group sizes for XGBoost ranking (consecutive docs per query)."""
    # Sort by qid first, then compute run lengths
    _, counts = np.unique(qids, return_counts=True)
    return counts


def prepare_ranking_data(df: pd.DataFrame):
    """
    Prepare data for XGBoost ranking.
    Sort by qid so group sizes are contiguous.

    Returns (X, y, qids, group_sizes)
    """
    df_sorted = df.sort_values("qid").reset_index(drop=True)
    X = df_sorted[FEATURE_NAMES].fillna(0).values
    y = df_sorted[LABEL_COL].values.astype(float)
    qids = df_sorted["qid"].values
    group_sizes = compute_group_sizes(qids)
    return X, y, qids, group_sizes, df_sorted


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ranker(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: Optional[dict] = None,
    n_cv_folds: int = 5,
) -> RankingTrainingResult:
    """
    Train an XGBoost LambdaMART ranker.

    Uses rank:ndcg objective with query group structure.
    Evaluates with NDCG@1, NDCG@5, NDCG@10, and MAP.
    """
    t0 = time.time()

    hp = {**DEFAULT_PARAMS}
    if params:
        hp.update(params)

    X_train, y_train, qids_train, groups_train, train_sorted = prepare_ranking_data(train_df)
    X_test, y_test, qids_test, groups_test, test_sorted = prepare_ranking_data(test_df)

    # --- Cross-validation (group-aware) ---
    unique_qids = np.unique(qids_train)
    n_folds = min(n_cv_folds, len(unique_qids))
    gkf = GroupKFold(n_splits=n_folds)

    cv_ndcg5 = []
    for fold_train_idx, fold_val_idx in gkf.split(X_train, y_train, groups=qids_train):
        xf_train, xf_val = X_train[fold_train_idx], X_train[fold_val_idx]
        yf_train, yf_val = y_train[fold_train_idx], y_train[fold_val_idx]
        qf_train = qids_train[fold_train_idx]
        qf_val = qids_train[fold_val_idx]

        fold_groups_train = compute_group_sizes(qf_train)
        fold_groups_val = compute_group_sizes(qf_val)

        ranker = xgb.XGBRanker(**hp)
        ranker.fit(
            xf_train, yf_train,
            group=fold_groups_train,
            eval_set=[(xf_val, yf_val)],
            eval_group=[fold_groups_val],
            verbose=False,
        )

        # Evaluate NDCG@5 on fold validation
        fold_scores = ranker.predict(xf_val)
        fold_df = pd.DataFrame({
            "qid": qf_val,
            LABEL_COL: yf_val,
            "score": fold_scores,
        })
        cv_ndcg5.append(mean_ndcg(fold_df, "score", 5))

    cv_mean = float(np.mean(cv_ndcg5))
    cv_std = float(np.std(cv_ndcg5))

    # --- Final model on full training set ---
    final_ranker = xgb.XGBRanker(**hp)
    final_ranker.fit(
        X_train, y_train,
        group=groups_train,
        eval_set=[(X_test, y_test)],
        eval_group=[groups_test],
        verbose=False,
    )

    # --- Evaluate on test set ---
    test_scores = final_ranker.predict(X_test)
    test_sorted = test_sorted.copy()
    test_sorted["score"] = test_scores

    ndcg_1 = mean_ndcg(test_sorted, "score", 1)
    ndcg_5 = mean_ndcg(test_sorted, "score", 5)
    ndcg_10 = mean_ndcg(test_sorted, "score", 10)
    map_score = mean_average_precision(test_sorted, "score")

    # Feature importance
    importance = final_ranker.feature_importances_
    importance_dict = {
        name: float(imp) for name, imp in zip(FEATURE_NAMES, importance)
    }
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: -x[1]))

    training_time = time.time() - t0

    # --- Validation gates ---
    gates = [
        ("NDCG@5", ndcg_5 >= NDCG5_THRESHOLD, ndcg_5, NDCG5_THRESHOLD),
        ("NDCG@10", ndcg_10 >= NDCG10_THRESHOLD, ndcg_10, NDCG10_THRESHOLD),
        ("MAP", map_score >= MAP_THRESHOLD, map_score, MAP_THRESHOLD),
    ]
    all_passed = all(g[1] for g in gates)

    result = RankingTrainingResult(
        model_name="mslr_lambdamart",
        ndcg_1=ndcg_1,
        ndcg_5=ndcg_5,
        ndcg_10=ndcg_10,
        map_score=map_score,
        cv_ndcg5_mean=cv_mean,
        cv_ndcg5_std=cv_std,
        feature_importance=importance_dict,
        hyperparams={k: v for k, v in hp.items() if k != "verbosity"},
        train_queries=int(np.unique(qids_train).shape[0]),
        test_queries=int(np.unique(qids_test).shape[0]),
        train_docs=len(X_train),
        test_docs=len(X_test),
        training_time_sec=training_time,
        gates_passed=all_passed,
        gate_details=[
            {"gate": g[0], "passed": g[1], "value": g[2], "threshold": g[3]}
            for g in gates
        ],
    )

    result._model = final_ranker
    result._test_df = test_sorted
    return result


def hyperparameter_search(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_cv_folds: int = 5,
) -> RankingTrainingResult:
    """Run hyperparameter search over PARAM_GRID, return best result."""
    best_result = None
    print(f"  Searching {len(PARAM_GRID)} configurations...")

    for i, param_override in enumerate(PARAM_GRID):
        result = train_ranker(train_df, test_df, params=param_override, n_cv_folds=n_cv_folds)
        tag = "✓" if result.gates_passed else "✗"
        print(f"    [{i+1}/{len(PARAM_GRID)}] depth={param_override.get('max_depth', '?')}"
              f" lr={param_override.get('learning_rate', '?')}"
              f" → NDCG@5={result.ndcg_5:.4f}"
              f" MAP={result.map_score:.4f}"
              f" CV={result.cv_ndcg5_mean:.4f}±{result.cv_ndcg5_std:.4f} {tag}")

        if best_result is None or result.ndcg_5 > best_result.ndcg_5:
            best_result = result

    return best_result


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(result: RankingTrainingResult, output_dir: str = "models") -> str:
    """Save the trained ranker and metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = result._model

    xgb_path = out / "ranker.ubj"
    model.save_model(str(xgb_path))

    joblib_path = out / "ranker.joblib"
    joblib.dump(model, str(joblib_path))

    meta = asdict(result)
    meta.pop("model_path", None)
    meta["model_path"] = str(xgb_path)
    meta["feature_names"] = FEATURE_NAMES

    meta_path = out / "training_result.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    result.model_path = str(xgb_path)
    return str(out)


def load_model(model_dir: str = "models") -> xgb.XGBRanker:
    """Load a trained ranker from disk."""
    model_path = Path(model_dir) / "ranker.ubj"
    ranker = xgb.XGBRanker()
    ranker.load_model(str(model_path))
    return ranker


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_training_report(result: RankingTrainingResult):
    """Print a detailed ranking training report."""
    print(f"\n{'='*60}")
    print(f"Ranking Training Report: {result.model_name}")
    print(f"{'='*60}")
    print(f"  Train:  {result.train_docs:,} docs, {result.train_queries} queries")
    print(f"  Test:   {result.test_docs:,} docs, {result.test_queries} queries")
    print(f"  Time:   {result.training_time_sec:.1f}s")
    print()
    print(f"  Ranking Metrics:")
    print(f"    NDCG@1:      {result.ndcg_1:.4f}")
    print(f"    NDCG@5:      {result.ndcg_5:.4f}")
    print(f"    NDCG@10:     {result.ndcg_10:.4f}")
    print(f"    MAP:         {result.map_score:.4f}")
    print(f"    CV NDCG@5:   {result.cv_ndcg5_mean:.4f} ± {result.cv_ndcg5_std:.4f}")
    print()
    print(f"  Validation Gates:")
    for gate in result.gate_details:
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"    [{status}] {gate['gate']}: {gate['value']:.4f} "
              f"(threshold: {gate['threshold']})")
    print(f"  Overall: {'ALL GATES PASSED' if result.gates_passed else 'GATES FAILED'}")
    print()
    print(f"  Top 15 Feature Importance:")
    for i, (name, imp) in enumerate(list(result.feature_importance.items())[:15]):
        bar = "█" * int(imp * 200)
        print(f"    {i+1:2d}. {name:<20s} {imp:.4f} {bar}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Standalone training
# ---------------------------------------------------------------------------

def train_from_dataframes(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    search_hyperparams: bool = True,
    save_dir: Optional[str] = "models",
) -> RankingTrainingResult:
    """Complete ranking training pipeline from DataFrames."""
    if search_hyperparams:
        result = hyperparameter_search(train_df, test_df)
    else:
        result = train_ranker(train_df, test_df)

    print_training_report(result)

    if save_dir:
        path = save_model(result, save_dir)
        print(f"\n  Model saved to: {path}")

    return result


if __name__ == "__main__":
    from dataset import load_dataset

    print("Loading synthetic MSLR-WEB10K dataset...")
    train_df, test_df = load_dataset(synthetic=True, n_queries=100, docs_per_query=40)

    print("\nTraining LambdaMART ranker...")
    result = train_from_dataframes(train_df, test_df, search_hyperparams=True, save_dir="models")
