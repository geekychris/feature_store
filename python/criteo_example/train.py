"""
XGBoost training for Click-Through Rate prediction on Criteo data.

Binary classification (click/no-click) with:
  - LogLoss as primary metric
  - AUC-ROC for ranking quality
  - Class-weight handling for imbalanced CTR (~3.4% positive rate)
  - Hyperparameter search
  - Calibration analysis

Designed for high-volume ad impression data (millions of rows).
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
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold

from dataset import FEATURE_NAMES, LABEL_COL, NUM_FEATURES


# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------
AUC_ROC_THRESHOLD = 0.65     # Lower than fraud since CTR is harder
LOGLOSS_THRESHOLD = 0.50     # Must be better than 0.50
AUC_PR_THRESHOLD = 0.08      # Low positive rate makes PR harder


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

@dataclass
class CTRTrainingResult:
    """Captures everything about a CTR training run."""
    model_name: str
    auc_roc: float
    logloss: float
    auc_pr: float
    score_std: float
    cv_auc_mean: float
    cv_auc_std: float
    feature_importance: dict
    hyperparams: dict
    train_size: int
    test_size: int
    positive_rate: float
    training_time_sec: float
    model_path: Optional[str] = None
    gates_passed: bool = False
    gate_details: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hyperparameter configurations
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 50,     # High because of large dataset
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cpu",
    "random_state": 42,
    "verbosity": 0,
}

PARAM_GRID = [
    {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 500, "min_child_weight": 100},
    {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 300, "min_child_weight": 50},
    {"max_depth": 8, "learning_rate": 0.08, "n_estimators": 200, "min_child_weight": 20},
    {"max_depth": 5, "learning_rate": 0.04, "n_estimators": 400, "min_child_weight": 75, "subsample": 0.7},
    {"max_depth": 7, "learning_rate": 0.06, "n_estimators": 250, "min_child_weight": 30, "colsample_bytree": 0.7},
]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ctr_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: Optional[dict] = None,
    n_cv_folds: int = 5,
) -> CTRTrainingResult:
    """
    Train an XGBoost binary classifier for CTR prediction.

    Handles class imbalance via scale_pos_weight.
    Evaluates with LogLoss, AUC-ROC, and AUC-PR.
    """
    t0 = time.time()

    hp = {**DEFAULT_PARAMS}
    if params:
        hp.update(params)

    # Handle class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = max((y_train == 1).sum(), 1)
    hp["scale_pos_weight"] = n_neg / n_pos

    # --- Cross-validation ---
    skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
    cv_auc = []

    for fold_train_idx, fold_val_idx in skf.split(X_train, y_train):
        xf_train, xf_val = X_train[fold_train_idx], X_train[fold_val_idx]
        yf_train, yf_val = y_train[fold_train_idx], y_train[fold_val_idx]

        clf = xgb.XGBClassifier(**hp)
        clf.fit(
            xf_train, yf_train,
            eval_set=[(xf_val, yf_val)],
            verbose=False,
        )
        y_val_prob = clf.predict_proba(xf_val)[:, 1]
        cv_auc.append(roc_auc_score(yf_val, y_val_prob))

    cv_mean = float(np.mean(cv_auc))
    cv_std = float(np.std(cv_auc))

    # --- Final model ---
    final_clf = xgb.XGBClassifier(**hp)
    final_clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # --- Evaluate ---
    y_prob = final_clf.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)
    score_std = float(np.std(y_prob))

    # Feature importance
    importance = final_clf.feature_importances_
    importance_dict = {
        name: float(imp) for name, imp in zip(FEATURE_NAMES, importance)
    }
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: -x[1]))

    training_time = time.time() - t0

    # --- Validation gates ---
    gates = [
        ("AUC-ROC", auc_roc >= AUC_ROC_THRESHOLD, auc_roc, AUC_ROC_THRESHOLD),
        ("LogLoss", ll <= LOGLOSS_THRESHOLD, ll, LOGLOSS_THRESHOLD),
        ("AUC-PR", auc_pr >= AUC_PR_THRESHOLD, auc_pr, AUC_PR_THRESHOLD),
    ]
    all_passed = all(g[1] for g in gates)

    result = CTRTrainingResult(
        model_name="criteo_ctr_xgb",
        auc_roc=auc_roc,
        logloss=ll,
        auc_pr=auc_pr,
        score_std=score_std,
        cv_auc_mean=cv_mean,
        cv_auc_std=cv_std,
        feature_importance=importance_dict,
        hyperparams={k: v for k, v in hp.items() if k != "verbosity"},
        train_size=len(X_train),
        test_size=len(X_test),
        positive_rate=float(y_test.mean()),
        training_time_sec=training_time,
        gates_passed=all_passed,
        gate_details=[
            {"gate": g[0], "passed": g[1], "value": g[2], "threshold": g[3]}
            for g in gates
        ],
    )

    result._model = final_clf
    result._y_prob = y_prob
    return result


def hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_cv_folds: int = 5,
) -> CTRTrainingResult:
    """Run hyperparameter search over PARAM_GRID, return best result."""
    best_result = None
    print(f"  Searching {len(PARAM_GRID)} configurations...")

    for i, param_override in enumerate(PARAM_GRID):
        result = train_ctr_model(X_train, y_train, X_test, y_test,
                                 params=param_override, n_cv_folds=n_cv_folds)
        tag = "✓" if result.gates_passed else "✗"
        print(f"    [{i+1}/{len(PARAM_GRID)}] depth={param_override.get('max_depth', '?')}"
              f" lr={param_override.get('learning_rate', '?')}"
              f" → AUC={result.auc_roc:.4f}"
              f" LL={result.logloss:.4f}"
              f" CV={result.cv_auc_mean:.4f}±{result.cv_auc_std:.4f} {tag}")

        if best_result is None or result.auc_roc > best_result.auc_roc:
            best_result = result

    return best_result


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(result: CTRTrainingResult, output_dir: str = "models") -> str:
    """Save the trained CTR model and metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = result._model

    xgb_path = out / "model.ubj"
    model.save_model(str(xgb_path))

    joblib_path = out / "model.joblib"
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


def load_model(model_dir: str = "models") -> xgb.XGBClassifier:
    """Load a trained CTR model from disk."""
    model_path = Path(model_dir) / "model.ubj"
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))
    return clf


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_training_report(result: CTRTrainingResult):
    """Print a detailed CTR training report."""
    print(f"\n{'='*60}")
    print(f"CTR Training Report: {result.model_name}")
    print(f"{'='*60}")
    print(f"  Train size:      {result.train_size:,}")
    print(f"  Test size:       {result.test_size:,}")
    print(f"  Positive rate:   {result.positive_rate*100:.2f}%")
    print(f"  Training time:   {result.training_time_sec:.1f}s")
    print()
    print(f"  CTR Metrics:")
    print(f"    AUC-ROC:       {result.auc_roc:.4f}")
    print(f"    LogLoss:       {result.logloss:.4f}")
    print(f"    AUC-PR:        {result.auc_pr:.4f}")
    print(f"    Score std:     {result.score_std:.4f}")
    print(f"    CV AUC-ROC:    {result.cv_auc_mean:.4f} ± {result.cv_auc_std:.4f}")
    print()
    print(f"  Validation Gates:")
    for gate in result.gate_details:
        status = "PASS" if gate["passed"] else "FAIL"
        direction = "<=" if gate["gate"] == "LogLoss" else ">="
        print(f"    [{status}] {gate['gate']}: {gate['value']:.4f} "
              f"({direction} {gate['threshold']})")
    print(f"  Overall: {'ALL GATES PASSED' if result.gates_passed else 'GATES FAILED'}")
    print()
    print(f"  Top 15 Feature Importance:")
    for i, (name, imp) in enumerate(list(result.feature_importance.items())[:15]):
        bar = "█" * int(imp * 100)
        print(f"    {i+1:2d}. {name:<8s} {imp:.4f} {bar}")

    # Calibration analysis
    if hasattr(result, '_y_prob'):
        probs = result._y_prob
        print()
        print(f"  Score Distribution:")
        print(f"    Mean:   {probs.mean():.4f}")
        print(f"    Std:    {probs.std():.4f}")
        print(f"    P10:    {np.percentile(probs, 10):.4f}")
        print(f"    P50:    {np.percentile(probs, 50):.4f}")
        print(f"    P90:    {np.percentile(probs, 90):.4f}")
        print(f"    P99:    {np.percentile(probs, 99):.4f}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Standalone training
# ---------------------------------------------------------------------------

def train_from_dataframes(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    search_hyperparams: bool = True,
    save_dir: Optional[str] = "models",
) -> CTRTrainingResult:
    """Complete CTR training pipeline from DataFrames."""
    X_train = train_df[FEATURE_NAMES].fillna(0).values
    y_train = train_df[LABEL_COL].values
    X_test = test_df[FEATURE_NAMES].fillna(0).values
    y_test = test_df[LABEL_COL].values

    if search_hyperparams:
        result = hyperparameter_search(X_train, y_train, X_test, y_test)
    else:
        result = train_ctr_model(X_train, y_train, X_test, y_test)

    print_training_report(result)

    if save_dir:
        path = save_model(result, save_dir)
        print(f"\n  Model saved to: {path}")

    return result


if __name__ == "__main__":
    from dataset import load_dataset

    print("Loading synthetic Criteo CTR dataset...")
    train_df, test_df = load_dataset(synthetic=True, n_samples=50_000)

    print("\nTraining CTR model...")
    result = train_from_dataframes(train_df, test_df, search_hyperparams=True, save_dir="models")
