"""Train, compare, explain, and serialize the churn models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from .evaluate import evaluate_classifier, extract_feature_importance
from .preprocess import build_preprocessor, load_dataset

RANDOM_STATE = 42
CV_FOLDS = 5
F1_TIE_TOLERANCE = 0.005
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODEL_DIR = ROOT / "models"


def build_model_pipelines() -> dict[str, Pipeline]:
    """Return three deliberately modest, explainable baseline classifiers."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=240,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def add_xgboost_pipeline(pipelines: dict[str, Pipeline]) -> None:
    """Add XGBoost when the optional dependency is available."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("XGBoost is not installed; continuing with the two required baseline models.")
        return

    pipelines["XGBoost"] = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                XGBClassifier(
                    n_estimators=180,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=2,
                ),
            ),
        ]
    )


def cross_validate_models(
    pipelines: dict[str, Pipeline], X_train, y_train
) -> dict[str, dict[str, float]]:
    """Measure each candidate on the training partition using the same stratified folds."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"f1": "f1", "roc_auc": "roc_auc"}
    results: dict[str, dict[str, float]] = {}

    for name, pipeline in pipelines.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        results[name] = {
            "f1_mean": float(scores["test_f1"].mean()),
            "f1_std": float(scores["test_f1"].std()),
            "roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "roc_auc_std": float(scores["test_roc_auc"].std()),
        }

    return results


def select_model(cv_results: dict[str, dict[str, float]]) -> str:
    """Select by mean F1, using ROC-AUC only for practically tied F1 scores."""
    best_f1 = max(result["f1_mean"] for result in cv_results.values())
    tied_models = {
        name: result
        for name, result in cv_results.items()
        if best_f1 - result["f1_mean"] <= F1_TIE_TOLERANCE
    }
    return max(
        tied_models,
        key=lambda name: (tied_models[name]["roc_auc_mean"], tied_models[name]["f1_mean"]),
    )


def train() -> dict:
    """Cross-validate candidates, select one, and evaluate it on a final holdout set."""
    X, y = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipelines = build_model_pipelines()
    add_xgboost_pipeline(pipelines)

    print(f"Running {CV_FOLDS}-fold stratified cross-validation...")
    cv_results = cross_validate_models(pipelines, X_train, y_train)
    selected_name = select_model(cv_results)
    selected_pipeline = pipelines[selected_name]
    print(f"Selected by cross-validated F1: {selected_name}")

    # The final holdout is not used until after model selection is complete.
    selected_pipeline.fit(X_train, y_train)
    final_holdout_metrics = evaluate_classifier(selected_pipeline, X_test, y_test)
    feature_importance = extract_feature_importance(selected_pipeline)

    MODEL_DIR.mkdir(exist_ok=True)
    import joblib

    joblib.dump(selected_pipeline, MODEL_DIR / "churn_model.joblib")
    metadata = {
        "selected_model": selected_name,
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "dataset_rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "cv": {
            "folds": CV_FOLDS,
            "selection_metric": "f1_mean",
            "f1_tie_tolerance": F1_TIE_TOLERANCE,
            "results": cv_results,
        },
        "final_holdout_metrics": final_holdout_metrics,
        "feature_importance": feature_importance,
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    # Support both `python -m src.train` and direct execution from the repository root.
    if __package__ in (None, ""):
        sys.path.insert(0, str(ROOT))
        from src.train import train

    train()
