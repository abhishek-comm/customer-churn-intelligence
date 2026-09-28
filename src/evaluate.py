"""Evaluation and explainability helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .preprocess import clean_feature_name


def evaluate_classifier(model: Any, X_test: Any, y_test: Any) -> dict[str, float]:
    """Calculate the metrics required by the assignment on an untouched holdout set."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def extract_feature_importance(pipeline: Any, top_n: int = 12) -> list[dict[str, float | str]]:
    """Return the strongest transformed features for a tree or linear estimator."""
    transformer = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    feature_names = [clean_feature_name(name) for name in transformer.get_feature_names_out()]

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        importance_type = "feature_importance"
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_[0], dtype=float))
        importance_type = "absolute_coefficient"
    else:
        return []

    ranked = sorted(zip(feature_names, values), key=lambda item: item[1], reverse=True)[:top_n]
    return [
        {"feature": feature, "importance": round(float(value), 6), "method": importance_type}
        for feature, value in ranked
    ]
