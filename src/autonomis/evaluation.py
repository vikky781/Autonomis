"""Score existing predictions; this module never constructs or trains a model."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def validate_scores(values, count):
    scores = np.asarray(values, dtype=float)
    if (scores.ndim != 1 or len(scores) != count or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()):
        raise ValueError("Predictions must be finite one-dimensional probabilities in [0, 1]")
    return scores


def positive_scores(model, X):
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    probabilities = np.asarray(model.predict_proba(X))
    row_count = X.shape[0] if hasattr(X, "shape") else len(X)
    if len(positive) != 1 or probabilities.shape != (row_count, len(classes)):
        raise ValueError("Invalid positive-class mapping or prediction shape")
    return validate_scores(probabilities[:, positive[0]], row_count)


def score_predictions(labels, values):
    labels = np.asarray(labels)
    if labels.ndim != 1 or set(labels.tolist()) != {0, 1}:
        raise ValueError("Evaluation requires both binary target classes")
    scores = validate_scores(values, len(labels))
    return {"roc_auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores))}
