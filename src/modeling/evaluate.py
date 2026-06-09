from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def severe_error_rate(y_true: Iterable[str], y_pred: Iterable[str], age_order: List[str], threshold: int = 2) -> float:
    order = {label: index for index, label in enumerate(age_order)}
    total = 0
    severe = 0
    for true, pred in zip(y_true, y_pred):
        if true not in order or pred not in order:
            continue
        total += 1
        severe += int(abs(order[true] - order[pred]) >= threshold)
    return severe / total if total else 0.0


def mean_absolute_age_error(y_true: Iterable[str], y_pred: Iterable[str], age_order: List[str]) -> float:
    order = {label: index for index, label in enumerate(age_order)}
    errors = [
        abs(order[true] - order[pred])
        for true, pred in zip(y_true, y_pred)
        if true in order and pred in order
    ]
    return float(np.mean(errors)) if errors else 0.0


def evaluate_predictions(y_true, y_pred, labels: List[str], age_order: List[str], severe_threshold: int = 2) -> Dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mean_absolute_age_error": mean_absolute_age_error(y_true, y_pred, age_order),
        "severe_error_rate": severe_error_rate(y_true, y_pred, age_order, severe_threshold),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }
