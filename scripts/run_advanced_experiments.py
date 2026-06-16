#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from src.common import ensure_dir, load_yaml, project_path, read_json, write_json
from src.modeling.evaluate import evaluate_predictions
from src.modeling.features import build_feature_matrix


UNSAFE_FEATURE_CHARS = re.compile(r"[^0-9A-Za-z_]+")
ANSWER_FEATURE_RE = re.compile(r"^answer__(q_[0-9a-f]+)_(.+)$")


def _safe_name(value: str) -> str:
    safe = UNSAFE_FEATURE_CHARS.sub("_", str(value)).strip("_")
    if not safe:
        safe = "feature"
    if safe[0].isdigit():
        safe = f"f_{safe}"
    return safe[:180].rstrip("_") or "feature"


def _load_catalog(path: Path) -> Dict[str, Dict[str, Any]]:
    data = read_json(path, default={})
    questions = data.get("questions", []) if isinstance(data, dict) else []
    catalog: Dict[str, Dict[str, Any]] = {}
    for question in questions:
        key = question.get("question_key")
        if not key:
            continue
        option_labels = list((question.get("option_labels") or {}).keys())
        catalog[key] = {
            "question_text": question.get("question_text", ""),
            "question_type": question.get("question_type", ""),
            "seen_count": question.get("seen_count", 0),
            "option_labels": option_labels,
            "safe_options": {_safe_name(label): label for label in option_labels},
        }
    return catalog


def _describe_feature(feature: str, catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    match = ANSWER_FEATURE_RE.match(feature)
    if not match:
        return {
            "feature_type": "derived",
            "question_key": "",
            "question_text": "",
            "question_type": "",
            "option_text": "",
            "question_seen_count": "",
        }
    question_key, option_safe = match.groups()
    question = catalog.get(question_key, {})
    safe_options = question.get("safe_options", {})
    option_text = safe_options.get(option_safe, "")
    if not option_text:
        matches = [label for safe, label in safe_options.items() if safe and safe in option_safe]
        option_text = " | ".join(matches) if matches else option_safe.replace("_", " ")
    return {
        "feature_type": "answer",
        "question_key": question_key,
        "question_text": question.get("question_text", ""),
        "question_type": question.get("question_type", ""),
        "option_text": option_text,
        "question_seen_count": question.get("seen_count", ""),
    }


def _load_dataset(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Dict[str, Any]]:
    modeling = config.get("modeling", {})
    df = pd.read_csv(modeling["dataset_path"], keep_default_na=False, na_values=[""])
    X, y = build_feature_matrix(df, label_column=modeling.get("label_column", "result_age_rating"))
    return df, X, y, modeling


def _split_indices(y: pd.Series, modeling: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    return train_test_split(
        indices,
        test_size=float(modeling.get("test_size", 0.15)),
        random_state=int(modeling.get("random_seed", 42)),
        stratify=y,
    )


def _align_features(X: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    return X.reindex(columns=list(features), fill_value=0)


def write_readable_top_features(metrics_dir: Path, output_dir: Path, catalog: Dict[str, Dict[str, Any]], top_n: int) -> None:
    explanation_dir = project_path("outputs/analysis/current/explanations")
    rows: List[Dict[str, Any]] = []
    for path in sorted(explanation_dir.glob("*_feature_importance.csv")):
        model = path.name.replace("_feature_importance.csv", "")
        data = pd.read_csv(path).head(top_n)
        for rank, row in enumerate(data.itertuples(index=False), start=1):
            feature = str(row.feature)
            rows.append(
                {
                    "model": model,
                    "rank": rank,
                    "feature": feature,
                    "importance": getattr(row, "importance"),
                    **_describe_feature(feature, catalog),
                }
            )
    readable = pd.DataFrame(rows)
    readable.to_csv(metrics_dir / "top_features_readable.csv", index=False)

    lines = ["# Top Features With Question Text", ""]
    for model, group in readable.groupby("model", sort=True):
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| Rank | Feature | Question | Option | Importance |")
        lines.append("|---:|---|---|---|---:|")
        for row in group.head(top_n).itertuples(index=False):
            question = str(row.question_text).replace("|", "/")
            option = str(row.option_text).replace("|", "/")
            lines.append(f"| {row.rank} | `{row.feature}` | {question} | {option} | {float(row.importance):.6f} |")
        lines.append("")
    (output_dir / "top_features_readable.md").write_text("\n".join(lines), encoding="utf-8")


def run_counterfactual_flips(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    modeling: Dict[str, Any],
    metrics_dir: Path,
    catalog: Dict[str, Dict[str, Any]],
    max_features: int,
    max_rows_per_feature: int,
) -> None:
    payload = joblib.load(project_path("outputs/analysis/current/models/optimized_lightgbm.joblib"))
    model = payload["model"]
    X_model = _align_features(X, payload["features"])
    _, test_idx = _split_indices(y, modeling)
    X_test = X_model.iloc[test_idx]
    y_test = y.iloc[test_idx]
    base_pred = pd.Series(model.predict(X_test), index=X_test.index)
    base_proba = model.predict_proba(X_test)
    base_conf = pd.Series(base_proba.max(axis=1), index=X_test.index)

    importance = pd.read_csv(project_path("outputs/analysis/current/explanations/lightgbm_feature_importance.csv"))
    candidate_features = [
        feature
        for feature in importance["feature"].tolist()
        if feature.startswith("answer__") and feature in X_test.columns
    ][:max_features]

    rows: List[Dict[str, Any]] = []
    for feature in candidate_features:
        flipped = X_test.copy()
        original_values = flipped[feature].astype(int)
        flipped[feature] = 1 - original_values
        flipped_pred = pd.Series(model.predict(flipped), index=X_test.index)
        flipped_proba = model.predict_proba(flipped)
        flipped_conf = pd.Series(flipped_proba.max(axis=1), index=X_test.index)
        changed_indices = [idx for idx in X_test.index if base_pred.loc[idx] != flipped_pred.loc[idx]][:max_rows_per_feature]
        for idx in changed_indices:
            source = df.iloc[int(idx)]
            rows.append(
                {
                    "sample_id": source.get("sample_id"),
                    "true_label": y_test.loc[idx],
                    "base_prediction": base_pred.loc[idx],
                    "flipped_prediction": flipped_pred.loc[idx],
                    "base_confidence": float(base_conf.loc[idx]),
                    "flipped_confidence": float(flipped_conf.loc[idx]),
                    "feature": feature,
                    "original_value": int(original_values.loc[idx]),
                    "flipped_value": int(flipped.loc[idx, feature]),
                    **_describe_feature(feature, catalog),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(metrics_dir / "counterfactual_feature_flips.csv", index=False)
    transitions = Counter(f"{row['base_prediction']}->{row['flipped_prediction']}" for row in rows)
    write_json(
        metrics_dir / "counterfactual_summary.json",
        {
            "model": "optimized_lightgbm",
            "tested_features": len(candidate_features),
            "changed_prediction_examples": int(len(result)),
            "max_rows_per_feature": max_rows_per_feature,
            "note": "Feature flips are model-level perturbations and may not correspond to valid questionnaire paths.",
            "prediction_transitions": dict(sorted(transitions.items())),
        },
    )


def run_ordinal_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    modeling: Dict[str, Any],
    metrics_dir: Path,
) -> None:
    seed = int(modeling.get("random_seed", 42))
    age_order = list(modeling.get("age_order", ["3+", "7+", "12+", "16+", "18+"]))
    labels = [label for label in age_order if label in set(y)]
    label_to_level = {label: index for index, label in enumerate(age_order)}
    train_idx, test_idx = _split_indices(y, modeling)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    optimized = read_json(metrics_dir / "optimized_model_metrics.json", default={})
    params = optimized.get("lightgbm", {}).get(
        "best_params",
        {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.9, "num_leaves": 15, "min_child_samples": 10},
    )

    threshold_rows = []
    probabilities = []
    thresholds = age_order[1:]
    for threshold in thresholds:
        threshold_level = label_to_level[threshold]
        binary_train = y_train.map(lambda label: int(label_to_level[label] >= threshold_level))
        binary_test = y_test.map(lambda label: int(label_to_level[label] >= threshold_level))
        weights = compute_sample_weight(class_weight="balanced", y=binary_train)
        model = LGBMClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
            force_row_wise=True,
            verbosity=-1,
            **params,
        )
        model.fit(X_train, binary_train, sample_weight=weights)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        probabilities.append(proba)
        threshold_rows.append(
            {
                "threshold": f">={threshold}",
                "positive_support": int(binary_test.sum()),
                "accuracy": accuracy_score(binary_test, pred),
                "precision": precision_score(binary_test, pred, zero_division=0),
                "recall": recall_score(binary_test, pred, zero_division=0),
                "f1": f1_score(binary_test, pred, zero_division=0),
            }
        )

    probs = np.vstack(probabilities).T
    monotonic_probs = np.minimum.accumulate(probs, axis=1)
    pred_levels = (monotonic_probs >= 0.5).sum(axis=1)
    pred_labels = pd.Series([age_order[level] for level in pred_levels], index=y_test.index)
    metrics = evaluate_predictions(
        y_test,
        pred_labels,
        labels,
        age_order,
        int(modeling.get("severe_error_threshold", 2)),
    )
    metrics["model"] = "ordinal_lightgbm_thresholds"
    metrics["thresholds"] = [f">={threshold}" for threshold in thresholds]
    write_json(metrics_dir / "ordinal_model_metrics.json", metrics)
    pd.DataFrame(threshold_rows).to_csv(metrics_dir / "ordinal_threshold_metrics.csv", index=False)


def run_top2_confidence(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    modeling: Dict[str, Any],
    metrics_dir: Path,
) -> None:
    payload = joblib.load(project_path("outputs/analysis/current/models/optimized_lightgbm.joblib"))
    model = payload["model"]
    X_model = _align_features(X, payload["features"])
    _, test_idx = _split_indices(y, modeling)
    X_test = X_model.iloc[test_idx]
    y_test = y.iloc[test_idx]
    proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    top_indices = np.argsort(proba, axis=1)[:, ::-1]

    rows = []
    for row_pos, idx in enumerate(X_test.index):
        top1_idx, top2_idx = top_indices[row_pos, 0], top_indices[row_pos, 1]
        true_label = y_test.loc[idx]
        rows.append(
            {
                "sample_id": df.iloc[int(idx)].get("sample_id"),
                "true_label": true_label,
                "top1_label": classes[top1_idx],
                "top1_confidence": float(proba[row_pos, top1_idx]),
                "top2_label": classes[top2_idx],
                "top2_confidence": float(proba[row_pos, top2_idx]),
                "top1_correct": int(classes[top1_idx] == true_label),
                "top2_contains_true": int(true_label in {classes[top1_idx], classes[top2_idx]}),
            }
        )
    predictions = pd.DataFrame(rows)
    predictions.to_csv(metrics_dir / "top2_predictions.csv", index=False)

    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    predictions["confidence_bin"] = pd.cut(predictions["top1_confidence"], bins=bins, include_lowest=True)
    calibration = (
        predictions.groupby("confidence_bin", observed=True)
        .agg(
            count=("sample_id", "count"),
            avg_confidence=("top1_confidence", "mean"),
            top1_accuracy=("top1_correct", "mean"),
            top2_accuracy=("top2_contains_true", "mean"),
        )
        .reset_index()
    )
    calibration["confidence_bin"] = calibration["confidence_bin"].astype(str)
    calibration.to_csv(metrics_dir / "confidence_bins.csv", index=False)

    write_json(
        metrics_dir / "top2_confidence_metrics.json",
        {
            "model": "optimized_lightgbm",
            "sample_count": int(len(predictions)),
            "top1_accuracy": float(predictions["top1_correct"].mean()),
            "top2_accuracy": float(predictions["top2_contains_true"].mean()),
            "mean_top1_confidence": float(predictions["top1_confidence"].mean()),
            "median_top1_confidence": float(predictions["top1_confidence"].median()),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run advanced post-hoc experiments for the age-rating task.")
    parser.add_argument("--config", default="configs/modeling.yaml")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--counterfactual-features", type=int, default=25)
    parser.add_argument("--counterfactual-rows-per-feature", type=int, default=5)
    args = parser.parse_args()

    config = load_yaml(args.config)
    df, X, y, modeling = _load_dataset(config)
    metrics_dir = ensure_dir(modeling.get("metrics_dir", "outputs/analysis/current/metrics"))
    advanced_dir = ensure_dir("outputs/analysis/current/advanced")
    catalog_path = project_path("data/questionnaire/real_question_catalog_20260615_full.json")
    catalog = _load_catalog(catalog_path)

    write_readable_top_features(metrics_dir, advanced_dir, catalog, args.top_n)
    run_counterfactual_flips(
        df,
        X,
        y,
        modeling,
        metrics_dir,
        catalog,
        args.counterfactual_features,
        args.counterfactual_rows_per_feature,
    )
    run_ordinal_experiment(X, y, modeling, metrics_dir)
    run_top2_confidence(df, X, y, modeling, metrics_dir)

    print(f"wrote {metrics_dir / 'top_features_readable.csv'}")
    print(f"wrote {advanced_dir / 'top_features_readable.md'}")
    print(f"wrote {metrics_dir / 'counterfactual_feature_flips.csv'}")
    print(f"wrote {metrics_dir / 'counterfactual_summary.json'}")
    print(f"wrote {metrics_dir / 'ordinal_model_metrics.json'}")
    print(f"wrote {metrics_dir / 'ordinal_threshold_metrics.csv'}")
    print(f"wrote {metrics_dir / 'top2_confidence_metrics.json'}")
    print(f"wrote {metrics_dir / 'top2_predictions.csv'}")
    print(f"wrote {metrics_dir / 'confidence_bins.csv'}")


if __name__ == "__main__":
    main()
