#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from src.common import ensure_dir, load_yaml, project_path, read_json, write_json
from src.data.storage import JsonlStore
from src.modeling.features import build_feature_matrix, records_to_dataframe


DEFAULT_INPUT = "data/raw/real_20260615_full.samples.jsonl"
DEFAULT_METRICS_OUTPUT = "outputs/analysis/current/metrics/region_rating_model_metrics.json"
DEFAULT_SUMMARY_OUTPUT = "outputs/analysis/current/metrics/region_rating_model_summary.csv"
DEFAULT_CONFUSION_DIR = "outputs/analysis/current/metrics/region_rating_confusion"
DEFAULT_PREDICTIONS_DIR = "outputs/analysis/current/metrics/region_rating_predictions"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return slug or "authority"


def _lightgbm_params(config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    optimized_metrics = read_json("outputs/analysis/current/metrics/optimized_model_metrics.json", default={}) or {}
    best_params = optimized_metrics.get("lightgbm", {}).get("best_params", {})
    params = {
        "n_estimators": int(config.get("n_estimators", 300)),
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "num_leaves": 15,
        "min_child_samples": 10,
        "force_row_wise": True,
        "verbosity": -1,
        "random_state": seed,
    }
    params.update({key: value for key, value in best_params.items() if key in params or key in {"num_leaves"}})
    params.update(config.get("lightgbm_params", {}) or {})
    return params


def _make_model(params: Dict[str, Any]):
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:  # pragma: no cover - exercised only when optional dependency is missing.
        raise RuntimeError("LightGBM is required for the region rating prediction task.") from exc
    return LGBMClassifier(**params)


def _authority_distributions(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    distributions: Dict[str, Dict[str, int]] = {}
    for record in records:
        if record.get("status") != "success":
            continue
        for authority, rating in (record.get("result_region_ratings") or {}).items():
            authority_name = str(authority or "").strip()
            rating_text = str(rating or "").strip()
            if not authority_name or not rating_text:
                continue
            distributions.setdefault(authority_name, {})
            distributions[authority_name][rating_text] = distributions[authority_name].get(rating_text, 0) + 1
    return {
        authority: dict(sorted(distribution.items(), key=lambda item: (-item[1], item[0])))
        for authority, distribution in sorted(distributions.items())
    }


def _labels_for_authority(records: List[Dict[str, Any]], authority: str) -> List[str | None]:
    labels: List[str | None] = []
    for record in records:
        region_ratings = record.get("result_region_ratings") or {}
        label = str(region_ratings.get(authority) or "").strip()
        labels.append(label or None)
    return labels


def _ordered_labels(y: pd.Series) -> List[str]:
    counts = y.value_counts()
    return sorted(counts.index.tolist(), key=lambda label: (-int(counts[label]), str(label)))


def _evaluate_region_predictions(y_true: pd.Series, y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def _confusion_dataframe(metrics: Dict[str, Any]) -> pd.DataFrame:
    labels = metrics["labels"]
    return pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)


def _skip_reason(
    distribution: Dict[str, int],
    min_samples: int,
    min_classes: int,
    min_class_count: int,
) -> str | None:
    if sum(distribution.values()) < min_samples:
        return f"sample_count < {min_samples}"
    if len(distribution) < min_classes:
        return f"class_count < {min_classes}"
    if min(distribution.values()) < min_class_count:
        return f"min_class_count < {min_class_count}"
    return None


def train_authority_model(
    base_df: pd.DataFrame,
    records: List[Dict[str, Any]],
    authority: str,
    distribution: Dict[str, int],
    model_params: Dict[str, Any],
    seed: int,
    test_size: float,
    models_dir: Path,
    confusion_dir: Path,
    predictions_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    labels = _labels_for_authority(records, authority)
    df = base_df.copy()
    df["region_rating_label"] = labels
    df = df[df["region_rating_label"].notna()].copy()

    # The primary IARC label would leak target information for regional tasks, so keep only questionnaire features.
    df = df.drop(columns=["result_age_rating"], errors="ignore")
    X, y = build_feature_matrix(df, label_column="region_rating_label")
    label_order = _ordered_labels(y)
    stratify = y if y.value_counts().min() >= 2 and y.nunique() > 1 else None
    indices = pd.Series(range(len(y)), index=y.index)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    model = _make_model(model_params)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train, sample_weight=sample_weight)
    predictions = model.predict(X_test).tolist()
    metrics = _evaluate_region_predictions(y_test, predictions, label_order)

    slug = _slugify(authority)
    model_path = models_dir / f"region_rating_{slug}.joblib"
    confusion_path = confusion_dir / f"{slug}.csv"
    predictions_path = predictions_dir / f"{slug}.csv"

    joblib.dump(
        {
            "authority": authority,
            "model": model,
            "features": X.columns.tolist(),
            "labels": label_order,
            "class_distribution": distribution,
            "model_params": model_params,
        },
        model_path,
    )
    _confusion_dataframe(metrics).to_csv(confusion_path, encoding="utf-8-sig")

    predictions_df = pd.DataFrame(
        {
            "sample_id": base_df.loc[idx_test.index, "sample_id"].tolist(),
            "authority": authority,
            "true_label": y_test.tolist(),
            "predicted_label": predictions,
            "is_correct": [truth == pred for truth, pred in zip(y_test.tolist(), predictions)],
        }
    )
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        probability_labels = list(getattr(model, "classes_", label_order))
        for index, label in enumerate(probability_labels):
            predictions_df[f"proba__{label}"] = probabilities[:, index]
        predictions_df["confidence"] = probabilities.max(axis=1)
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    result = {
        **metrics,
        "authority": authority,
        "authority_slug": slug,
        "class_distribution": distribution,
        "test_distribution": y_test.value_counts().to_dict(),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "sample_count": int(len(y)),
        "class_count": int(y.nunique()),
        "min_class_count": int(y.value_counts().min()),
        "feature_count": int(X.shape[1]),
        "model_path": str(model_path),
        "confusion_matrix_path": str(confusion_path),
        "predictions_path": str(predictions_path),
    }
    summary_row = {
        "authority": authority,
        "trained": True,
        "skip_reason": "",
        "sample_count": result["sample_count"],
        "class_count": result["class_count"],
        "min_class_count": result["min_class_count"],
        "feature_count": result["feature_count"],
        "train_size": result["train_size"],
        "test_size": result["test_size"],
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "weighted_f1": result["weighted_f1"],
        "balanced_accuracy": result["balanced_accuracy"],
    }
    return result, summary_row


def run_region_rating_training(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_yaml(args.config)
    modeling = config.get("modeling", {})
    seed = int(args.seed if args.seed is not None else modeling.get("random_seed", 42))
    test_size = float(args.test_size if args.test_size is not None else modeling.get("test_size", 0.15))
    min_samples = int(args.min_samples)
    min_classes = int(args.min_classes)
    min_class_count = int(args.min_class_count)

    records = [record for record in JsonlStore(args.input).read_all() if record.get("status") == "success"]
    base_df = records_to_dataframe(records)
    distributions = _authority_distributions(records)
    requested_authorities = args.authorities or sorted(distributions)

    models_dir = ensure_dir(modeling.get("models_dir", "outputs/analysis/current/models"))
    confusion_dir = ensure_dir(args.confusion_dir)
    predictions_dir = ensure_dir(args.predictions_dir)
    model_params = _lightgbm_params(vars(args), seed)

    report: Dict[str, Any] = {
        "metadata": {
            "input_path": str(project_path(args.input)),
            "model": "LightGBM",
            "random_seed": seed,
            "test_size": test_size,
            "min_samples": min_samples,
            "min_classes": min_classes,
            "min_class_count": min_class_count,
            "trained_authorities": [],
            "skipped_authorities": [],
            "model_params": model_params,
        },
        "authorities": {},
    }
    summary_rows: List[Dict[str, Any]] = []

    for authority in requested_authorities:
        distribution = distributions.get(authority, {})
        skip_reason = _skip_reason(distribution, min_samples, min_classes, min_class_count)
        if skip_reason:
            row = {
                "authority": authority,
                "trained": False,
                "skip_reason": skip_reason,
                "sample_count": sum(distribution.values()),
                "class_count": len(distribution),
                "min_class_count": min(distribution.values()) if distribution else 0,
                "feature_count": 0,
                "train_size": 0,
                "test_size": 0,
                "accuracy": None,
                "macro_f1": None,
                "weighted_f1": None,
                "balanced_accuracy": None,
            }
            summary_rows.append(row)
            report["metadata"]["skipped_authorities"].append(authority)
            report["authorities"][authority] = {
                "trained": False,
                "skip_reason": skip_reason,
                "class_distribution": distribution,
            }
            continue

        result, row = train_authority_model(
            base_df=base_df,
            records=records,
            authority=authority,
            distribution=distribution,
            model_params=model_params,
            seed=seed,
            test_size=test_size,
            models_dir=models_dir,
            confusion_dir=confusion_dir,
            predictions_dir=predictions_dir,
        )
        summary_rows.append(row)
        report["metadata"]["trained_authorities"].append(authority)
        report["authorities"][authority] = {"trained": True, **result}

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["trained", "macro_f1", "sample_count"],
        ascending=[False, False, False],
        na_position="last",
    )
    summary_df.to_csv(args.summary_output, index=False, encoding="utf-8-sig")
    write_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-authority regional rating prediction models.")
    parser.add_argument("--config", default="configs/modeling.yaml")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_METRICS_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--confusion-dir", default=DEFAULT_CONFUSION_DIR)
    parser.add_argument("--predictions-dir", default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--authorities", nargs="*", default=None, help="Optional authority allow-list.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--min-classes", type=int, default=2)
    parser.add_argument("--min-class-count", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    report = run_region_rating_training(parse_args())
    trained = report["metadata"]["trained_authorities"]
    skipped = report["metadata"]["skipped_authorities"]
    print(f"trained_authorities={len(trained)} skipped_authorities={len(skipped)}")
    for authority in trained:
        metrics = report["authorities"][authority]
        print(
            f"{authority}: accuracy={metrics['accuracy']:.3f} "
            f"macro_f1={metrics['macro_f1']:.3f} balanced_accuracy={metrics['balanced_accuracy']:.3f}"
        )
    if skipped:
        print("skipped=" + "; ".join(skipped))


if __name__ == "__main__":
    main()
