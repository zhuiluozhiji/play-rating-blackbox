#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from src.common import ensure_dir, load_yaml, read_json, write_json
from src.modeling.evaluate import evaluate_predictions
from src.modeling.features import build_feature_matrix


def _feature_subsets(columns: List[str]) -> Dict[str, List[str]]:
    answer_cols = [col for col in columns if col.startswith("answer__")]
    strategy_cols = [col for col in columns if col.startswith("strategy_")]
    descriptor_cols = ["content_descriptor_count", "interactive_element_count"]
    aggregate_cols = [
        col
        for col in columns
        if col.endswith("_score") or col in {"high_risk_count", "medium_risk_count", "triggered_branch_count"}
    ]
    count_cols = [
        col
        for col in columns
        if col.endswith("_count") and not col.startswith("answer__")
    ]
    metadata_cols = sorted(set(count_cols + aggregate_cols + descriptor_cols) & set(columns))
    return {
        "full": columns,
        "answer_only": answer_cols,
        "counts_and_scores_only": metadata_cols,
        "no_strategy": [col for col in columns if col not in strategy_cols],
        "no_descriptor_interactive": [col for col in columns if col not in descriptor_cols],
        "no_aggregate_scores": [col for col in columns if col not in aggregate_cols],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature ablation experiments with the current best model.")
    parser.add_argument("--config", default="configs/modeling.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    modeling = config.get("modeling", {})
    seed = int(modeling.get("random_seed", 42))
    test_size = float(modeling.get("test_size", 0.15))
    age_order = list(modeling.get("age_order", ["3+", "7+", "12+", "16+", "18+"]))
    severe_threshold = int(modeling.get("severe_error_threshold", 2))
    metrics_dir = ensure_dir(modeling.get("metrics_dir", "outputs/analysis/current/metrics"))

    df = pd.read_csv(modeling["dataset_path"], keep_default_na=False, na_values=[""])
    X, y = build_feature_matrix(df, label_column=modeling.get("label_column", "result_age_rating"))
    labels = sorted(y.dropna().unique().tolist(), key=lambda value: age_order.index(value) if value in age_order else 999)

    optimized_metrics = read_json(metrics_dir / "optimized_model_metrics.json", default={})
    lightgbm_params = optimized_metrics.get("lightgbm", {}).get(
        "best_params",
        {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.9, "num_leaves": 15, "min_child_samples": 10},
    )

    train_idx, test_idx = train_test_split(
        range(len(y)),
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    X_train, X_test = X.iloc[list(train_idx)], X.iloc[list(test_idx)]
    y_train, y_test = y.iloc[list(train_idx)], y.iloc[list(test_idx)]
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    results: Dict[str, Dict] = {}
    for subset_name, subset_cols in _feature_subsets(X.columns.tolist()).items():
        if not subset_cols:
            continue
        model = LGBMClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
            force_row_wise=True,
            verbosity=-1,
            **lightgbm_params,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train[subset_cols], y_train, sample_weight=sample_weight)
        predictions = model.predict(X_test[subset_cols])
        metrics = evaluate_predictions(y_test, predictions, labels, age_order, severe_threshold)
        metrics["feature_count"] = len(subset_cols)
        metrics["model"] = "lightgbm"
        metrics["params"] = lightgbm_params
        results[subset_name] = metrics

    write_json(metrics_dir / "feature_ablation_metrics.json", results)
    rows = [
        {
            "subset": subset,
            "feature_count": values["feature_count"],
            "accuracy": values["accuracy"],
            "macro_f1": values["macro_f1"],
            "balanced_accuracy": values["balanced_accuracy"],
            "weighted_f1": values["weighted_f1"],
            "mean_absolute_age_error": values["mean_absolute_age_error"],
            "severe_error_rate": values["severe_error_rate"],
        }
        for subset, values in sorted(results.items(), key=lambda item: item[1]["macro_f1"], reverse=True)
    ]
    pd.DataFrame(rows).to_csv(metrics_dir / "feature_ablation_summary.csv", index=False)

    for row in rows:
        print(
            f"{row['subset']}: features={row['feature_count']} "
            f"accuracy={row['accuracy']:.3f} macro_f1={row['macro_f1']:.3f} "
            f"balanced_accuracy={row['balanced_accuracy']:.3f} severe_error={row['severe_error_rate']:.3f}"
        )
    print(f"wrote {metrics_dir / 'feature_ablation_metrics.json'}")
    print(f"wrote {metrics_dir / 'feature_ablation_summary.csv'}")


if __name__ == "__main__":
    main()
