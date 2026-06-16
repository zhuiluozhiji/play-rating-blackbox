#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common import ensure_dir, load_yaml, read_json, write_json


def _param_key(params: Dict[str, Any]) -> str:
    return repr(sorted(params.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize optimized CV stability and holdout errors.")
    parser.add_argument("--config", default="configs/modeling.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    modeling = config.get("modeling", {})
    metrics_dir = ensure_dir(modeling.get("metrics_dir", "outputs/analysis/current/metrics"))

    optimized_metrics = read_json(metrics_dir / "optimized_model_metrics.json", default={})
    cv_results = pd.read_csv(metrics_dir / "optimized_cv_results.csv")
    errors_path = metrics_dir / "optimized_holdout_errors.csv"
    errors = pd.read_csv(errors_path) if errors_path.exists() else pd.DataFrame()

    stability_rows = []
    for model_name, holdout in sorted(optimized_metrics.items()):
        key = _param_key(holdout.get("best_params", {}))
        model_cv = cv_results[(cv_results["model"] == model_name) & (cv_results["param_key"] == key)]
        stability_rows.append(
            {
                "model": model_name,
                "best_params": str(holdout.get("best_params", {})),
                "cv_macro_f1_mean": model_cv["macro_f1"].mean(),
                "cv_macro_f1_std": model_cv["macro_f1"].std(ddof=0),
                "cv_severe_error_mean": model_cv["severe_error_rate"].mean(),
                "cv_severe_error_std": model_cv["severe_error_rate"].std(ddof=0),
                "holdout_accuracy": holdout.get("accuracy"),
                "holdout_macro_f1": holdout.get("macro_f1"),
                "holdout_balanced_accuracy": holdout.get("balanced_accuracy"),
                "holdout_severe_error_rate": holdout.get("severe_error_rate"),
            }
        )
    stability = pd.DataFrame(stability_rows).sort_values("holdout_macro_f1", ascending=False)
    stability.to_csv(metrics_dir / "optimized_cv_stability_summary.csv", index=False)

    if errors.empty:
        error_summary = {
            "total_errors": 0,
            "severe_errors": 0,
            "by_transition": [],
            "by_true_label": [],
            "by_predicted_label": [],
        }
        pd.DataFrame().to_csv(metrics_dir / "optimized_error_transitions.csv", index=False)
        pd.DataFrame().to_csv(metrics_dir / "optimized_16plus_errors.csv", index=False)
    else:
        transitions = (
            errors.groupby(["true_label", "predicted_label"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["true_label", "predicted_label"])
        )
        transitions.to_csv(metrics_dir / "optimized_error_transitions.csv", index=False)
        focus_16 = errors[(errors["true_label"] == "16+") | (errors["predicted_label"] == "16+")]
        focus_16.to_csv(metrics_dir / "optimized_16plus_errors.csv", index=False)
        error_summary = {
            "total_errors": int(len(errors)),
            "severe_errors": int(errors["severe_error"].astype(int).sum()),
            "by_transition": transitions.to_dict(orient="records"),
            "by_true_label": errors.groupby("true_label").size().reset_index(name="count").to_dict(orient="records"),
            "by_predicted_label": errors.groupby("predicted_label").size().reset_index(name="count").to_dict(orient="records"),
            "sixteen_plus_related_errors": int(len(focus_16)),
        }
    write_json(metrics_dir / "optimized_error_analysis_summary.json", error_summary)

    print(stability[["model", "cv_macro_f1_mean", "cv_macro_f1_std", "holdout_macro_f1", "holdout_severe_error_rate"]].to_string(index=False))
    print(f"wrote {metrics_dir / 'optimized_cv_stability_summary.csv'}")
    print(f"wrote {metrics_dir / 'optimized_error_analysis_summary.json'}")
    print(f"wrote {metrics_dir / 'optimized_error_transitions.csv'}")
    print(f"wrote {metrics_dir / 'optimized_16plus_errors.csv'}")


if __name__ == "__main__":
    main()
