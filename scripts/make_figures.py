#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common import load_yaml, project_path
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_label_distribution,
    plot_model_metrics,
    plot_top_feature_importance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report figures.")
    parser.add_argument("--config", default="configs/modeling.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--metrics", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    modeling = config.get("modeling", {})
    figures_dir = modeling.get("figures_dir", "outputs/analysis/current/figures")
    dataset_path = args.dataset or modeling.get("dataset_path", "data/processed/dataset.csv")
    metrics_path = project_path(args.metrics or "outputs/analysis/current/metrics/model_metrics.json")
    df = pd.read_csv(dataset_path, keep_default_na=False, na_values=[""])
    paths = [plot_label_distribution(df, figures_dir)]
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        paths.append(plot_model_metrics(metrics, figures_dir))
        best_name = max(metrics.items(), key=lambda item: item[1].get("macro_f1", 0))[0]
        paths.append(plot_confusion_matrix(metrics, best_name, figures_dir))
    paths.extend(
        plot_top_feature_importance(
            modeling.get("explanations_dir", "outputs/analysis/current/explanations"),
            figures_dir,
        )
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
