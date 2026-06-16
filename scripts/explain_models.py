#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common import load_yaml
from src.modeling.explain import explain_saved_models
from src.modeling.features import build_feature_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model explanations.")
    parser.add_argument("--config", default="configs/modeling.yaml")
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    modeling = config.get("modeling", {})
    dataset_path = args.dataset or modeling.get("dataset_path", "data/processed/dataset.csv")
    df = pd.read_csv(dataset_path)
    X, y = build_feature_matrix(df, label_column=modeling.get("label_column", "result_age_rating"))
    written = explain_saved_models(
        models_dir=modeling.get("models_dir", "outputs/analysis/current/models"),
        X=X,
        y=y,
        output_dir=modeling.get("explanations_dir", "outputs/analysis/current/explanations"),
    )
    for name, path in written.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
