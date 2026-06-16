#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common import load_yaml
from src.modeling.features import build_feature_matrix
from src.modeling.train import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rating prediction models.")
    parser.add_argument("--config", default="configs/modeling.yaml")
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    modeling = config.get("modeling", {})
    dataset_path = args.dataset or modeling.get("dataset_path", "data/processed/dataset.csv")
    df = pd.read_csv(dataset_path, keep_default_na=False, na_values=[""])
    X, y = build_feature_matrix(df, label_column=modeling.get("label_column", "result_age_rating"))
    metrics = train_and_evaluate(X, y, config)
    for model_name, values in metrics.items():
        print(f"{model_name}: accuracy={values['accuracy']:.3f} macro_f1={values['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
