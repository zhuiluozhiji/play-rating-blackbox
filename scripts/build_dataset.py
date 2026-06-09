#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ensure_parent
from src.data.storage import JsonlStore
from src.modeling.features import build_feature_matrix, records_to_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed dataset and feature matrix.")
    parser.add_argument("--input", default="data/raw/samples.jsonl")
    parser.add_argument("--dataset-output", default="data/processed/dataset.csv")
    parser.add_argument("--features-output", default="data/processed/features.csv")
    args = parser.parse_args()

    records = [
        row for row in JsonlStore(args.input).read_all()
        if row.get("status") == "success" and row.get("result_age_rating")
    ]
    df = records_to_dataframe(records)
    dataset_path = ensure_parent(args.dataset_output)
    features_path = ensure_parent(args.features_output)
    df.to_csv(dataset_path, index=False)
    X, y = build_feature_matrix(df)
    feature_df = X.copy()
    feature_df["result_age_rating"] = y.values
    feature_df.to_csv(features_path, index=False)
    print(f"wrote dataset={dataset_path} rows={len(df)}")
    print(f"wrote features={features_path} shape={feature_df.shape}")


if __name__ == "__main__":
    main()
