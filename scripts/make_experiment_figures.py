#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.common import ensure_dir, load_yaml, project_path


ORDER = ["3+", "7+", "12+", "16+", "18+"]


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _metric_rows(metrics: Dict) -> pd.DataFrame:
    rows = []
    for model, values in metrics.items():
        rows.append(
            {
                "model": model.replace("optimized_", ""),
                "accuracy": values.get("accuracy", 0),
                "macro_f1": values.get("macro_f1", 0),
                "balanced_accuracy": values.get("balanced_accuracy", 0),
                "weighted_f1": values.get("weighted_f1", 0),
                "severe_error_rate": values.get("severe_error_rate", 0),
            }
        )
    return pd.DataFrame(rows)


def _save(path: Path) -> Path:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_base_vs_optimized(base_metrics: Dict, optimized_metrics: Dict, output_dir: Path) -> Path | None:
    base = _metric_rows(base_metrics)
    opt = _metric_rows(optimized_metrics)
    if base.empty or opt.empty:
        return None
    merged = base[["model", "macro_f1"]].rename(columns={"macro_f1": "base_macro_f1"}).merge(
        opt[["model", "macro_f1"]].rename(columns={"macro_f1": "optimized_macro_f1"}),
        on="model",
        how="inner",
    )
    if merged.empty:
        return None
    data = merged.melt(id_vars="model", var_name="stage", value_name="macro_f1")
    data["stage"] = data["stage"].map({"base_macro_f1": "Base", "optimized_macro_f1": "Optimized"})
    plt.figure(figsize=(10, 5))
    sns.barplot(data=data, x="model", y="macro_f1", hue="stage")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.title("Base vs Optimized Macro-F1")
    return _save(output_dir / "base_vs_optimized_macro_f1.png")


def plot_optimized_performance(optimized_metrics: Dict, output_dir: Path) -> List[Path]:
    data = _metric_rows(optimized_metrics)
    if data.empty:
        return []
    data = data.sort_values("macro_f1", ascending=False)
    perf = data.melt(
        id_vars="model",
        value_vars=["accuracy", "macro_f1", "balanced_accuracy"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=perf, x="model", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.title("Optimized Model Performance")
    paths = [_save(output_dir / "optimized_model_performance.png")]

    plt.figure(figsize=(8, 4))
    sns.barplot(data=data, x="model", y="severe_error_rate", color="#d95f02")
    plt.xticks(rotation=25, ha="right")
    plt.title("Optimized Model Severe Error Rate")
    paths.append(_save(output_dir / "optimized_severe_error_rate.png"))
    return paths


def plot_feature_ablation(metrics_dir: Path, output_dir: Path) -> Path | None:
    path = metrics_dir / "feature_ablation_summary.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path).sort_values("macro_f1", ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=data, x="subset", y="macro_f1", color="#4c78a8")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.title("Feature Ablation Macro-F1")
    for index, row in enumerate(data.itertuples()):
        plt.text(index, row.macro_f1 + 0.015, f"{int(row.feature_count)}f", ha="center", fontsize=8)
    return _save(output_dir / "feature_ablation_macro_f1.png")


def plot_cv_stability(metrics_dir: Path, output_dir: Path) -> Path | None:
    path = metrics_dir / "optimized_cv_stability_summary.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path).sort_values("holdout_macro_f1", ascending=False)
    x = range(len(data))
    plt.figure(figsize=(10, 5))
    plt.errorbar(
        x,
        data["cv_macro_f1_mean"],
        yerr=data["cv_macro_f1_std"],
        fmt="o",
        capsize=5,
        label="5-fold CV macro-F1",
    )
    plt.scatter(x, data["holdout_macro_f1"], marker="s", color="#d95f02", label="Holdout macro-F1")
    plt.xticks(list(x), data["model"], rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("macro-F1")
    plt.title("CV Stability vs Holdout Performance")
    plt.legend()
    return _save(output_dir / "cv_stability_macro_f1.png")


def plot_error_transitions(metrics_dir: Path, output_dir: Path) -> Path | None:
    path = metrics_dir / "optimized_error_transitions.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path)
    if data.empty:
        return None
    matrix = data.pivot_table(index="true_label", columns="predicted_label", values="count", fill_value=0)
    matrix = matrix.reindex(index=ORDER, columns=ORDER, fill_value=0)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="g", cmap="Reds")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Optimized Model Error Transitions")
    return _save(output_dir / "optimized_error_transitions.png")


def plot_region_summary(metrics_dir: Path, output_dir: Path) -> List[Path]:
    report = _read_json(metrics_dir / "region_rating_summary.json")
    if not report:
        return []
    paths: List[Path] = []
    gp = report.get("google_play_vs_iarc_generic", {})
    match_rows = pd.DataFrame(
        [
            {"status": "Exact match", "count": gp.get("exact_match_count", 0)},
            {"status": "Mismatch", "count": gp.get("mismatch_count", 0)},
        ]
    )
    plt.figure(figsize=(6, 4))
    sns.barplot(data=match_rows, x="status", y="count", hue="status", legend=False)
    plt.title("Google Play vs IARC Generic")
    paths.append(_save(output_dir / "google_play_vs_iarc_match.png"))

    presence = report.get("authority_presence_counts", {})
    if presence:
        data = pd.DataFrame([{"authority": key, "count": value} for key, value in presence.items()]).sort_values(
            "count", ascending=True
        )
        plt.figure(figsize=(9, 5))
        sns.barplot(data=data, y="authority", x="count", color="#59a14f")
        plt.title("Rating Authority Presence")
        paths.append(_save(output_dir / "authority_presence_counts.png"))
    return paths


def plot_region_prediction_performance(metrics_dir: Path, output_dir: Path) -> Path | None:
    path = metrics_dir / "region_rating_model_summary.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path)
    if data.empty or "trained" not in data.columns:
        return None
    data = data[data["trained"].astype(str).str.lower().isin(["true", "1"])]
    if data.empty:
        return None

    data = data.sort_values("macro_f1", ascending=False)
    plot_data = data.melt(
        id_vars="authority",
        value_vars=["accuracy", "macro_f1", "balanced_accuracy"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_data, x="authority", y="score", hue="metric")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right")
    plt.title("Regional Rating Prediction Performance")
    return _save(output_dir / "region_rating_prediction_performance.png")


def plot_advanced_results(metrics_dir: Path, output_dir: Path) -> List[Path]:
    paths: List[Path] = []

    top2_path = metrics_dir / "confidence_bins.csv"
    if top2_path.exists():
        data = pd.read_csv(top2_path)
        if not data.empty:
            plt.figure(figsize=(9, 5))
            x = range(len(data))
            plt.plot(x, data["avg_confidence"], marker="o", label="Avg confidence")
            plt.plot(x, data["top1_accuracy"], marker="s", label="Top-1 accuracy")
            plt.plot(x, data["top2_accuracy"], marker="^", label="Top-2 accuracy")
            plt.xticks(list(x), data["confidence_bin"], rotation=25, ha="right")
            plt.ylim(0, 1.05)
            plt.ylabel("score")
            plt.title("Confidence Calibration by Bin")
            plt.legend()
            paths.append(_save(output_dir / "top2_confidence_bins.png"))

    ordinal_path = metrics_dir / "ordinal_model_metrics.json"
    optimized_path = metrics_dir / "optimized_model_metrics.json"
    if ordinal_path.exists() and optimized_path.exists():
        ordinal = _read_json(ordinal_path)
        optimized = _read_json(optimized_path).get("lightgbm", {})
        rows = [
            {
                "model": "optimized_lightgbm",
                "accuracy": optimized.get("accuracy", 0),
                "macro_f1": optimized.get("macro_f1", 0),
                "balanced_accuracy": optimized.get("balanced_accuracy", 0),
                "severe_error_rate": optimized.get("severe_error_rate", 0),
            },
            {
                "model": "ordinal_lightgbm",
                "accuracy": ordinal.get("accuracy", 0),
                "macro_f1": ordinal.get("macro_f1", 0),
                "balanced_accuracy": ordinal.get("balanced_accuracy", 0),
                "severe_error_rate": ordinal.get("severe_error_rate", 0),
            },
        ]
        data = pd.DataFrame(rows).melt(id_vars="model", var_name="metric", value_name="score")
        plt.figure(figsize=(9, 5))
        sns.barplot(data=data, x="model", y="score", hue="metric")
        plt.ylim(0, 1.05)
        plt.title("Multiclass vs Ordinal LightGBM")
        paths.append(_save(output_dir / "ordinal_vs_multiclass_metrics.png"))

    counterfactual_path = metrics_dir / "counterfactual_summary.json"
    if counterfactual_path.exists():
        report = _read_json(counterfactual_path)
        transitions = report.get("prediction_transitions", {})
        if transitions:
            data = pd.DataFrame(
                [{"transition": transition, "count": count} for transition, count in transitions.items()]
            ).sort_values("count", ascending=False).head(12)
            plt.figure(figsize=(10, 5))
            sns.barplot(data=data, x="transition", y="count", color="#e15759")
            plt.xticks(rotation=25, ha="right")
            plt.title("Counterfactual Feature Flip Prediction Changes")
            paths.append(_save(output_dir / "counterfactual_prediction_transitions.png"))

    return paths


def plot_label_distribution(dataset_path: Path, output_dir: Path) -> Path | None:
    if not dataset_path.exists():
        return None
    df = pd.read_csv(dataset_path, keep_default_na=False, na_values=[""])
    counts = df["result_age_rating"].value_counts().reindex(ORDER).fillna(0).reset_index()
    counts.columns = ["rating", "count"]
    plt.figure(figsize=(7, 4))
    sns.barplot(data=counts, x="rating", y="count", color="#4c78a8")
    plt.title("Age Rating Label Distribution")
    return _save(output_dir / "experiment_label_distribution.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures for completed and supplementary experiments.")
    parser.add_argument("--config", default="configs/modeling.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    modeling = config.get("modeling", {})
    metrics_dir = project_path(modeling.get("metrics_dir", "outputs/analysis/current/metrics"))
    output_dir = ensure_dir(modeling.get("figures_dir", "outputs/analysis/current/figures"))
    dataset_path = project_path(modeling.get("dataset_path", "data/processed/dataset.csv"))

    written: List[Path] = []
    base_metrics = _read_json(metrics_dir / "model_metrics.json")
    optimized_metrics = _read_json(metrics_dir / "optimized_model_metrics.json")
    for maybe_path in [
        plot_label_distribution(dataset_path, output_dir),
        plot_base_vs_optimized(base_metrics, optimized_metrics, output_dir),
        plot_feature_ablation(metrics_dir, output_dir),
        plot_cv_stability(metrics_dir, output_dir),
        plot_error_transitions(metrics_dir, output_dir),
        plot_region_prediction_performance(metrics_dir, output_dir),
    ]:
        if maybe_path:
            written.append(maybe_path)
    written.extend(plot_optimized_performance(optimized_metrics, output_dir))
    written.extend(plot_region_summary(metrics_dir, output_dir))
    written.extend(plot_advanced_results(metrics_dir, output_dir))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
