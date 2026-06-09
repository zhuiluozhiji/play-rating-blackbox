from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.common import ensure_dir


def plot_label_distribution(df: pd.DataFrame, output_dir: str) -> Path:
    out = ensure_dir(output_dir)
    path = out / "label_distribution.png"
    plt.figure(figsize=(8, 5))
    order = ["3+", "7+", "12+", "16+", "18+"]
    counts = df["result_age_rating"].value_counts().reindex(order).dropna()
    sns.barplot(x=counts.index, y=counts.values, color="#4c78a8")
    plt.xlabel("Age rating")
    plt.ylabel("Samples")
    plt.title("Age Rating Distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_model_metrics(metrics: Dict, output_dir: str) -> Path:
    out = ensure_dir(output_dir)
    path = out / "model_performance.png"
    rows = [
        {
            "model": model,
            "macro_f1": values.get("macro_f1", 0),
            "accuracy": values.get("accuracy", 0),
            "balanced_accuracy": values.get("balanced_accuracy", 0),
        }
        for model, values in metrics.items()
    ]
    data = pd.DataFrame(rows)
    melted = data.melt(id_vars="model", var_name="metric", value_name="score")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=melted, x="model", y="score", hue="metric")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.title("Model Performance")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_confusion_matrix(metrics: Dict, model_name: str, output_dir: str) -> Path:
    out = ensure_dir(output_dir)
    path = out / f"confusion_matrix_{model_name}.png"
    values = metrics[model_name]
    labels = values["labels"]
    matrix = values["confusion_matrix"]
    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_top_feature_importance(explanations_dir: str, output_dir: str) -> list[Path]:
    out = ensure_dir(output_dir)
    written: list[Path] = []
    for csv_path in Path(explanations_dir).glob("*_feature_importance.csv"):
        df = pd.read_csv(csv_path).head(20)
        if df.empty:
            continue
        path = out / f"{csv_path.stem}.png"
        plt.figure(figsize=(9, 7))
        sns.barplot(data=df, y="feature", x="importance", color="#59a14f")
        plt.title(csv_path.stem.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        written.append(path)
    return written
