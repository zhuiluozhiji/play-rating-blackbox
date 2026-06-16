from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text

from src.common import ensure_dir

MAX_PERMUTATION_SAMPLES = 300
PERMUTATION_REPEATS = 2


def _sample_for_permutation_importance(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= MAX_PERMUTATION_SAMPLES:
        return X, y
    stratify = y if y.value_counts().min() >= 2 and y.nunique() > 1 else None
    X_sampled, _, y_sampled, _ = train_test_split(
        X,
        y,
        train_size=MAX_PERMUTATION_SAMPLES,
        random_state=42,
        stratify=stratify,
    )
    return X_sampled, y_sampled


def explain_saved_models(models_dir: str, X: pd.DataFrame, y: pd.Series, output_dir: str) -> Dict[str, str]:
    output = ensure_dir(output_dir)
    written: Dict[str, str] = {}
    X_permutation, y_permutation = _sample_for_permutation_importance(X, y)
    for model_path in sorted(Path(models_dir).glob("*.joblib")):
        payload = joblib.load(model_path)
        model = payload["model"]
        name = model_path.stem
        feature_names = payload.get("features", X.columns.tolist())
        lines: List[str] = [f"# {name}\n"]
        if hasattr(model, "feature_importances_"):
            importances = pd.DataFrame(
                {"feature": feature_names, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)
            path = output / f"{name}_feature_importance.csv"
            importances.to_csv(path, index=False)
            written[f"{name}_feature_importance"] = str(path)
            lines.append(importances.head(30).to_string(index=False))
        if name == "decision_tree":
            tree_text = export_text(model, feature_names=list(feature_names), max_depth=5)
            path = output / "decision_tree_rules.txt"
            path.write_text(tree_text, encoding="utf-8")
            written["decision_tree_rules"] = str(path)
        if name in {"majority", "stratified"}:
            summary_path = output / f"{name}_explanation.md"
            summary_path.write_text("\n\n".join(lines), encoding="utf-8")
            written[f"{name}_summary"] = str(summary_path)
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                permutation = permutation_importance(
                    model,
                    X_permutation,
                    y_permutation,
                    n_repeats=PERMUTATION_REPEATS,
                    random_state=42,
                    n_jobs=1,
                )
            perm_df = pd.DataFrame(
                {"feature": X_permutation.columns, "importance_mean": permutation.importances_mean}
            ).sort_values("importance_mean", ascending=False)
            path = output / f"{name}_permutation_importance.csv"
            perm_df.to_csv(path, index=False)
            written[f"{name}_permutation_importance"] = str(path)
        except Exception:
            pass
        summary_path = output / f"{name}_explanation.md"
        summary_path.write_text("\n\n".join(lines), encoding="utf-8")
        written[f"{name}_summary"] = str(summary_path)
    return written
