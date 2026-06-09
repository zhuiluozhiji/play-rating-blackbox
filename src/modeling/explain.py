from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text

from src.common import ensure_dir


def explain_saved_models(models_dir: str, X: pd.DataFrame, y: pd.Series, output_dir: str) -> Dict[str, str]:
    output = ensure_dir(output_dir)
    written: Dict[str, str] = {}
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
                permutation = permutation_importance(model, X, y, n_repeats=3, random_state=42, n_jobs=1)
            perm_df = pd.DataFrame(
                {"feature": X.columns, "importance_mean": permutation.importances_mean}
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
