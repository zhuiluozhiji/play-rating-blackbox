#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from src.common import ensure_dir, load_yaml, write_json
from src.modeling.evaluate import evaluate_predictions, mean_absolute_age_error, severe_error_rate
from src.modeling.features import build_feature_matrix


def _parameter_grid(grid: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def _split_fit_weight(params: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    model_params = dict(params)
    fit_weight = str(model_params.pop("fit_weight", "balanced"))
    return model_params, fit_weight


def _make_model(name: str, params: Dict[str, Any], seed: int, label_count: int):
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed, **params),
        )
    if name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier(random_state=seed, **params)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1, **params)
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=-1, **params)
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=200,
            objective="multi:softprob",
            num_class=label_count,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
            **params,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
            force_row_wise=True,
            verbosity=-1,
            **params,
        )
    raise ValueError(f"Unsupported optimized model: {name}")


def _fit_predict(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    sample_weight: np.ndarray,
    labels: List[str],
):
    if name == "xgboost":
        label_to_id = {label: index for index, label in enumerate(labels)}
        id_to_label = {index: label for label, index in label_to_id.items()}
        y_encoded = y_train.map(label_to_id)
        if sample_weight is None:
            model.fit(X_train, y_encoded)
        else:
            model.fit(X_train, y_encoded, sample_weight=sample_weight)
        return pd.Series([id_to_label[int(value)] for value in model.predict(X_eval)], index=X_eval.index)
    if name == "logistic_regression":
        if sample_weight is None:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, logisticregression__sample_weight=sample_weight)
        return pd.Series(model.predict(X_eval), index=X_eval.index)
    if sample_weight is None:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    return pd.Series(model.predict(X_eval), index=X_eval.index)


def _score_predictions(y_true: pd.Series, y_pred: pd.Series, age_order: List[str], severe_threshold: int) -> Dict[str, float]:
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mean_absolute_age_error": mean_absolute_age_error(y_true, y_pred, age_order),
        "severe_error_rate": severe_error_rate(y_true, y_pred, age_order, severe_threshold),
    }


def _error_rows(
    df: pd.DataFrame,
    test_index: np.ndarray,
    y_true: pd.Series,
    y_pred: pd.Series,
    age_order: List[str],
    severe_threshold: int,
) -> pd.DataFrame:
    order = {label: index for index, label in enumerate(age_order)}
    rows = []
    for position, true_label, pred_label in zip(test_index, y_true, y_pred):
        if true_label == pred_label:
            continue
        source = df.iloc[int(position)]
        age_error = abs(order.get(true_label, -999) - order.get(pred_label, -999))
        answer_count = sum(
            pd.notna(source[col]) and str(source[col]) != ""
            for col in df.columns
            if col.startswith("answer__")
        )
        rows.append(
            {
                "sample_id": source.get("sample_id"),
                "strategy": source.get("strategy"),
                "true_label": true_label,
                "predicted_label": pred_label,
                "age_error": age_error,
                "severe_error": int(age_error >= severe_threshold),
                "visible_question_count": source.get("visible_question_count"),
                "skipped_question_count": source.get("skipped_question_count"),
                "content_descriptor_count": source.get("content_descriptor_count"),
                "interactive_element_count": source.get("interactive_element_count"),
                "active_answer_count": answer_count,
            }
        )
    return pd.DataFrame(rows)


def run_optimization(config: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    modeling = config.get("modeling", config)
    seed = int(modeling.get("random_seed", 42))
    test_size = float(modeling.get("test_size", 0.15))
    age_order = list(modeling.get("age_order", ["3+", "7+", "12+", "16+", "18+"]))
    severe_threshold = int(modeling.get("severe_error_threshold", 2))

    df = pd.read_csv(modeling["dataset_path"], keep_default_na=False, na_values=[""])
    X, y = build_feature_matrix(df, label_column=modeling.get("label_column", "result_age_rating"))
    labels = sorted(y.dropna().unique().tolist(), key=lambda value: age_order.index(value) if value in age_order else 999)

    indices = np.arange(len(y))
    train_index, test_index = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    grids = {
        "logistic_regression": (
            _parameter_grid({"C": [0.3, 1.0, 3.0], "class_weight": [None, "balanced"], "fit_weight": ["none"]})
            + _parameter_grid({"C": [0.3, 1.0, 3.0], "class_weight": [None], "fit_weight": ["balanced"]})
        ),
        "decision_tree": (
            _parameter_grid(
                {
                    "max_depth": [8, 12, None],
                    "min_samples_leaf": [1, 3],
                    "class_weight": [None, "balanced"],
                    "fit_weight": ["none"],
                }
            )
            + _parameter_grid(
                {
                    "max_depth": [8, 12, None],
                    "min_samples_leaf": [1, 3],
                    "class_weight": [None],
                    "fit_weight": ["balanced"],
                }
            )
        ),
        "random_forest": (
            _parameter_grid(
                {
                    "max_depth": [None, 12],
                    "min_samples_leaf": [1, 2, 3],
                    "max_features": ["sqrt"],
                    "class_weight": [None, "balanced"],
                    "fit_weight": ["none"],
                }
            )
            + _parameter_grid(
                {
                    "max_depth": [None],
                    "min_samples_leaf": [1, 3],
                    "max_features": ["sqrt"],
                    "class_weight": [None],
                    "fit_weight": ["balanced"],
                }
            )
        ),
        "extra_trees": (
            _parameter_grid(
                {
                    "max_depth": [None, 12],
                    "min_samples_leaf": [1, 2, 3],
                    "max_features": ["sqrt"],
                    "class_weight": [None, "balanced"],
                    "fit_weight": ["none"],
                }
            )
            + _parameter_grid(
                {
                    "max_depth": [None],
                    "min_samples_leaf": [1, 3],
                    "max_features": ["sqrt"],
                    "class_weight": [None],
                    "fit_weight": ["balanced"],
                }
            )
        ),
        "xgboost": _parameter_grid(
            {
                "max_depth": [3, 4],
                "learning_rate": [0.05],
                "subsample": [0.9],
                "colsample_bytree": [0.8, 1.0],
                "min_child_weight": [1],
            }
        ),
        "lightgbm": _parameter_grid(
            {
                "max_depth": [3, 5],
                "learning_rate": [0.05],
                "subsample": [0.9],
                "num_leaves": [15, 31],
                "min_child_samples": [10],
            }
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_rows: List[Dict[str, Any]] = []
    best_params: Dict[str, Dict[str, Any]] = {}

    for model_name, param_grid in grids.items():
        best_score = -1.0
        for params in param_grid:
            param_key = repr(sorted(params.items()))
            model_params, fit_weight = _split_fit_weight(params)
            fold_scores = []
            for fold, (fold_train_idx, fold_valid_idx) in enumerate(cv.split(X_train, y_train), start=1):
                model = _make_model(model_name, model_params, seed, len(labels))
                fold_X_train = X_train.iloc[fold_train_idx]
                fold_y_train = y_train.iloc[fold_train_idx]
                fold_X_valid = X_train.iloc[fold_valid_idx]
                fold_y_valid = y_train.iloc[fold_valid_idx]
                weights = (
                    compute_sample_weight(class_weight="balanced", y=fold_y_train)
                    if fit_weight == "balanced"
                    else None
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fold_pred = _fit_predict(model_name, model, fold_X_train, fold_y_train, fold_X_valid, weights, labels)
                fold_metrics = _score_predictions(fold_y_valid, fold_pred, age_order, severe_threshold)
                fold_scores.append(fold_metrics["macro_f1"])
                cv_rows.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "params": str(params),
                        "param_key": param_key,
                        **fold_metrics,
                    }
                )
            mean_score = float(np.mean(fold_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_params[model_name] = params

    metrics: Dict[str, Dict[str, Any]] = {}
    predictions: Dict[str, pd.Series] = {}
    models_dir = ensure_dir(modeling.get("models_dir", "outputs/analysis/current/models"))
    cv_df = pd.DataFrame(cv_rows)
    for model_name, params in best_params.items():
        param_key = repr(sorted(params.items()))
        model_params, fit_weight = _split_fit_weight(params)
        model = _make_model(model_name, model_params, seed, len(labels))
        weights = compute_sample_weight(class_weight="balanced", y=y_train) if fit_weight == "balanced" else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = _fit_predict(model_name, model, X_train, y_train, X_test, weights, labels)
        metrics[model_name] = evaluate_predictions(y_test, y_pred, labels, age_order, severe_threshold)
        metrics[model_name]["best_params"] = params
        cv_mask = (cv_df["model"] == model_name) & (cv_df["param_key"] == param_key)
        metrics[model_name]["cv_macro_f1_mean"] = float(cv_df.loc[cv_mask, "macro_f1"].mean())
        predictions[model_name] = y_pred
        joblib.dump(
            {"model": model, "features": X.columns.tolist(), "labels": labels, "best_params": params},
            models_dir / f"optimized_{model_name}.joblib",
        )

    best_model_name = max(metrics.items(), key=lambda item: item[1]["macro_f1"])[0]
    errors = _error_rows(df, test_index, y_test, predictions[best_model_name], age_order, severe_threshold)
    errors.insert(0, "model", best_model_name)
    return metrics, pd.DataFrame(cv_rows), errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Run weighted CV tuning for the strongest age-rating models.")
    parser.add_argument("--config", default="configs/modeling.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    modeling = config.get("modeling", {})
    metrics_dir = ensure_dir(modeling.get("metrics_dir", "outputs/analysis/current/metrics"))
    metrics, cv_results, errors = run_optimization(config)

    write_json(metrics_dir / "optimized_model_metrics.json", metrics)
    cv_results.to_csv(metrics_dir / "optimized_cv_results.csv", index=False)
    errors.to_csv(metrics_dir / "optimized_holdout_errors.csv", index=False)

    for model_name, values in sorted(metrics.items(), key=lambda item: item[1]["macro_f1"], reverse=True):
        print(
            f"{model_name}: accuracy={values['accuracy']:.3f} "
            f"macro_f1={values['macro_f1']:.3f} "
            f"balanced_accuracy={values['balanced_accuracy']:.3f} "
            f"severe_error={values['severe_error_rate']:.3f} "
            f"params={values['best_params']}"
        )
    print(f"wrote {metrics_dir / 'optimized_model_metrics.json'}")
    print(f"wrote {metrics_dir / 'optimized_cv_results.csv'}")
    print(f"wrote {metrics_dir / 'optimized_holdout_errors.csv'}")


if __name__ == "__main__":
    main()
