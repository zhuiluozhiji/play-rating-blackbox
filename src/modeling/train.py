from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.common import ensure_dir, write_json
from src.modeling.evaluate import evaluate_predictions


def _optional_xgboost(seed: int):
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        eval_metric="mlogloss",
        random_state=seed,
    )


def _optional_lightgbm(seed: int):
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return None
    return LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        random_state=seed,
    )


def make_models(model_names: List[str], seed: int) -> Dict[str, object]:
    registry = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "stratified": DummyClassifier(strategy="stratified", random_state=seed),
        "logistic_regression": make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
            ),
        ),
        "decision_tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=3, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }
    xgb = _optional_xgboost(seed)
    if xgb is not None:
        registry["xgboost"] = xgb
    lgbm = _optional_lightgbm(seed)
    if lgbm is not None:
        registry["lightgbm"] = lgbm
    return {name: registry[name] for name in model_names if name in registry}


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict,
) -> Dict[str, Dict]:
    modeling = config.get("modeling", config)
    seed = int(modeling.get("random_seed", 42))
    test_size = float(modeling.get("test_size", 0.15))
    models_dir = ensure_dir(modeling.get("models_dir", "outputs/analysis/current/models"))
    metrics_dir = ensure_dir(modeling.get("metrics_dir", "outputs/analysis/current/metrics"))
    age_order = list(modeling.get("age_order", ["3+", "7+", "12+", "16+", "18+"]))
    labels = sorted(y.dropna().unique().tolist(), key=lambda value: age_order.index(value) if value in age_order else 999)

    stratify = y if y.value_counts().min() >= 2 and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    model_names = list(modeling.get("models", ["majority", "logistic_regression", "decision_tree", "random_forest"]))
    models = make_models(model_names, seed)
    all_metrics: Dict[str, Dict] = {}
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
            except ValueError as exc:
                if name != "xgboost":
                    raise
                encoded = {label: index for index, label in enumerate(labels)}
                decoded = {index: label for label, index in encoded.items()}
                y_train_encoded = y_train.map(encoded)
                model.fit(X_train, y_train_encoded)
                predictions = [decoded[int(value)] for value in model.predict(X_test)]
            metrics = evaluate_predictions(
                y_test,
                predictions,
                labels=labels,
                age_order=age_order,
                severe_threshold=int(modeling.get("severe_error_threshold", 2)),
            )
        all_metrics[name] = metrics
        joblib.dump({"model": model, "features": X.columns.tolist(), "labels": labels}, models_dir / f"{name}.joblib")

    write_json(metrics_dir / "model_metrics.json", all_metrics)
    return all_metrics
