from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


def records_to_dataframe(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for record in records:
        row: Dict[str, Any] = {
            "sample_id": record.get("sample_id"),
            "strategy": record.get("strategy"),
            "result_age_rating": record.get("result_age_rating"),
            "status": record.get("status"),
        }
        answers = record.get("answers_json") or {}
        for question_id, value in answers.items():
            if isinstance(value, list):
                row[f"answer__{question_id}"] = "|".join(sorted(map(str, value)))
            else:
                row[f"answer__{question_id}"] = str(value)
        row["visible_question_count"] = len(record.get("visible_questions") or [])
        row["skipped_question_count"] = len(record.get("skipped_questions") or [])
        row["content_descriptor_count"] = len(record.get("content_descriptors") or [])
        row["interactive_element_count"] = len(record.get("interactive_elements") or [])
        rows.append(row)
    return pd.DataFrame(rows)


def add_theme_scores(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    theme_keywords = {
        "violence_score": ["violence", "blood"],
        "sexual_content_score": ["sexual"],
        "language_score": ["language"],
        "drug_score": ["drug"],
        "gambling_score": ["gambling"],
        "fear_score": ["fear"],
        "ugc_score": ["ugc"],
        "interaction_score": ["purchase", "interaction"],
    }
    risk_map = {
        "not_visible": 0,
        "no": 0,
        "none": 0,
        "yes": 1,
        "mild": 1,
        "suggestive": 1,
        "reference": 1,
        "simulated": 2,
        "realistic": 2,
        "strong": 2,
        "intense": 2,
        "blood": 2,
        "nudity": 3,
        "use": 3,
        "gore": 4,
        "graphic": 4,
        "explicit": 5,
        "real_money": 5,
        "user_generated_content": 1,
        "chat": 1,
        "location_sharing": 1,
    }

    def value_score(value: Any) -> int:
        parts = str(value).split("|")
        return sum(risk_map.get(part, 0) for part in parts)

    answer_cols = [col for col in result.columns if col.startswith("answer__")]
    for theme_col, keywords in theme_keywords.items():
        matched = [
            col for col in answer_cols
            if any(keyword in col.lower() for keyword in keywords)
        ]
        result[theme_col] = result[matched].apply(
            lambda row: sum(value_score(value) for value in row.values),
            axis=1,
        ) if matched else 0
    score_cols = list(theme_keywords.keys())
    result["high_risk_count"] = (result[score_cols] >= 3).sum(axis=1)
    result["medium_risk_count"] = ((result[score_cols] >= 1) & (result[score_cols] < 3)).sum(axis=1)
    result["triggered_branch_count"] = (result[answer_cols] != "not_visible").sum(axis=1) if answer_cols else 0
    return result


def build_feature_matrix(
    df: pd.DataFrame,
    label_column: str = "result_age_rating",
) -> Tuple[pd.DataFrame, pd.Series]:
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=str)
    enriched = add_theme_scores(df)
    y = enriched[label_column].astype(str)
    drop_cols = {
        "sample_id",
        "status",
        label_column,
    }
    feature_df = enriched.drop(columns=[col for col in drop_cols if col in enriched.columns])
    categorical_cols = [
        col for col in feature_df.columns
        if feature_df[col].dtype == object or col.startswith("answer__") or col == "strategy"
    ]
    feature_df = pd.get_dummies(feature_df, columns=categorical_cols, dummy_na=False)
    feature_df = feature_df.fillna(0)
    return feature_df, y
