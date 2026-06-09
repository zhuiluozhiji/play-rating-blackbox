from src.modeling.features import build_feature_matrix, records_to_dataframe


def test_features_encode_not_visible_separately():
    records = [
        {
            "sample_id": "s1",
            "strategy": "baseline",
            "status": "success",
            "result_age_rating": "3+",
            "answers_json": {"violence": "no", "blood": "not_visible", "ugc": ["chat"]},
            "visible_questions": ["violence", "ugc"],
            "skipped_questions": ["blood"],
        },
        {
            "sample_id": "s2",
            "strategy": "single_factor",
            "status": "success",
            "result_age_rating": "12+",
            "answers_json": {"violence": "realistic", "blood": "no", "ugc": ["none"]},
            "visible_questions": ["violence", "blood", "ugc"],
            "skipped_questions": [],
        },
    ]
    df = records_to_dataframe(records)
    X, y = build_feature_matrix(df)
    assert "answer__blood_not_visible" in X.columns
    assert "answer__blood_no" in X.columns
    assert y.tolist() == ["3+", "12+"]
