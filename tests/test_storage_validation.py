from src.data.storage import JsonlStore
from src.data.validation import validate_records


def test_jsonl_store_and_validation(tmp_path):
    path = tmp_path / "samples.jsonl"
    store = JsonlStore(path)
    record = {
        "sample_id": "s1",
        "status": "success",
        "result_age_rating": "12+",
        "answers_json": {"violence": "mild"},
    }
    store.append(record)
    store.append({**record, "sample_id": "s2"})
    rows = store.read_all()
    assert len(rows) == 2
    report = validate_records(rows)
    assert report["valid_samples"] == 2
    assert report["duplicate_answer_count"] == 1
    assert report["label_distribution"]["12+"] == 2
