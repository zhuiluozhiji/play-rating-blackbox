from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List

from src.data.storage import answer_hash


def validate_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    hashes = [answer_hash(row.get("answers_json") or {}) for row in rows]
    hash_counts = Counter(hashes)
    labels = [row.get("result_age_rating") for row in rows if row.get("result_age_rating")]
    statuses = Counter(row.get("status") or row.get("submit_status") or "unknown" for row in rows)
    failures = Counter(row.get("error") or row.get("status") for row in rows if row.get("status") != "success")
    questions = Counter()
    option_coverage: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        answers = row.get("answers_json") or {}
        for question_id, value in answers.items():
            questions[question_id] += 1
            if isinstance(value, list):
                for item in value:
                    option_coverage[question_id][str(item)] += 1
            else:
                option_coverage[question_id][str(value)] += 1

    valid = [
        row for row in rows
        if row.get("status") == "success"
        and row.get("result_age_rating")
        and row.get("answers_json")
    ]
    return {
        "total_samples": len(rows),
        "valid_samples": len(valid),
        "invalid_samples": len(rows) - len(valid),
        "duplicate_answer_count": sum(count - 1 for count in hash_counts.values() if count > 1),
        "unique_answer_count": len(hash_counts),
        "label_distribution": dict(Counter(labels)),
        "status_distribution": dict(statuses),
        "failure_reasons": dict(failures),
        "question_coverage": dict(questions),
        "option_coverage": {
            question_id: dict(counter)
            for question_id, counter in option_coverage.items()
        },
        "missing_label_count": sum(1 for row in rows if not row.get("result_age_rating")),
    }
