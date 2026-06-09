#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import read_json
from src.collector.sample_generator import generate_samples
from src.data.schema import SampleRecord, default_question_schema
from src.data.storage import JsonlStore


def infer_rating(answers: dict) -> str:
    risk = 0
    high_trigger = False
    score_map = {
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
    for value in answers.values():
        values = value if isinstance(value, list) else [value]
        for item in values:
            score = score_map.get(item, 0)
            risk += score
            high_trigger = high_trigger or score >= 5
    if high_trigger or risk >= 10:
        return "18+"
    if risk >= 7:
        return "16+"
    if risk >= 4:
        return "12+"
    if risk >= 2:
        return "7+"
    return "3+"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic samples for local pipeline testing.")
    parser.add_argument("--output", default="data/raw/samples.jsonl")
    parser.add_argument("--schema", default="data/questionnaire/question_schema.json")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--strategy", default="all")
    args = parser.parse_args()

    schema = read_json(args.schema, default=None) or default_question_schema()
    samples = generate_samples(schema, args.strategy, args.count)
    store = JsonlStore(args.output)
    records = []
    for answers in samples:
        visible = [key for key, value in answers.items() if value != "not_visible"]
        skipped = [key for key, value in answers.items() if value == "not_visible"]
        rating = infer_rating(answers)
        record = SampleRecord(
            strategy=f"synthetic_{args.strategy}",
            questionnaire_version=schema.get("questionnaire_version", "synthetic_v1"),
            answers_json=answers,
            visible_questions=visible,
            skipped_questions=skipped,
            submit_status="synthetic",
            result_age_rating=rating,
            result_region_ratings={"IARC": rating},
            content_descriptors=[],
            interactive_elements=[],
            status="success",
            notes="Synthetic sample for local pipeline validation.",
        )
        records.append(record.to_dict())
    store.append_many(records)
    print(f"Wrote {len(records)} synthetic records to {args.output}")


if __name__ == "__main__":
    main()
