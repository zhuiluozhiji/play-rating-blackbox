#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ensure_parent, write_json
from src.data.storage import answer_hash


DEFAULT_INPUT = "outputs/questionnaire_samples_cdp/20260611_142334/samples.jsonl"
DEFAULT_OUTPUT = "data/raw/real_20260611_142334.samples.jsonl"
DEFAULT_REPORT = "outputs/analysis/current/metrics/conversion_report.json"
DEFAULT_CATALOG = "data/questionnaire/real_question_catalog_20260611_142334.json"

AGE_RATING_PATTERN = re.compile(r"(?P<age>3|7|12|16|18)\s*\+")


def parse_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def normalize_age_rating(value: Any) -> str:
    text = str(value or "")
    match = AGE_RATING_PATTERN.search(text)
    if not match:
        return ""
    return f"{match.group('age')}+"


def normalize_answer_value(response: Dict[str, Any]) -> Any:
    labels = [str(label) for label in response.get("option_labels") or [] if str(label)]
    question_type = str(response.get("question_type") or "").lower()
    if question_type == "multi":
        return labels
    if len(labels) == 1:
        return labels[0]
    if labels:
        return labels
    return ""


def build_answers_json(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    answers: Dict[str, Any] = {}
    for response in responses:
        question_key = str(response.get("question_key") or "")
        if not question_key:
            continue
        value = normalize_answer_value(response)
        if value == "" or value == []:
            continue
        answers[question_key] = value
    return answers


def build_region_ratings(ratings: List[Dict[str, Any]]) -> Dict[str, str]:
    region_ratings: Dict[str, str] = {}
    for rating in ratings:
        authority = str(rating.get("authority") or "")
        value = str(rating.get("rating") or "")
        if authority and value:
            region_ratings[authority] = value
    return region_ratings


def update_question_catalog(
    catalog: Dict[str, Dict[str, Any]],
    responses: List[Dict[str, Any]],
) -> None:
    for response in responses:
        question_key = str(response.get("question_key") or "")
        if not question_key:
            continue
        entry = catalog.setdefault(
            question_key,
            {
                "question_key": question_key,
                "question_text": str(response.get("question_text") or ""),
                "question_type": str(response.get("question_type") or "unknown"),
                "seen_count": 0,
                "option_labels": Counter(),
                "option_keys": Counter(),
            },
        )
        entry["seen_count"] += 1
        for label in response.get("option_labels") or []:
            if str(label):
                entry["option_labels"][str(label)] += 1
        for option_key in response.get("option_keys") or []:
            if str(option_key):
                entry["option_keys"][str(option_key)] += 1


def render_question_catalog(catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    questions = []
    for entry in catalog.values():
        questions.append(
            {
                "question_key": entry["question_key"],
                "question_text": entry["question_text"],
                "question_type": entry["question_type"],
                "seen_count": entry["seen_count"],
                "option_labels": dict(entry["option_labels"]),
                "option_keys": dict(entry["option_keys"]),
            }
        )
    questions.sort(key=lambda item: (-item["seen_count"], item["question_key"]))
    return {
        "question_count": len(questions),
        "questions": questions,
    }


def skip_reason(record: Dict[str, Any], answers: Dict[str, Any], label: str, seen_signatures: set[str]) -> Optional[str]:
    status = record.get("status")
    rating_result = record.get("rating_result") or {}
    response_signature = str(record.get("response_signature") or "")

    if status != "complete":
        return str(status or "non_complete")
    if not rating_result.get("ok"):
        return "rating_not_ok"
    if not label:
        return "unparseable_rating"
    if not answers:
        return "empty_responses"
    if response_signature and response_signature in seen_signatures:
        return "duplicate_signature"
    return None


def convert_record(record: Dict[str, Any], answers: Dict[str, Any], label: str) -> Dict[str, Any]:
    rating_result = record.get("rating_result") or {}
    ratings = rating_result.get("ratings") or []
    return {
        "sample_id": record.get("sample_id"),
        "timestamp": record.get("completed_at"),
        "strategy": "cdp_random_path",
        "questionnaire_version": record.get("final_state_signature") or "google_play_cdp_20260611",
        "answers_json": answers,
        "visible_questions": list(answers.keys()),
        "skipped_questions": [],
        "result_age_rating": label,
        "result_region_ratings": build_region_ratings(ratings),
        "content_descriptors": rating_result.get("primary_content_descriptors") or [],
        "interactive_elements": rating_result.get("primary_interactive_elements") or [],
        "status": "success",
        "submit_status": record.get("status"),
        "error": record.get("error") or None,
        "evidence": {
            "summary_url": rating_result.get("summary_url"),
            "summary_title": rating_result.get("summary_title"),
            "summary_state_signature": rating_result.get("summary_state_signature"),
            "summary_body_fingerprint": rating_result.get("summary_body_fingerprint"),
        },
        "notes": record.get("path_summary"),
        "source": {
            "format": "questionnaire_samples_cdp",
            "path_id": record.get("path_id"),
            "response_signature": record.get("response_signature"),
            "rating_signature": record.get("rating_signature"),
            "answer_count": record.get("answer_count"),
            "continue_count": record.get("continue_count"),
            "final_url": record.get("final_url"),
            "final_title": record.get("final_title"),
        },
        "response_signature": record.get("response_signature"),
        "answer_hash": answer_hash(answers),
    }


def convert_samples(input_paths: Sequence[Path], output_path: Path, report_path: Path, catalog_path: Path) -> Dict[str, Any]:
    converted: List[Dict[str, Any]] = []
    skip_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    source_rating_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    seen_signatures: set[str] = set()
    question_catalog: Dict[str, Dict[str, Any]] = {}
    missing_inputs: List[str] = []

    total = 0
    normalized_inputs = [Path(path) for path in input_paths]
    for input_path in normalized_inputs:
        if not input_path.exists():
            missing_inputs.append(str(input_path))
            continue
        for record in parse_jsonl(input_path):
            total += 1
            status_counts[str(record.get("status") or "unknown")] += 1
            rating_result = record.get("rating_result") or {}
            primary_rating = str(rating_result.get("primary_rating") or "")
            if primary_rating:
                source_rating_counts[primary_rating] += 1

            responses = record.get("responses") or []
            answers = build_answers_json(responses)
            label = normalize_age_rating(primary_rating)
            reason = skip_reason(record, answers, label, seen_signatures)
            if reason:
                skip_counts[reason] += 1
                continue

            response_signature = str(record.get("response_signature") or "")
            if response_signature:
                seen_signatures.add(response_signature)
            update_question_catalog(question_catalog, responses)
            converted_record = convert_record(record, answers, label)
            converted.append(converted_record)
            label_counts[label] += 1

    output = ensure_parent(output_path)
    with output.open("w", encoding="utf-8") as handle:
        for record in converted:
            json.dump(record, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")

    rendered_catalog = render_question_catalog(question_catalog)
    write_json(catalog_path, rendered_catalog)

    report = {
        "input_path": str(normalized_inputs[0]) if len(normalized_inputs) == 1 else "",
        "input_paths": [str(path) for path in normalized_inputs],
        "missing_input_paths": missing_inputs,
        "output_path": str(output),
        "catalog_path": str(catalog_path),
        "total_input_samples": total,
        "converted_samples": len(converted),
        "skipped_samples": total - len(converted),
        "status_distribution": dict(status_counts),
        "source_primary_rating_distribution": dict(source_rating_counts),
        "label_distribution": dict(label_counts),
        "skip_reasons": dict(skip_counts),
        "unique_response_signatures": len(seen_signatures),
        "question_count": rendered_catalog["question_count"],
    }
    write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CDP questionnaire samples into the standard training JSONL format.")
    parser.add_argument("--input", nargs="+", default=[DEFAULT_INPUT])
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = convert_samples(
        input_paths=[Path(path) for path in args.input],
        output_path=Path(args.output),
        report_path=Path(args.report),
        catalog_path=Path(args.catalog),
    )
    print(
        "converted={converted_samples} skipped={skipped_samples} questions={question_count} output={output_path}".format(
            **report
        )
    )


if __name__ == "__main__":
    main()
