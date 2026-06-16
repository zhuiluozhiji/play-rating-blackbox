#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ensure_parent, write_json


DEFAULT_INPUT = "data/raw/real_20260611_142334.samples.jsonl"
DEFAULT_OUTPUT = "data/raw/real_20260611_142334_n1150.samples.jsonl"
DEFAULT_REPORT = "outputs/analysis/current/metrics/sample_subset_report.json"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return records


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> Path:
    output = ensure_parent(path)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
    return output


def label_distribution(records: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(record.get("result_age_rating") or "missing") for record in records))


def make_subset(input_path: Path, output_path: Path, report_path: Path, target_count: int, seed: int) -> Dict[str, Any]:
    records = read_jsonl(input_path)
    if target_count <= 0:
        raise ValueError("--count must be positive.")
    if target_count > len(records):
        raise ValueError(f"--count={target_count} exceeds input sample count {len(records)}.")

    rng = random.Random(seed)
    selected_indexes = set(rng.sample(range(len(records)), target_count))
    selected = [record for index, record in enumerate(records) if index in selected_indexes]
    removed = [record for index, record in enumerate(records) if index not in selected_indexes]

    output = write_jsonl(output_path, selected)
    report = {
        "input_path": str(input_path),
        "output_path": str(output),
        "seed": seed,
        "input_samples": len(records),
        "selected_samples": len(selected),
        "removed_samples": len(removed),
        "input_label_distribution": label_distribution(records),
        "selected_label_distribution": label_distribution(selected),
        "removed_label_distribution": label_distribution(removed),
        "removed_sample_ids": [record.get("sample_id") for record in removed],
        "removed_response_signatures": [record.get("response_signature") for record in removed],
    }
    write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reproducible random subset from a standard JSONL sample file.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--count", type=int, default=1150)
    parser.add_argument("--seed", type=int, default=20260613)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = make_subset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        report_path=Path(args.report),
        target_count=args.count,
        seed=args.seed,
    )
    print(
        "selected={selected_samples} removed={removed_samples} seed={seed} output={output_path}".format(
            **report
        )
    )


if __name__ == "__main__":
    main()
