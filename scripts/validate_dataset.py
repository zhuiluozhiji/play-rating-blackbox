#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import write_json
from src.data.storage import JsonlStore
from src.data.validation import validate_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw JSONL samples.")
    parser.add_argument("--input", default="data/raw/samples.jsonl")
    parser.add_argument("--output", default="outputs/metrics/dataset_validation.json")
    args = parser.parse_args()

    records = JsonlStore(args.input).read_all()
    report = validate_records(records)
    write_json(args.output, report)
    print(f"total={report['total_samples']} valid={report['valid_samples']} unique={report['unique_answer_count']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
