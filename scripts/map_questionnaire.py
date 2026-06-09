#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import write_json
from src.data.schema import default_question_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an initial questionnaire schema template.")
    parser.add_argument("--output", default="data/questionnaire/question_schema.json")
    args = parser.parse_args()
    write_json(args.output, default_question_schema())
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
