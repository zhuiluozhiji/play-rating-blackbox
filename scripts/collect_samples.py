#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collector.collect_controller import collect_samples_async


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Google Play content rating questionnaire samples.")
    parser.add_argument("--config", default="configs/collector.yaml")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--strategy", default="baseline")
    parser.add_argument("--start-url", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    result = asyncio.run(
        collect_samples_async(
            config_path=args.config,
            limit=args.limit,
            strategy=args.strategy,
            start_url=args.start_url,
            dry_run=args.dry_run,
            no_submit=args.no_submit,
            resume=args.resume,
        )
    )
    print(result)


if __name__ == "__main__":
    main()
