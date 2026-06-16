#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import write_json
from src.data.storage import JsonlStore


def summarize_region_ratings(records: List[Dict[str, Any]], example_limit: int) -> Dict[str, Any]:
    authority_presence = Counter()
    authority_rating_distributions: Dict[str, Counter[str]] = {}
    authority_count_distribution = Counter()
    primary_label_distribution = Counter()

    samples_with_region_ratings = 0
    samples_with_multiple_authorities = 0
    missing_region_ratings = 0

    gp_iarc_both_present = 0
    gp_iarc_exact_match = 0
    gp_iarc_mismatch_examples: List[Dict[str, Any]] = []

    for record in records:
        if record.get("status") != "success":
            continue

        primary_label = str(record.get("result_age_rating") or "")
        if primary_label:
            primary_label_distribution[primary_label] += 1

        region_ratings = record.get("result_region_ratings") or {}
        if not region_ratings:
            missing_region_ratings += 1
            continue

        samples_with_region_ratings += 1
        authority_count_distribution[len(region_ratings)] += 1
        if len(region_ratings) > 1:
            samples_with_multiple_authorities += 1

        for authority, rating in region_ratings.items():
            authority_name = str(authority or "").strip()
            rating_text = str(rating or "").strip()
            if not authority_name or not rating_text:
                continue
            authority_presence[authority_name] += 1
            authority_rating_distributions.setdefault(authority_name, Counter())[rating_text] += 1

        google_play = str(region_ratings.get("Google Play") or "").strip()
        iarc_generic = str(region_ratings.get("IARC Generic") or "").strip()
        if google_play and iarc_generic:
            gp_iarc_both_present += 1
            if google_play == iarc_generic:
                gp_iarc_exact_match += 1
            elif len(gp_iarc_mismatch_examples) < example_limit:
                gp_iarc_mismatch_examples.append(
                    {
                        "sample_id": record.get("sample_id"),
                        "result_age_rating": primary_label,
                        "google_play": google_play,
                        "iarc_generic": iarc_generic,
                        "response_signature": record.get("response_signature"),
                    }
                )

    authority_rating_summary = {
        authority: dict(sorted(ratings.items(), key=lambda item: (-item[1], item[0])))
        for authority, ratings in sorted(authority_rating_distributions.items())
    }
    authority_presence_summary = dict(sorted(authority_presence.items(), key=lambda item: (-item[1], item[0])))

    return {
        "total_success_samples": sum(primary_label_distribution.values()),
        "samples_with_region_ratings": samples_with_region_ratings,
        "missing_region_ratings": missing_region_ratings,
        "samples_with_multiple_authorities": samples_with_multiple_authorities,
        "authority_count_distribution": dict(sorted(authority_count_distribution.items())),
        "distinct_authority_count": len(authority_presence_summary),
        "authority_presence_counts": authority_presence_summary,
        "authority_rating_distributions": authority_rating_summary,
        "primary_label_distribution": dict(sorted(primary_label_distribution.items())),
        "google_play_vs_iarc_generic": {
            "both_present_count": gp_iarc_both_present,
            "exact_match_count": gp_iarc_exact_match,
            "mismatch_count": gp_iarc_both_present - gp_iarc_exact_match,
            "mismatch_examples": gp_iarc_mismatch_examples,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize multi-authority age-rating results from standard JSONL samples.")
    parser.add_argument("--input", default="data/raw/real_20260615_full.samples.jsonl")
    parser.add_argument("--output", default="outputs/analysis/current/metrics/region_rating_summary.json")
    parser.add_argument("--example-limit", type=int, default=10)
    args = parser.parse_args()

    records = JsonlStore(args.input).read_all()
    report = summarize_region_ratings(records, args.example_limit)
    write_json(args.output, report)

    print(
        f"samples={report['total_success_samples']} "
        f"with_region_ratings={report['samples_with_region_ratings']} "
        f"authorities={report['distinct_authority_count']}"
    )
    gp_iarc = report["google_play_vs_iarc_generic"]
    print(
        f"google_play_vs_iarc: both_present={gp_iarc['both_present_count']} "
        f"exact_match={gp_iarc['exact_match_count']} mismatch={gp_iarc['mismatch_count']}"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
