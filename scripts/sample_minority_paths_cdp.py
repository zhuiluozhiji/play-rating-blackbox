#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sample_questionnaire_paths_cdp as sampler
from sample_questionnaire_paths_cdp import SampleConfig, endpoint_available


LOW_LABEL_HINTS = (
    "no",
    "none",
    "rare",
    "rarely",
    "brief",
    "mild",
    "limited",
    "small and infrequent",
    "fantastical",
    "unrealistic",
    "referred",
    "distant",
    "obscured",
    "no nudity",
    "not seen",
)
MEDIUM_LABEL_HINTS = (
    "yes",
    "moderate",
    "often",
    "scary",
    "suggestive",
    "revealing",
    "minor profanities",
    "alcohol",
    "tobacco",
    "users interact",
    "shares location",
    "purchases of digital goods",
)
HIGH_LABEL_HINTS = (
    "explicit",
    "pornographic",
    "sexual violence",
    "under 18",
    "illegal or recreational drugs",
    "encourages/glamorizes",
    "detailed instruction",
    "real money",
    "cash payouts",
    "cash rewards",
    "wager",
    "terrorism",
    "nazi",
    "swastika",
    "graphic",
    "gore",
    "gory",
    "large or frequent",
    "high",
    "realistic descriptions of crimes",
)
SEVERE_QUESTION_HINTS = (
    "terrorism",
    "nazi",
    "swastika",
    "sexual violence",
    "under 18",
    "pornographic",
    "illegal or recreational drugs",
    "cash payouts",
    "cash rewards",
    "real money",
    "wager",
    "national identity",
    "criminal offenses",
    "graphic violence outside",
)
CATEGORY_ORDER = {
    "low": ("All Other App Types", "Social or Communication", "Game"),
    "medium": ("Game", "All Other App Types", "Social or Communication"),
}
PROFILE_ACCEPTED_RATINGS = {
    "low": frozenset({"3+", "7+", "12+"}),
    "medium": frozenset({"7+", "12+", "16+"}),
    "mixed": frozenset({"3+", "7+", "12+", "16+"}),
}
PROFILE_TARGET_RATING_ORDER = {
    "low": ("7+", "3+", "12+"),
    "medium": ("12+", "16+", "7+"),
    "mixed": ("7+", "12+", "3+", "16+"),
}
DEFAULT_POLICY_SOURCES = (
    Path("data/raw/real_20260615_full.samples.jsonl"),
    Path("data/raw/real_20260611_142334.samples.jsonl"),
)

ACTIVE_PROFILE = "mixed"
CURRENT_ATTEMPT_INDEX = 0
CURRENT_TARGET_RATING = ""
ACTIVE_ACCEPTED_RATINGS: frozenset[str] = frozenset()
ANSWER_POLICY: Dict[str, Dict[str, Any]] = {}
ANSWER_POLICY_RATING_COUNTS: Counter[str] = Counter()
ANSWER_POLICY_SOURCES: List[str] = []
TEMPLATE_LIBRARY: Dict[str, List[Dict[str, Any]]] = {}
CURRENT_TEMPLATE_ANSWERS: Dict[str, Any] = {}
CURRENT_TEMPLATE_ID = ""
CURRENT_MUTATION_QUESTION_KEY = ""


def normalize(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def current_relaxation_level() -> int:
    # Relax slowly so later samples do not drift into obviously high-risk branches.
    return min(2, CURRENT_ATTEMPT_INDEX // 20)


def normalize_rating(value: Any) -> str:
    text = str(value or "").strip()
    if text in {"3+", "7+", "12+", "16+", "18+"}:
        return text
    for rating in ("3+", "7+", "12+", "16+", "18+"):
        if rating in text:
            return rating
    return text


def deterministic_index(key: str, size: int) -> int:
    if size <= 1:
        return 0
    payload = f"{CURRENT_ATTEMPT_INDEX}|{CURRENT_TARGET_RATING}|{key}"
    digest = hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16) % size


def normalize_answer_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(sorted(str(item) for item in value if str(item)))
    return str(value or "")


def policy_sources() -> List[Path]:
    for path in DEFAULT_POLICY_SOURCES:
        if path.exists():
            return [path]
    return []


def load_template_library(accepted_ratings: frozenset[str]) -> Dict[str, List[Dict[str, Any]]]:
    templates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    sources = policy_sources()
    if not sources:
        return {}
    path = sources[0]
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rating = normalize_rating(record.get("result_age_rating"))
            if rating not in accepted_ratings:
                continue
            answers_json = record.get("answers_json") or {}
            if not answers_json:
                continue
            templates[rating].append(
                {
                    "sample_id": str(record.get("sample_id") or ""),
                    "rating": rating,
                    "answers_json": answers_json,
                }
            )
    return dict(templates)


def load_answer_policy(accepted_ratings: frozenset[str]) -> tuple[Dict[str, Dict[str, Any]], Counter[str], List[str]]:
    policy: Dict[str, Dict[str, Any]] = {}
    rating_counts: Counter[str] = Counter()
    used_sources: List[str] = []

    for path in policy_sources():
        used_sources.append(str(path))
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rating = normalize_rating(record.get("result_age_rating"))
                if rating not in accepted_ratings:
                    continue
                rating_counts[rating] += 1
                answers = record.get("answers_json") or {}
                for question_key, answer_value in answers.items():
                    question_key = str(question_key or "")
                    if not question_key:
                        continue
                    entry = policy.setdefault(
                        question_key,
                        {
                            "single_by_rating": defaultdict(Counter),
                            "multi_by_rating": defaultdict(Counter),
                            "single_all": Counter(),
                            "multi_all": Counter(),
                        },
                    )
                    if isinstance(answer_value, list):
                        labels = tuple(sorted(str(value) for value in answer_value if str(value)))
                        if not labels:
                            continue
                        entry["multi_by_rating"][rating][labels] += 1
                        entry["multi_all"][labels] += 1
                    else:
                        label = str(answer_value or "")
                        if not label:
                            continue
                        entry["single_by_rating"][rating][label] += 1
                        entry["single_all"][label] += 1

    return policy, rating_counts, used_sources


def target_rating_cycle(profile: str) -> List[str]:
    desired = [rating for rating in PROFILE_TARGET_RATING_ORDER.get(profile, ()) if rating in ACTIVE_ACCEPTED_RATINGS]
    if not desired:
        desired = sorted(ACTIVE_ACCEPTED_RATINGS)
    cycle: List[str] = []
    for rating in desired:
        count = ANSWER_POLICY_RATING_COUNTS.get(rating, 0)
        if count <= 10:
            weight = 3
        elif count <= 40:
            weight = 2
        else:
            weight = 1
        cycle.extend([rating] * weight)
    return cycle or desired


def choose_target_rating() -> str:
    cycle = target_rating_cycle(ACTIVE_PROFILE)
    if not cycle:
        return ""
    return cycle[(CURRENT_ATTEMPT_INDEX - 1) % len(cycle)]


def choose_current_template() -> Dict[str, Any] | None:
    ratings_to_try = []
    if CURRENT_TARGET_RATING:
        ratings_to_try.append(CURRENT_TARGET_RATING)
    for rating in PROFILE_TARGET_RATING_ORDER.get(ACTIVE_PROFILE, ()):
        if rating not in ratings_to_try and rating in ACTIVE_ACCEPTED_RATINGS:
            ratings_to_try.append(rating)
    for rating in sorted(ACTIVE_ACCEPTED_RATINGS):
        if rating not in ratings_to_try:
            ratings_to_try.append(rating)
    for rating in ratings_to_try:
        templates = TEMPLATE_LIBRARY.get(rating) or []
        if not templates:
            continue
        return templates[deterministic_index(f"template|{rating}", len(templates))]
    return None


def policy_single_labels(question_key: str) -> List[str]:
    entry = ANSWER_POLICY.get(question_key)
    if not entry:
        return []
    labels: List[str] = []
    seen: set[str] = set()
    counters: List[Counter[str]] = []
    if CURRENT_TARGET_RATING:
        counters.append(entry["single_by_rating"].get(CURRENT_TARGET_RATING, Counter()))
    for rating in PROFILE_TARGET_RATING_ORDER.get(ACTIVE_PROFILE, ()):
        if rating != CURRENT_TARGET_RATING and rating in ACTIVE_ACCEPTED_RATINGS:
            counters.append(entry["single_by_rating"].get(rating, Counter()))
    counters.append(entry["single_all"])
    for counter in counters:
        for label, _ in counter.most_common():
            normalized_label = normalize(label)
            if normalized_label in seen:
                continue
            seen.add(normalized_label)
            labels.append(normalized_label)
    return labels


def policy_multi_label_sets(question_key: str) -> List[tuple[str, ...]]:
    entry = ANSWER_POLICY.get(question_key)
    if not entry:
        return []
    subsets: List[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    counters: List[Counter[tuple[str, ...]]] = []
    if CURRENT_TARGET_RATING:
        counters.append(entry["multi_by_rating"].get(CURRENT_TARGET_RATING, Counter()))
    for rating in PROFILE_TARGET_RATING_ORDER.get(ACTIVE_PROFILE, ()):
        if rating != CURRENT_TARGET_RATING and rating in ACTIVE_ACCEPTED_RATINGS:
            counters.append(entry["multi_by_rating"].get(rating, Counter()))
    counters.append(entry["multi_all"])
    for counter in counters:
        ordered = sorted(
            counter.items(),
            key=lambda item: (-item[1], len(item[0]), tuple(normalize(label) for label in item[0])),
        )
        for subset, _ in ordered:
            normalized_subset = tuple(sorted(normalize(label) for label in subset))
            if normalized_subset in seen:
                continue
            seen.add(normalized_subset)
            subsets.append(normalized_subset)
    return subsets


def choose_mutation_question_key(template_answers: Dict[str, Any]) -> str:
    candidates: List[str] = []
    for question_key, template_value in template_answers.items():
        if isinstance(template_value, list):
            variants = policy_multi_label_sets(question_key)
            template_variant = normalize_answer_value(template_value)
            alternative_count = sum(1 for variant in variants if variant != template_variant)
        else:
            variants = policy_single_labels(question_key)
            template_variant = normalize_answer_value(template_value)
            alternative_count = sum(1 for variant in variants if variant != normalize(template_variant))
        if alternative_count > 0:
            candidates.append(question_key)
    if not candidates:
        return ""
    return candidates[deterministic_index(f"mutation|{CURRENT_TEMPLATE_ID}", len(candidates))]


def choose_policy_single_option(question: Dict[str, Any], options: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    entry = ANSWER_POLICY.get(str(question.get("question_key") or ""))
    if not entry:
        return None
    by_label = {normalize(option.get("label")): option for option in options}
    counters: List[Counter[str]] = []
    if CURRENT_TARGET_RATING:
        counters.append(entry["single_by_rating"].get(CURRENT_TARGET_RATING, Counter()))
    for rating in PROFILE_TARGET_RATING_ORDER.get(ACTIVE_PROFILE, ()):
        if rating != CURRENT_TARGET_RATING and rating in ACTIVE_ACCEPTED_RATINGS:
            counters.append(entry["single_by_rating"].get(rating, Counter()))
    counters.append(entry["single_all"])

    seen_labels: set[str] = set()
    ranked_labels: List[str] = []
    for counter in counters:
        if not counter:
            continue
        for label, _ in counter.most_common():
            normalized_label = normalize(label)
            if normalized_label not in by_label or normalized_label in seen_labels:
                continue
            seen_labels.add(normalized_label)
            ranked_labels.append(normalized_label)
    if not ranked_labels:
        return None
    chosen_index = deterministic_index(str(question.get("question_key") or ""), len(ranked_labels))
    return by_label[ranked_labels[chosen_index]]


def choose_policy_multi_options(question: Dict[str, Any], options: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]] | None:
    entry = ANSWER_POLICY.get(str(question.get("question_key") or ""))
    if not entry:
        return None
    by_label = {normalize(option.get("label")): option for option in options}
    counters: List[Counter[tuple[str, ...]]] = []
    if CURRENT_TARGET_RATING:
        counters.append(entry["multi_by_rating"].get(CURRENT_TARGET_RATING, Counter()))
    for rating in PROFILE_TARGET_RATING_ORDER.get(ACTIVE_PROFILE, ()):
        if rating != CURRENT_TARGET_RATING and rating in ACTIVE_ACCEPTED_RATINGS:
            counters.append(entry["multi_by_rating"].get(rating, Counter()))
    counters.append(entry["multi_all"])

    ranked_subsets: List[tuple[str, ...]] = []
    seen_subsets: set[tuple[str, ...]] = set()
    for counter in counters:
        if not counter:
            continue
        ordered = sorted(
            counter.items(),
            key=lambda item: (-item[1], len(item[0]), tuple(normalize(label) for label in item[0])),
        )
        for subset, _ in ordered:
            normalized_subset = tuple(sorted(normalize(label) for label in subset))
            if normalized_subset in seen_subsets:
                continue
            if not all(label in by_label for label in normalized_subset):
                continue
            seen_subsets.add(normalized_subset)
            ranked_subsets.append(normalized_subset)
    if not ranked_subsets:
        return None
    chosen_index = deterministic_index(str(question.get("question_key") or ""), len(ranked_subsets))
    chosen_subset = ranked_subsets[chosen_index]
    return [by_label[label] for label in chosen_subset]


def choose_template_single_option(question: Dict[str, Any], options: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    question_key = str(question.get("question_key") or "")
    if not question_key or question_key not in CURRENT_TEMPLATE_ANSWERS:
        return None
    by_label = {normalize(option.get("label")): option for option in options}
    template_label = normalize(str(CURRENT_TEMPLATE_ANSWERS[question_key] or ""))
    alternatives = policy_single_labels(question_key)
    if question_key == CURRENT_MUTATION_QUESTION_KEY:
        mutation_candidates = [label for label in alternatives if label != template_label and label in by_label]
        if mutation_candidates:
            chosen_index = deterministic_index(f"mut-single|{question_key}", len(mutation_candidates))
            return by_label[mutation_candidates[chosen_index]]
    return by_label.get(template_label)


def choose_template_multi_options(question: Dict[str, Any], options: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]] | None:
    question_key = str(question.get("question_key") or "")
    if not question_key or question_key not in CURRENT_TEMPLATE_ANSWERS:
        return None
    by_label = {normalize(option.get("label")): option for option in options}
    template_subset = normalize_answer_value(CURRENT_TEMPLATE_ANSWERS[question_key])
    if not isinstance(template_subset, tuple):
        return None
    if question_key == CURRENT_MUTATION_QUESTION_KEY:
        alternatives = [
            subset for subset in policy_multi_label_sets(question_key)
            if subset != template_subset and all(label in by_label for label in subset)
        ]
        if alternatives:
            chosen_index = deterministic_index(f"mut-multi|{question_key}", len(alternatives))
            chosen_subset = alternatives[chosen_index]
            return [by_label[label] for label in chosen_subset]
    if all(label in by_label for label in template_subset):
        return [by_label[label] for label in template_subset]
    return None


def option_score(question_text: str, option_label: str) -> int:
    question = normalize(question_text)
    label = normalize(option_label)

    score = 2 if label == "yes" else 1
    if label == "no" or label == "none":
        score = 0

    if any(hint in label for hint in LOW_LABEL_HINTS):
        score = min(score, 1)
    if any(hint in label for hint in MEDIUM_LABEL_HINTS):
        score = max(score, 2)
    if any(hint in label for hint in HIGH_LABEL_HINTS):
        score = max(score, 5)
    if any(hint in question for hint in SEVERE_QUESTION_HINTS) and label == "yes":
        score = max(score, 5)
    if "realistic" in label and "unrealistic" not in label:
        score = max(score, 3)
    if "close-up" in label:
        score = max(score, 3)
    return score


def choose_category_option(options: Sequence[Dict[str, Any]], profile: str, rng: random.Random) -> Dict[str, Any] | None:
    if profile == "low":
        relaxation = current_relaxation_level()
        if relaxation == 0:
            preferred = ["All Other App Types"]
        elif relaxation == 1:
            preferred = ["All Other App Types", "Game"]
        else:
            preferred = ["All Other App Types", "Game", "Social or Communication"]
    else:
        preferred = list(CATEGORY_ORDER["medium"])
        if preferred:
            offset = CURRENT_ATTEMPT_INDEX % len(preferred)
            preferred = preferred[offset:] + preferred[:offset]
    unselected = [option for option in options if not option.get("selected")]
    by_label = {option.get("label"): option for option in unselected}
    for label in preferred:
        if label in by_label:
            return by_label[label]
    by_label = {option.get("label"): option for option in options}
    for label in preferred:
        if label in by_label:
            return by_label[label]
    pool = unselected or list(options)
    return rng.choice(pool) if pool else None


def best_single_option(question: Dict[str, Any], profile: str, rng: random.Random) -> Dict[str, Any]:
    all_options = list(question["options"])
    options = [option for option in all_options if not option.get("selected")] or all_options
    if profile == "low":
        template_option = choose_template_single_option(question, options)
        if template_option is not None:
            return template_option
    if profile in {"low", "medium", "mixed"}:
        policy_option = choose_policy_single_option(question, options)
        if policy_option is not None:
            return policy_option
    if normalize(question.get("text")) == "category":
        category_option = choose_category_option(options, profile, rng)
        if category_option is not None:
            return category_option

    scored = [
        (option_score(question.get("text", ""), option.get("label", "")), index, option)
        for index, option in enumerate(options)
    ]
    if profile == "low":
        minimum = min(score for score, _, _ in scored)
        threshold = minimum + current_relaxation_level()
        candidates = [option for score, _, option in scored if score <= threshold]
        lowest_score_candidates = [option for score, _, option in scored if score == minimum]
        if minimum <= 1:
            return rng.choice(lowest_score_candidates)
        return rng.choice(candidates)

    medium_threshold = 3 + min(1, current_relaxation_level())
    safe_medium = [(score, index, option) for score, index, option in scored if score <= medium_threshold]
    if not safe_medium:
        safe_medium = scored
    target = 2
    best_distance = min(abs(score - target) for score, _, _ in safe_medium)
    candidates = [option for score, _, option in safe_medium if abs(score - target) == best_distance]
    return rng.choice(candidates)


def best_multi_options(question: Dict[str, Any], profile: str, rng: random.Random, max_count: int = 3) -> List[Dict[str, Any]]:
    all_options = list(question["options"])
    options = [option for option in all_options if not option.get("selected")] or all_options
    if profile == "low":
        template_subset = choose_template_multi_options(question, options)
        if template_subset is not None:
            return template_subset
    if profile in {"low", "medium", "mixed"}:
        policy_subset = choose_policy_multi_options(question, options)
        if policy_subset is not None:
            return policy_subset
    scored = [
        (option_score(question.get("text", ""), option.get("label", "")), index, option)
        for index, option in enumerate(options)
    ]
    if profile == "low":
        safe_threshold = 1 + min(1, current_relaxation_level())
        safe = [(score, index, option) for score, index, option in scored if score <= safe_threshold]
        if not safe:
            safe = sorted(scored, key=lambda item: (item[0], item[1]))[:1]
        ordered_safe = sorted(safe, key=lambda item: (item[0], item[1]))
        max_low_count = 1 if current_relaxation_level() == 0 else 2
        count = min(len(ordered_safe), max_low_count, max_count)
        chosen = ordered_safe[:count]
        return [option for _, _, option in chosen]

    safe_medium = [
        (score, index, option)
        for score, index, option in scored
        if 1 <= score <= 3 + min(1, current_relaxation_level())
    ]
    if not safe_medium:
        safe_medium = [(score, index, option) for score, index, option in scored if score <= 3] or scored
    count = min(len(safe_medium), rng.randint(1, max_count))
    return [option for _, _, option in rng.sample(safe_medium, count)]


def choose_profile(rng: random.Random) -> str:
    if ACTIVE_PROFILE == "mixed":
        return "medium" if rng.random() < 0.65 else "low"
    if ACTIVE_PROFILE == "low":
        return "low"
    if ACTIVE_PROFILE == "medium":
        cycle = CURRENT_ATTEMPT_INDEX % 10
        if cycle == 9:
            return "low"
        return "medium"
    return ACTIVE_PROFILE


def choose_action_for_question(
    question: Dict[str, Any],
    rng: random.Random,
    max_options_per_question: int,
    max_multi_combinations: int,
) -> Dict[str, Any]:
    profile = choose_profile(rng)
    question = dict(question)
    question["options"] = list(question.get("options", []))[:max_options_per_question]
    if not question["options"]:
        raise RuntimeError(f"No selectable options for question {question.get('question_key')}.")

    if question.get("question_type") == "multi":
        subset = best_multi_options(question, profile, rng)
        return {
            "kind": "answer_multi",
            "question_key": question["question_key"],
            "question_text": question["text"],
            "option_keys": [option["option_key"] for option in subset],
            "option_labels": [option["label"] for option in subset],
        }

    option = best_single_option(question, profile, rng)
    return {
        "kind": "answer",
        "question_key": question["question_key"],
        "question_text": question["text"],
        "option_key": option["option_key"],
        "option_label": option["label"],
    }


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "questionnaire_samples_cdp" / f"minority_supplement_{stamp}"


def load_signatures_from_jsonl(path: Path) -> set[str]:
    signatures: set[str] = set()
    if not path.exists():
        return signatures
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            signature = record.get("response_signature")
            if signature:
                signatures.add(str(signature))
    return signatures


def load_exclusions(paths: Sequence[str]) -> frozenset[str]:
    signatures: set[str] = set()
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            continue
        if path.suffix.lower() == ".jsonl":
            signatures.update(load_signatures_from_jsonl(path))
        else:
            signatures.update(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return frozenset(signatures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect minority-oriented Google Play questionnaire samples through the current CDP session."
    )
    parser.add_argument("--endpoint-url", default="http://127.0.0.1:9222")
    parser.add_argument("--target-substring", default="play.google.com/console")
    parser.add_argument("--page-index", type=int, default=None)
    parser.add_argument("--assume-ready", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--settle-ms", type=int, default=900)
    parser.add_argument("--max-steps-per-sample", type=int, default=250)
    parser.add_argument("--max-options-per-question", type=int, default=8)
    parser.add_argument("--max-multi-combinations", type=int, default=32)
    parser.add_argument("--fallback-email", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--risk-profile", choices=["low", "medium", "mixed"], default="mixed")
    parser.add_argument(
        "--accepted-ratings",
        nargs="*",
        default=None,
        help="Canonical target ratings to keep, such as 3+ 7+ 12+ 16+ 18+. Defaults depend on --risk-profile.",
    )
    parser.add_argument(
        "--exclude-existing-jsonl",
        action="append",
        default=["data/raw/real_20260611_142334.samples.jsonl"],
        help="Standard JSONL files whose response_signature values should be skipped.",
    )
    parser.add_argument(
        "--exclude-signatures",
        action="append",
        default=[],
        help="Files containing response signatures to skip, one per line.",
    )
    return parser.parse_args()


ORIGINAL_COLLECT_ONE_SAMPLE = sampler.collect_one_sample


async def collect_one_sample_with_attempt(*args, **kwargs):
    global CURRENT_ATTEMPT_INDEX, CURRENT_TARGET_RATING, CURRENT_TEMPLATE_ANSWERS, CURRENT_TEMPLATE_ID, CURRENT_MUTATION_QUESTION_KEY
    CURRENT_ATTEMPT_INDEX += 1
    CURRENT_TARGET_RATING = choose_target_rating()
    template = choose_current_template() or {}
    CURRENT_TEMPLATE_ANSWERS = dict(template.get("answers_json") or {})
    CURRENT_TEMPLATE_ID = str(template.get("sample_id") or f"attempt_{CURRENT_ATTEMPT_INDEX}")
    CURRENT_MUTATION_QUESTION_KEY = choose_mutation_question_key(CURRENT_TEMPLATE_ANSWERS)
    return await ORIGINAL_COLLECT_ONE_SAMPLE(*args, **kwargs)


async def async_main() -> None:
    args = parse_args()
    if not endpoint_available(args.endpoint_url):
        raise SystemExit(f"Could not connect to Chrome DevTools at {args.endpoint_url}.")

    global ACTIVE_PROFILE
    ACTIVE_PROFILE = args.risk_profile
    sampler.choose_action_for_question = choose_action_for_question
    sampler.collect_one_sample = collect_one_sample_with_attempt

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    exclusions = load_exclusions([*args.exclude_existing_jsonl, *args.exclude_signatures])
    global ACTIVE_ACCEPTED_RATINGS, ANSWER_POLICY, ANSWER_POLICY_RATING_COUNTS, ANSWER_POLICY_SOURCES, TEMPLATE_LIBRARY
    accepted_ratings = frozenset(args.accepted_ratings) if args.accepted_ratings else PROFILE_ACCEPTED_RATINGS[args.risk_profile]
    ACTIVE_ACCEPTED_RATINGS = accepted_ratings
    ANSWER_POLICY, ANSWER_POLICY_RATING_COUNTS, ANSWER_POLICY_SOURCES = load_answer_policy(accepted_ratings)
    TEMPLATE_LIBRARY = load_template_library(accepted_ratings)
    if exclusions:
        print(f"Loaded {len(exclusions)} exclusion signatures.")
    if accepted_ratings:
        print(f"Accepted ratings: {', '.join(sorted(accepted_ratings))}")
    if ANSWER_POLICY_SOURCES:
        print(f"Policy sources: {', '.join(ANSWER_POLICY_SOURCES)}")
        print(f"Policy rating counts: {dict(ANSWER_POLICY_RATING_COUNTS)}")
    if TEMPLATE_LIBRARY:
        template_counts = {rating: len(items) for rating, items in TEMPLATE_LIBRARY.items()}
        print(f"Template counts: {template_counts}")

    config = SampleConfig(
        endpoint_url=args.endpoint_url,
        target_substring=args.target_substring,
        output_dir=output_dir,
        sample_count=args.sample_count,
        settle_ms=args.settle_ms,
        max_steps_per_sample=args.max_steps_per_sample,
        max_options_per_question=args.max_options_per_question,
        max_multi_combinations=args.max_multi_combinations,
        fallback_email=args.fallback_email,
        page_index=args.page_index,
        assume_ready=args.assume_ready,
        seed=args.seed,
        resume=args.resume,
        exclude_signatures=exclusions,
        accepted_primary_ratings=accepted_ratings,
    )
    summary = await sampler.run_sampling(config)
    summary["risk_profile"] = args.risk_profile
    summary["exclusion_signature_count"] = len(exclusions)
    summary["accepted_primary_ratings"] = sorted(accepted_ratings)
    summary["policy_sources"] = ANSWER_POLICY_SOURCES
    summary["policy_rating_counts"] = dict(ANSWER_POLICY_RATING_COUNTS)
    summary["template_counts"] = {rating: len(items) for rating, items in TEMPLATE_LIBRARY.items()}
    sampler.write_json(output_dir / "minority_sampling_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
