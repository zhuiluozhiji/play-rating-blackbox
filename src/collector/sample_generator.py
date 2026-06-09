from __future__ import annotations

import itertools
import random
from typing import Any, Dict, Iterable, List, Sequence

from src.data.schema import default_question_schema


def load_question_schema(schema: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return schema or default_question_schema()


def _options(question: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(question.get("options") or [])


def _visible(question: Dict[str, Any], answers: Dict[str, Any]) -> bool:
    condition = question.get("condition_to_show")
    if not condition:
        return True
    for parent, allowed in condition.items():
        current = answers.get(parent)
        allowed_values = allowed if isinstance(allowed, list) else [allowed]
        if current not in allowed_values:
            return False
    return True


def _lowest_value(question: Dict[str, Any]) -> Any:
    opts = _options(question)
    if question.get("question_type") == "multi":
        return ["none"] if any(opt["value"] == "none" for opt in opts) else []
    return opts[0]["value"] if opts else "no"


def _highest_value(question: Dict[str, Any]) -> Any:
    opts = _options(question)
    if question.get("question_type") == "multi":
        values = [opt["value"] for opt in opts if opt["value"] != "none"]
        return values or ["none"]
    return max(opts, key=lambda opt: opt.get("risk_score", 0))["value"] if opts else "yes"


def normalize_visibility(schema: Dict[str, Any], answers: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for question in schema.get("questions", []):
        qid = question["question_id"]
        if _visible(question, normalized):
            normalized[qid] = answers.get(qid, _lowest_value(question))
        else:
            normalized[qid] = "not_visible"
    return normalized


def generate_baseline(schema: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
    baseline = normalize_visibility(
        schema,
        {question["question_id"]: _lowest_value(question) for question in schema.get("questions", [])},
    )
    variants = [dict(baseline) for _ in range(max(count, 1))]
    toggle_questions = [
        q for q in schema.get("questions", [])
        if q.get("theme") in {"interaction", "ugc_interaction"}
    ]
    for index, answers in enumerate(variants):
        if toggle_questions and index:
            question = toggle_questions[index % len(toggle_questions)]
            answers[question["question_id"]] = _highest_value(question)
    return variants[:count]


def generate_single_factor(schema: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
    baseline = generate_baseline(schema, 1)[0]
    samples: List[Dict[str, Any]] = []
    for question in schema.get("questions", []):
        for option in _options(question):
            value = option["value"]
            if value in {"no", "none", "not_visible"}:
                continue
            candidate = dict(baseline)
            candidate[question["question_id"]] = [value] if question.get("question_type") == "multi" else value
            samples.append(normalize_visibility(schema, candidate))
    while len(samples) < count and samples:
        samples.append(dict(samples[len(samples) % len(samples)]))
    return samples[:count]


def generate_tree_coverage(schema: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
    questions = schema.get("questions", [])
    option_values: List[List[Any]] = []
    for question in questions:
        values = [opt["value"] for opt in _options(question)]
        if question.get("question_type") == "multi":
            values = [["none"]] + [[v] for v in values if v != "none"]
        option_values.append(values[:4] or ["no"])

    samples: List[Dict[str, Any]] = []
    for combo in itertools.islice(itertools.product(*option_values), count * 3):
        raw = {
            question["question_id"]: value
            for question, value in zip(questions, combo)
        }
        normalized = normalize_visibility(schema, raw)
        if normalized not in samples:
            samples.append(normalized)
        if len(samples) >= count:
            break
    return samples[:count]


def generate_stratified_random(schema: Dict[str, Any], count: int, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    questions = schema.get("questions", [])
    themes = sorted({question.get("theme", "general") for question in questions})
    samples: List[Dict[str, Any]] = []
    for index in range(count):
        target_theme = themes[index % len(themes)] if themes else "general"
        raw: Dict[str, Any] = {}
        for question in questions:
            opts = _options(question)
            if not opts:
                raw[question["question_id"]] = "no"
                continue
            if question.get("theme") == target_theme:
                weights = [1 + opt.get("risk_score", 0) for opt in opts]
            else:
                weights = [3 if opt.get("risk_score", 0) == 0 else 1 for opt in opts]
            chosen = rng.choices(opts, weights=weights, k=1)[0]["value"]
            raw[question["question_id"]] = [chosen] if question.get("question_type") == "multi" and chosen != "none" else chosen
        samples.append(normalize_visibility(schema, raw))
    return samples


def generate_active_learning_seed(schema: Dict[str, Any], count: int, seed: int = 43) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    questions = schema.get("questions", [])
    medium_options = {
        q["question_id"]: [
            opt["value"] for opt in _options(q)
            if 1 <= opt.get("risk_score", 0) <= 3
        ]
        for q in questions
    }
    samples: List[Dict[str, Any]] = []
    for _ in range(count):
        raw = {q["question_id"]: _lowest_value(q) for q in questions}
        changed = rng.sample(questions, k=min(len(questions), rng.randint(2, 4)))
        for question in changed:
            values = medium_options.get(question["question_id"]) or [
                opt["value"] for opt in _options(question)
            ]
            chosen = rng.choice(values)
            raw[question["question_id"]] = [chosen] if question.get("question_type") == "multi" and chosen != "none" else chosen
        samples.append(normalize_visibility(schema, raw))
    return samples


def generate_samples(
    schema: Dict[str, Any],
    strategy: str,
    count: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    strategy = strategy.lower()
    if strategy == "baseline":
        return generate_baseline(schema, count)
    if strategy == "single_factor":
        return generate_single_factor(schema, count)
    if strategy == "tree_coverage":
        return generate_tree_coverage(schema, count)
    if strategy == "stratified_random":
        return generate_stratified_random(schema, count, seed)
    if strategy == "active_learning_seed":
        return generate_active_learning_seed(schema, count, seed)
    if strategy == "all":
        counts = {
            "baseline": max(1, int(count * 0.03)),
            "single_factor": max(1, int(count * 0.2)),
            "tree_coverage": max(1, int(count * 0.25)),
            "stratified_random": max(1, int(count * 0.34)),
        }
        used = sum(counts.values())
        counts["active_learning_seed"] = max(0, count - used)
        output: List[Dict[str, Any]] = []
        for name, n in counts.items():
            output.extend(generate_samples(schema, name, n, seed))
        return output[:count]
    raise ValueError(f"Unknown sampling strategy: {strategy}")
