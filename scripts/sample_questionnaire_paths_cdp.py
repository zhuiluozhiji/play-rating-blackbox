#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import ensure_dir, write_json

from probe_questionnaire_branches import (
    click_next_button,
    click_option,
    extract_snapshot,
    first_unanswered_question,
    nonempty_option_subsets,
    path_id,
    prepare_auxiliary_controls,
    set_multi_answer,
    stable_hash,
    summarize_path,
    wait_after_action,
)
from probe_questionnaire_branches_cdp import choose_page, endpoint_available, fetch_json


@dataclass
class SampleConfig:
    endpoint_url: str
    target_substring: str
    output_dir: Path
    sample_count: int
    settle_ms: int = 900
    max_steps_per_sample: int = 250
    max_options_per_question: int = 8
    max_multi_combinations: int = 32
    fallback_email: str = ""
    page_index: Optional[int] = None
    assume_ready: bool = False
    seed: Optional[int] = None
    resume: bool = False
    exclude_signatures: frozenset = frozenset()
    accepted_primary_ratings: frozenset = frozenset()


SUMMARY_PRIMARY_AUTHORITIES = (
    "IARC Generic",
    "Google Play",
)
SUMMARY_SKIP_LINES = {"warning", "Learn more"}
SUMMARY_STOP_LINES = {
    "check_circle",
    "Back",
    "Save",
    "Product updates",
    "Status dashboard",
    "Help",
}
PRIMARY_RATING_PATTERN = re.compile(r"(?P<age>3|7|12|16|18)\s*\+")


async def quick_wait_after_action(page: Any, settle_ms: int) -> None:
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=1200)
    except Exception:
        pass
    try:
        await page.wait_for_load_state("networkidle", timeout=1500)
    except Exception:
        pass
    await page.wait_for_timeout(max(150, settle_ms))


def normalize_line(value: str) -> str:
    return " ".join((value or "").split())


def dedupe_preserving_order(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


async def read_body_text(page: Any) -> str:
    try:
        return await page.locator("body").inner_text(timeout=5000)
    except Exception:
        return ""


async def click_button_label(page: Any, label: str) -> bool:
    escaped_label = label.replace("\\", "\\\\").replace("'", "\\'")
    try:
        clicked = await page.evaluate(
            f"""
            () => {{
              const normalize = (value) => (value || '').replace(/\\s+/g, ' ').trim();
              const elements = Array.from(document.querySelectorAll('button, [role="button"]'));
              for (const element of elements) {{
                const text = normalize(
                  element.innerText || element.textContent || element.getAttribute('aria-label') || ''
                );
                const disabled =
                  element.hasAttribute('disabled') || element.getAttribute('aria-disabled') === 'true';
                if (!disabled && text === '{escaped_label}') {{
                  element.dispatchEvent(new MouseEvent('mouseover', {{ bubbles: true }}));
                  element.dispatchEvent(new MouseEvent('mousedown', {{ bubbles: true }}));
                  element.click();
                  element.dispatchEvent(new MouseEvent('mouseup', {{ bubbles: true }}));
                  return true;
                }}
              }}
              return false;
            }}
            """
        )
        if clicked:
            return True
    except Exception:
        pass

    attempts = [
        page.get_by_role("button", name=label, exact=True),
        page.get_by_text(label, exact=True),
        page.get_by_text(label),
        page.locator(f"button:has-text('{label}')"),
    ]
    for locator in attempts:
        try:
            if await locator.count() > 0:
                await locator.first.click(timeout=3000, force=True)
                return True
        except Exception:
            continue
    return False


async def click_button_kind(page: Any, snapshot: Dict[str, Any], kind: str) -> bool:
    labels = [
        button["text"]
        for button in snapshot.get("buttons", [])
        if button.get("kind") == kind and not button.get("disabled")
    ]
    for label in dedupe_preserving_order(labels):
        if await click_button_label(page, label):
            return True
    return False


def is_category_step(snapshot: Dict[str, Any]) -> bool:
    return any(normalize_line(question.get("text", "")).lower() == "category" for question in snapshot.get("questions", []))


async def return_to_category_step(page: Any, settle_ms: int, max_back_steps: int = 4) -> Dict[str, Any]:
    snapshot = await extract_snapshot(page)
    if is_category_step(snapshot):
        return snapshot

    for _ in range(max_back_steps):
        clicked = await click_button_label(page, "Back")
        if not clicked:
            break
        await quick_wait_after_action(page, settle_ms)
        snapshot = await extract_snapshot(page)
        if is_category_step(snapshot):
            return snapshot
    return snapshot


async def wait_for_continue_after_save(page: Any, settle_ms: int, attempts: int = 8) -> Dict[str, Any]:
    snapshot = await extract_snapshot(page)
    if snapshot.get("can_continue"):
        return snapshot

    for _ in range(attempts):
        await page.wait_for_timeout(max(250, settle_ms))
        snapshot = await extract_snapshot(page)
        if snapshot.get("can_continue"):
            return snapshot
    return snapshot


def parse_rating_summary(body_text: str) -> Dict[str, Any]:
    lines = [normalize_line(line) for line in body_text.splitlines()]
    lines = [line for line in lines if line]

    ratings: List[Dict[str, Any]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("Rating authority:"):
            index += 1
            continue

        authority = normalize_line(line.split(":", 1)[1])
        block = {
            "authority": authority,
            "rating": "",
            "content_descriptors": [],
            "interactive_elements": [],
        }
        index += 1

        while index < len(lines):
            token = lines[index]
            if token.startswith("Rating authority:"):
                break
            if token in SUMMARY_STOP_LINES or token.startswith("If you save, changes will be saved"):
                break
            if token in SUMMARY_SKIP_LINES:
                index += 1
                continue

            if token == "Rating":
                index += 1
                while index < len(lines) and lines[index] in SUMMARY_SKIP_LINES:
                    index += 1
                if index < len(lines):
                    next_value = lines[index]
                    if (
                        next_value not in {"Rating", "Content descriptors", "Interactive elements"}
                        and not next_value.startswith("Rating authority:")
                        and next_value not in SUMMARY_STOP_LINES
                    ):
                        block["rating"] = next_value
                        index += 1
                continue

            if token == "Content descriptors":
                index += 1
                while index < len(lines):
                    next_value = lines[index]
                    if (
                        next_value == "Interactive elements"
                        or next_value.startswith("Rating authority:")
                        or next_value in SUMMARY_STOP_LINES
                        or next_value.startswith("If you save, changes will be saved")
                    ):
                        break
                    if next_value not in SUMMARY_SKIP_LINES:
                        block["content_descriptors"].append(next_value)
                    index += 1
                continue

            if token == "Interactive elements":
                index += 1
                while index < len(lines):
                    next_value = lines[index]
                    if (
                        next_value.startswith("Rating authority:")
                        or next_value in SUMMARY_STOP_LINES
                        or next_value.startswith("If you save, changes will be saved")
                    ):
                        break
                    if next_value not in SUMMARY_SKIP_LINES:
                        block["interactive_elements"].append(next_value)
                    index += 1
                continue

            index += 1

        block["content_descriptors"] = dedupe_preserving_order(block["content_descriptors"])
        block["interactive_elements"] = dedupe_preserving_order(block["interactive_elements"])
        if block["authority"] or block["rating"] or block["content_descriptors"] or block["interactive_elements"]:
            ratings.append(block)

    deduped_ratings: List[Dict[str, Any]] = []
    seen_blocks = set()
    for item in ratings:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in seen_blocks:
            continue
        seen_blocks.add(key)
        deduped_ratings.append(item)

    primary = None
    for authority in SUMMARY_PRIMARY_AUTHORITIES:
        primary = next((item for item in deduped_ratings if item["authority"] == authority), None)
        if primary is not None:
            break
    if primary is None and deduped_ratings:
        primary = deduped_ratings[0]

    return {
        "ratings": deduped_ratings,
        "primary_authority": primary["authority"] if primary else "",
        "primary_rating": primary["rating"] if primary else "",
        "primary_content_descriptors": primary["content_descriptors"] if primary else [],
        "primary_interactive_elements": primary["interactive_elements"] if primary else [],
        "contains_rating_authority": "Rating authority:" in body_text,
    }


async def save_and_extract_ratings(page: Any, settle_ms: int) -> Dict[str, Any]:
    questionnaire_snapshot = await extract_snapshot(page)
    if not questionnaire_snapshot.get("can_finalize"):
        return {
            "ok": False,
            "error": "Questionnaire page did not expose an enabled Save button.",
            "questionnaire_snapshot": questionnaire_snapshot,
        }

    saved = await click_button_kind(page, questionnaire_snapshot, "final")
    if not saved:
        return {
            "ok": False,
            "error": "Failed to click the questionnaire Save button.",
            "questionnaire_snapshot": questionnaire_snapshot,
        }

    await quick_wait_after_action(page, max(700, settle_ms))
    post_save_snapshot = await wait_for_continue_after_save(page, max(250, settle_ms // 2))
    if not post_save_snapshot.get("can_continue"):
        return {
            "ok": False,
            "error": "Next did not become enabled after Save.",
            "questionnaire_snapshot": questionnaire_snapshot,
            "post_save_snapshot": post_save_snapshot,
        }

    moved_to_summary = await click_next_button(page, post_save_snapshot)
    if not moved_to_summary:
        return {
            "ok": False,
            "error": "Failed to click Next after Save.",
            "questionnaire_snapshot": questionnaire_snapshot,
            "post_save_snapshot": post_save_snapshot,
        }

    await wait_after_action(page, max(1000, settle_ms))
    summary_snapshot = await extract_snapshot(page)
    summary_body_text = await read_body_text(page)
    parsed = parse_rating_summary(summary_body_text)
    if not parsed["ratings"]:
        return {
            "ok": False,
            "error": "Reached the summary page, but no rating blocks were parsed.",
            "questionnaire_snapshot": questionnaire_snapshot,
            "post_save_snapshot": post_save_snapshot,
            "summary_snapshot": summary_snapshot,
            "summary_body_excerpt": summary_body_text[:2000],
            "parsed": parsed,
        }

    return {
        "ok": True,
        "error": "",
        "questionnaire_snapshot": questionnaire_snapshot,
        "post_save_snapshot": post_save_snapshot,
        "summary_snapshot": summary_snapshot,
        "summary_body_fingerprint": stable_hash(summary_body_text),
        "summary_body_excerpt": summary_body_text[:2000],
        "parsed": parsed,
    }


def normalize_primary_rating(value: Any) -> str:
    text = str(value or "")
    match = PRIMARY_RATING_PATTERN.search(text)
    if not match:
        return ""
    return f"{match.group('age')}+"


def action_to_response(action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if action["kind"] == "answer":
        return {
            "question_key": action["question_key"],
            "question_text": action["question_text"],
            "question_type": "single",
            "option_keys": [action["option_key"]],
            "option_labels": [action["option_label"]],
        }
    if action["kind"] == "answer_multi":
        return {
            "question_key": action["question_key"],
            "question_text": action["question_text"],
            "question_type": "multi",
            "option_keys": action.get("option_keys", []),
            "option_labels": action.get("option_labels", []),
        }
    return None


def responses_from_path(path: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    responses = []
    for action in path:
        response = action_to_response(action)
        if response is not None:
            responses.append(response)
    return responses


def response_signature(responses: List[Dict[str, Any]]) -> str:
    normalized = [
        {
            "question_key": item["question_key"],
            "option_keys": sorted(item["option_keys"]),
        }
        for item in responses
    ]
    normalized.sort(key=lambda item: item["question_key"])
    return stable_hash(json.dumps(normalized, ensure_ascii=False, sort_keys=True))


def choose_action_for_question(
    question: Dict[str, Any],
    rng: random.Random,
    max_options_per_question: int,
    max_multi_combinations: int,
) -> Dict[str, Any]:
    options = question["options"][:max_options_per_question]
    if question.get("question_type") == "multi":
        current_selected = {option["option_key"] for option in options if option.get("selected")}
        subsets = [
            subset
            for subset in nonempty_option_subsets(options, max_multi_combinations)
            if {option["option_key"] for option in subset} != current_selected
        ]
        if not subsets:
            subsets = nonempty_option_subsets(options, max_multi_combinations)
        if not subsets:
            raise RuntimeError(f"No selectable subsets generated for multi question {question['question_key']}.")
        subset = rng.choice(subsets)
        return {
            "kind": "answer_multi",
            "question_key": question["question_key"],
            "question_text": question["text"],
            "option_keys": [option["option_key"] for option in subset],
            "option_labels": [option["label"] for option in subset],
        }

    unselected_options = [option for option in options if not option.get("selected")]
    candidate_options = unselected_options or options
    option = rng.choice(candidate_options)
    return {
        "kind": "answer",
        "question_key": question["question_key"],
        "question_text": question["text"],
        "option_key": option["option_key"],
        "option_label": option["label"],
    }


async def apply_action(page: Any, snapshot: Dict[str, Any], action: Dict[str, Any], settle_ms: int) -> None:
    if action["kind"] == "answer":
        question = next(
            (item for item in snapshot.get("questions", []) if item["question_key"] == action["question_key"]),
            None,
        )
        if question is None:
            raise RuntimeError(f"Question {action['question_key']} not found while applying action.")
        option = next(
            (item for item in question["options"] if item["option_key"] == action["option_key"]),
            None,
        )
        if option is None:
            raise RuntimeError(f"Option {action['option_key']} not found while applying action.")
        await click_option(page, question, option)
        await quick_wait_after_action(page, settle_ms)
        return

    if action["kind"] == "answer_multi":
        question = next(
            (item for item in snapshot.get("questions", []) if item["question_key"] == action["question_key"]),
            None,
        )
        if question is None:
            raise RuntimeError(f"Multi question {action['question_key']} not found while applying action.")
        await set_multi_answer(page, question, action.get("option_keys", []))
        await quick_wait_after_action(page, settle_ms)
        return

    raise RuntimeError(f"Unsupported action kind: {action['kind']}")


async def collect_one_sample(
    page: Any,
    start_url: str,
    sample_index: int,
    rng: random.Random,
    config: SampleConfig,
) -> Dict[str, Any]:
    # Guard against chrome-error URLs (tab crash / suspended)
    if "chrome-error" in (start_url or "") or "chrome-error" in (page.url or ""):
        raise RuntimeError(
            f"Page is showing a Chrome error page (url={page.url}). "
            "Reload the questionnaire page in Chrome and try again."
        )
    try:
        await page.goto(start_url, wait_until="domcontentloaded")
    except Exception:
        await page.goto(start_url)
    await wait_after_action(page, max(700, config.settle_ms))
    await return_to_category_step(page, config.settle_ms)

    path: List[Dict[str, Any]] = []
    seen_state_signatures: List[str] = []
    status = "complete"
    error = ""
    final_snapshot: Dict[str, Any] = {}
    rating_result: Dict[str, Any] = {
        "ok": False,
        "error": "",
        "ratings": [],
        "primary_authority": "",
        "primary_rating": "",
        "primary_content_descriptors": [],
        "primary_interactive_elements": [],
        "summary_url": "",
        "summary_title": "",
        "summary_state_signature": "",
        "summary_body_fingerprint": "",
        "summary_body_excerpt": "",
    }

    for step_index in range(config.max_steps_per_sample):
        snapshot = await extract_snapshot(page)
        final_snapshot = snapshot

        if (
            step_index == 0
            and not snapshot.get("questions")
            and not snapshot.get("can_continue")
            and not snapshot.get("can_finalize")
        ):
            await wait_after_action(page, max(700, config.settle_ms))
            snapshot = await extract_snapshot(page)
            final_snapshot = snapshot

        state_signature = snapshot.get("state_signature") or ""
        if state_signature:
            if state_signature in seen_state_signatures[-6:]:
                status = "loop_detected"
                error = f"Repeated recent state signature at step {step_index}."
                break
            seen_state_signatures.append(state_signature)

        next_question = first_unanswered_question(snapshot, path)
        if next_question is not None:
            action = choose_action_for_question(
                next_question,
                rng=rng,
                max_options_per_question=config.max_options_per_question,
                max_multi_combinations=config.max_multi_combinations,
            )
            await apply_action(page, snapshot, action, config.settle_ms)
            path.append(action)
            continue

        helper_actions = await prepare_auxiliary_controls(
            page,
            snapshot,
            fallback_email=config.fallback_email,
        )
        if helper_actions:
            await quick_wait_after_action(page, config.settle_ms)
            continue

        if snapshot.get("can_continue"):
            clicked = await click_next_button(page, snapshot)
            if not clicked:
                status = "continue_failed"
                error = f"Next button was expected but not clickable at step {step_index}."
                break
            path.append({"kind": "continue"})
            await quick_wait_after_action(page, config.settle_ms)
            continue

        if snapshot.get("can_finalize"):
            rating_attempt = await save_and_extract_ratings(page, config.settle_ms)
            parsed = rating_attempt.get("parsed", {})
            summary_snapshot = rating_attempt.get("summary_snapshot", {})
            rating_result = {
                "ok": bool(rating_attempt.get("ok")),
                "error": rating_attempt.get("error", ""),
                "ratings": parsed.get("ratings", []),
                "primary_authority": parsed.get("primary_authority", ""),
                "primary_rating": parsed.get("primary_rating", ""),
                "primary_content_descriptors": parsed.get("primary_content_descriptors", []),
                "primary_interactive_elements": parsed.get("primary_interactive_elements", []),
                "summary_url": summary_snapshot.get("url", page.url),
                "summary_title": summary_snapshot.get("title", ""),
                "summary_state_signature": summary_snapshot.get("state_signature", ""),
                "summary_body_fingerprint": rating_attempt.get("summary_body_fingerprint", ""),
                "summary_body_excerpt": rating_attempt.get("summary_body_excerpt", ""),
            }
            if rating_result["ok"]:
                status = "complete"
                error = ""
            else:
                status = "rating_extraction_failed"
                error = rating_result["error"] or "Questionnaire completed, but ratings could not be extracted."
            await return_to_category_step(page, config.settle_ms)
            break

        if snapshot.get("questions"):
            status = "blocked_with_questions"
            error = "Questions remained visible but there was no clickable Next button."
            await return_to_category_step(page, config.settle_ms)
            break

        if snapshot.get("errors"):
            status = "validation_error"
            error = " | ".join(snapshot["errors"])
            await return_to_category_step(page, config.settle_ms)
            break

        status = "unknown_terminal"
        error = "Reached a terminal-looking page without questions, Next, or Save."
        await return_to_category_step(page, config.settle_ms)
        break
    else:
        status = "max_steps_exceeded"
        error = f"Exceeded {config.max_steps_per_sample} steps."
        await return_to_category_step(page, config.settle_ms)

    responses = responses_from_path(path)
    rating_signature = ""
    if rating_result.get("ratings"):
        rating_signature = stable_hash(
            json.dumps(rating_result["ratings"], ensure_ascii=False, sort_keys=True)
        )
    sample = {
        "sample_id": f"sample_{sample_index:04d}",
        "status": status,
        "error": error,
        "path": path,
        "path_id": path_id(path),
        "path_summary": summarize_path(path),
        "responses": responses,
        "response_signature": response_signature(responses),
        "answer_count": len(responses),
        "continue_count": sum(1 for action in path if action["kind"] == "continue"),
        "final_url": final_snapshot.get("url", page.url),
        "final_title": final_snapshot.get("title", ""),
        "final_question_count": len(final_snapshot.get("questions", [])),
        "final_can_continue": bool(final_snapshot.get("can_continue")),
        "final_can_finalize": bool(final_snapshot.get("can_finalize")),
        "final_errors": final_snapshot.get("errors", []),
        "final_state_signature": final_snapshot.get("state_signature", ""),
        "rating_result": rating_result,
        "rating_signature": rating_signature,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    return sample


def _load_jsonl_signatures(jsonl_path: Path) -> set:
    """Read a JSONL file and return the set of response_signature values."""
    signatures: set = set()
    if not jsonl_path.exists():
        return signatures
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            sig = record.get("response_signature", "")
            if sig:
                signatures.add(sig)
        except json.JSONDecodeError:
            continue
    return signatures


def _load_jsonl_records(jsonl_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not jsonl_path.exists():
        return records
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _append_jsonl(jsonl_path: Path, record: Dict[str, Any]) -> None:
    """Append a single record to a JSONL file (atomic per line)."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()


async def run_sampling(config: SampleConfig) -> Dict[str, Any]:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is not installed in the current environment.") from exc

    ensure_dir(config.output_dir)

    # --- Resume: load already-collected signatures from JSONL ---
    samples_jsonl_path = config.output_dir / "samples.jsonl"
    existing_samples = _load_jsonl_records(samples_jsonl_path) if config.resume else []
    existing_signatures = {
        str(record.get("response_signature") or "")
        for record in existing_samples
        if str(record.get("response_signature") or "")
    }
    seen_signatures: set = set(config.exclude_signatures)
    seen_signatures.update(existing_signatures)
    existing_output_count = len(existing_samples)
    target_total_samples = config.sample_count
    target_new_samples = max(0, target_total_samples - existing_output_count) if config.resume else target_total_samples
    if config.resume:
        # Offset seed so RNG doesn't replay the same sequence
        effective_seed = (config.seed or 42) + existing_output_count
        print(f"Resume mode: loaded {existing_output_count} existing samples from {samples_jsonl_path}")
        print(f"  Seed adjusted: {config.seed} -> {effective_seed} (to skip already-explored paths)")
        if target_new_samples == 0:
            print(f"  Target already reached: existing={existing_output_count}, requested_total={target_total_samples}")
    else:
        effective_seed = config.seed
    if config.exclude_signatures:
        print(f"Exclusion list: {len(config.exclude_signatures)} extra signatures loaded")
    rng = random.Random(effective_seed)

    # --- In-memory buffer for the current session (used for final stats) ---
    samples: List[Dict[str, Any]] = []
    duplicate_skipped = 0
    rating_filtered_skipped = 0

    if target_new_samples > 0:
        async with async_playwright() as playwright:
            connect_target = config.endpoint_url
            if config.endpoint_url.startswith("http://") or config.endpoint_url.startswith("https://"):
                version_info = fetch_json(config.endpoint_url.rstrip("/") + "/json/version")
                connect_target = version_info.get("webSocketDebuggerUrl") or config.endpoint_url

            browser = await playwright.chromium.connect_over_cdp(connect_target, no_defaults=True)
            page = await choose_page(
                browser,
                config.target_substring,
                page_index=config.page_index,
                prompt_user=not config.assume_ready and config.page_index is None,
            )
            print(f"Using page: {page.url}")
            if not config.assume_ready:
                await asyncio.to_thread(input, "When the questionnaire page is ready, press Enter to start sampling...")

            await quick_wait_after_action(page, config.settle_ms)
            start_url = page.url
            if not start_url or start_url == "about:blank":
                raise RuntimeError("The selected page is still about:blank. Open the questionnaire page first.")

            collected_this_session = 0
            consecutive_duplicate_skips = 0
            recoveries = 0
            consecutive_errors = 0
            max_attempt_multiplier = 50 if config.accepted_primary_ratings else 10
            max_collection_attempts = max(1, target_new_samples) * max_attempt_multiplier
            max_duplicate_streak = max(25, max(1, target_new_samples) * 3)
            collection_attempts = 0
            # Stable total: initial known signatures + target new samples
            base_seen_count = len(seen_signatures)
            while collected_this_session < target_new_samples and collection_attempts < max_collection_attempts:
                progress_index = existing_output_count + collected_this_session + 1
                running_index = base_seen_count + collected_this_session + 1
                collection_attempts += 1
                try:
                    sample = await collect_one_sample(
                        page,
                        start_url=start_url,
                        sample_index=running_index,
                        rng=rng,
                        config=config,
                    )
                    consecutive_errors = 0  # reset on success
                except Exception as exc:
                    recoveries += 1
                    consecutive_errors += 1
                    collection_attempts -= 1  # don't count failures toward limit
                    error_msg = str(exc)
                    print(
                        f"[{progress_index:04d}/{target_total_samples:04d}] "
                        f"ERROR collecting sample: {exc} "
                        f"(recovery #{recoveries}, consecutive={consecutive_errors})"
                    )

                    # Fatal: browser/context closed — need full CDP reconnect
                    if "has been closed" in error_msg:
                        print(f"  Browser connection lost. Reconnecting to Chrome CDP...")
                        try:
                            await browser.close()
                        except Exception:
                            pass
                        await asyncio.sleep(3)
                        try:
                            connect_target = config.endpoint_url
                            if config.endpoint_url.startswith("http://") or config.endpoint_url.startswith("https://"):
                                version_info = fetch_json(config.endpoint_url.rstrip("/") + "/json/version")
                                connect_target = version_info.get("webSocketDebuggerUrl") or config.endpoint_url
                            browser = await playwright.chromium.connect_over_cdp(connect_target, no_defaults=True)
                            page = await choose_page(
                                browser,
                                config.target_substring,
                                page_index=config.page_index,
                                prompt_user=False,
                            )
                            print(f"  Reconnected to page: {page.url}")
                            start_url = page.url
                            consecutive_errors = 0
                        except Exception as reconnect_exc:
                            print(f"  Reconnect failed: {reconnect_exc}")
                            await asyncio.sleep(10)
                        continue

                    # Page-level error: try recovery
                    if consecutive_errors >= 10:
                        print(f"  Attempting aggressive recovery: reloading start URL...")
                        try:
                            await page.goto(start_url, wait_until="domcontentloaded", timeout=15000)
                            await asyncio.sleep(2)
                            await return_to_category_step(page, config.settle_ms)
                        except Exception:
                            pass
                    cool_down = min(consecutive_errors * 3, 60)
                    await asyncio.sleep(cool_down)
                    try:
                        await return_to_category_step(page, config.settle_ms)
                    except Exception:
                        pass
                    continue

                sig = sample.get("response_signature", "")
                primary_rating = sample.get("rating_result", {}).get("primary_rating", "")
                normalized_primary_rating = normalize_primary_rating(primary_rating)
                accepted_ratings = set(config.accepted_primary_ratings or [])
                if accepted_ratings and normalized_primary_rating not in accepted_ratings and primary_rating not in accepted_ratings:
                    rating_filtered_skipped += 1
                    print(
                        f"[{progress_index:04d}/{target_total_samples:04d}] "
                        f"SKIP rating={primary_rating or '-'} signature={sig or '-'} "
                        f"(outside target ratings, {rating_filtered_skipped} skipped so far)"
                    )
                    continue
                if sig and sig in seen_signatures:
                    duplicate_skipped += 1
                    consecutive_duplicate_skips += 1
                    print(
                        f"[{progress_index:04d}/{target_total_samples:04d}] "
                        f"SKIP duplicate signature={sig} "
                        f"(retrying, {duplicate_skipped} skipped so far)"
                    )
                    if consecutive_duplicate_skips % 10 == 0:
                        print("  Duplicate streak detected. Reloading the questionnaire page before retrying...")
                        try:
                            await page.goto(start_url, wait_until="domcontentloaded", timeout=15000)
                            await asyncio.sleep(1)
                            await return_to_category_step(page, config.settle_ms)
                        except Exception:
                            pass
                    if consecutive_duplicate_skips >= max_duplicate_streak:
                        print(
                            "  Stopping early because the sampler hit too many consecutive duplicates. "
                            "Try a different risk profile or a new seed."
                        )
                        break
                    continue

                consecutive_duplicate_skips = 0
                # --- Immediately persist to JSONL ---
                _append_jsonl(samples_jsonl_path, sample)
                seen_signatures.add(sig)
                samples.append(sample)
                collected_this_session += 1

                print(
                    f"[{existing_output_count + collected_this_session:04d}/{target_total_samples:04d}] "
                    f"status={sample['status']} answers={sample['answer_count']} "
                    f"signature={sig} "
                    f"rating={primary_rating or '-'}"
                )

                # --- Periodically write checkpoint ---
                if collected_this_session % 20 == 0:
                    checkpoint = {
                        "checkpoint_at": datetime.now().isoformat(timespec="seconds"),
                        "resume_mode": config.resume,
                        "existing_samples_before_resume": existing_output_count,
                        "session_collected": collected_this_session,
                        "output_total_samples": existing_output_count + collected_this_session,
                        "target_total_samples": target_total_samples,
                        "remaining_samples": max(0, target_total_samples - (existing_output_count + collected_this_session)),
                        "total_known_signatures": len(seen_signatures),
                        "seed": config.seed,
                    }
                    write_json(config.output_dir / "checkpoint.json", checkpoint)

            if duplicate_skipped:
                print(f"Deduplication: skipped {duplicate_skipped} duplicate response signatures.")
            if rating_filtered_skipped:
                print(f"Rating filter: skipped {rating_filtered_skipped} samples outside accepted ratings.")

            await browser.close()

    # --- Final stats and summary ---
    all_samples = existing_samples + samples
    status_counts = Counter(sample["status"] for sample in all_samples)
    signature_counts = Counter(sample["response_signature"] for sample in all_samples)
    primary_rating_counts = Counter(
        sample.get("rating_result", {}).get("primary_rating", "")
        for sample in all_samples
        if sample.get("status") == "complete" and sample.get("rating_result", {}).get("primary_rating")
    )
    session_status_counts = Counter(sample["status"] for sample in samples)
    session_primary_rating_counts = Counter(
        sample.get("rating_result", {}).get("primary_rating", "")
        for sample in samples
        if sample.get("status") == "complete" and sample.get("rating_result", {}).get("primary_rating")
    )
    completion_count = status_counts.get("complete", 0)
    output_total_samples = len(all_samples)
    session_completion_count = session_status_counts.get("complete", 0)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "endpoint_url": config.endpoint_url,
        "target_substring": config.target_substring,
        "resume_mode": config.resume,
        "existing_samples_before_resume": existing_output_count,
        "target_total_samples": target_total_samples,
        "target_new_samples": target_new_samples,
        "session_new_samples": len(samples),
        "output_total_samples": output_total_samples,
        "total_known_signatures": len(seen_signatures),
        "completion_count": completion_count,
        "completion_rate": completion_count / output_total_samples if output_total_samples else 0.0,
        "session_completion_count": session_completion_count,
        "session_completion_rate": session_completion_count / len(samples) if samples else 0.0,
        "unique_response_count": len(signature_counts),
        "unique_primary_rating_count": len(primary_rating_counts),
        "primary_rating_counts": dict(primary_rating_counts),
        "session_primary_rating_counts": dict(session_primary_rating_counts),
        "accepted_primary_ratings": sorted(config.accepted_primary_ratings),
        "rating_filtered_skipped": rating_filtered_skipped,
        "duplicate_skipped": duplicate_skipped,
        "seed": config.seed,
        "status_counts": dict(status_counts),
        "session_status_counts": dict(session_status_counts),
        "samples_jsonl_path": str(samples_jsonl_path),
        "samples_json_path": str(config.output_dir / "samples.json"),
        "summary_path": str(config.output_dir / "summary.json"),
    }

    write_json(config.output_dir / "samples.json", all_samples)
    write_json(config.output_dir / "summary.json", summary)
    checkpoint = {
        "checkpoint_at": datetime.now().isoformat(timespec="seconds"),
        "resume_mode": config.resume,
        "existing_samples_before_resume": existing_output_count,
        "session_collected": len(samples),
        "output_total_samples": output_total_samples,
        "target_total_samples": target_total_samples,
        "remaining_samples": max(0, target_total_samples - output_total_samples),
        "total_known_signatures": len(seen_signatures),
        "seed": config.seed,
        "completed": output_total_samples >= target_total_samples,
    }
    write_json(config.output_dir / "checkpoint.json", checkpoint)
    return summary


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "questionnaire_samples_cdp" / stamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample full questionnaire response paths through the current Google Play content-rating flow."
    )
    parser.add_argument("--endpoint-url", default="http://127.0.0.1:9222")
    parser.add_argument("--target-substring", default="play.google.com/console")
    parser.add_argument("--page-index", type=int, default=None)
    parser.add_argument("--assume-ready", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--settle-ms", type=int, default=900)
    parser.add_argument("--max-steps-per-sample", type=int, default=250)
    parser.add_argument("--max-options-per-question", type=int, default=8)
    parser.add_argument("--max-multi-combinations", type=int, default=32)
    parser.add_argument("--fallback-email", default="")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from an existing samples.jsonl in the output directory.")
    parser.add_argument("--exclude-signatures", default=None,
                        help="Path to a file containing response signatures to skip (one per line).")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    if not endpoint_available(args.endpoint_url):
        raise SystemExit(f"Could not connect to Chrome DevTools at {args.endpoint_url}.")

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    exclude_signatures: frozenset = frozenset()
    if args.exclude_signatures:
        exclude_path = Path(args.exclude_signatures)
        if exclude_path.exists():
            exclude_signatures = frozenset(
                line.strip() for line in exclude_path.read_text(encoding="utf-8").splitlines() if line.strip()
            )
            print(f"Loaded {len(exclude_signatures)} exclusion signatures from {exclude_path}")
    if args.resume:
        print(f"Resume mode enabled — will load existing samples from {output_dir / 'samples.jsonl'}")
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
        exclude_signatures=exclude_signatures,
    )
    summary = await run_sampling(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
