from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.collector.manual_ops import append_manual_action
from src.collector.questionnaire_mapper import page_fingerprint
from src.collector.result_parser import parse_age_rating, parse_list_after_heading, parse_region_ratings
from src.common import ensure_dir, project_path
from src.data.schema import Evidence, SampleRecord


STOP_PATTERNS = [
    "captcha",
    "verify it",
    "verification",
    "2-step verification",
    "two-step verification",
    "security check",
    "suspicious",
    "policy warning",
    "cannot proceed",
    "sign in",
    "login",
]


async def _safe_text(page: Any) -> str:
    try:
        return await page.locator("body").inner_text(timeout=5000)
    except Exception:
        return ""


def detect_blocking_condition(text: str) -> Optional[str]:
    lowered = (text or "").lower()
    for pattern in STOP_PATTERNS:
        if pattern in lowered:
            return pattern
    return None


async def save_evidence(page: Any, sample_id: str, screenshots_dir: str, html_dir: str) -> Evidence:
    screenshot_dir = ensure_dir(screenshots_dir)
    html_output_dir = ensure_dir(html_dir)
    screenshot_path = screenshot_dir / f"{sample_id}.png"
    html_path = html_output_dir / f"{sample_id}.html"
    try:
        await page.screenshot(path=str(screenshot_path), full_page=True)
    except Exception:
        screenshot_path = None
    try:
        content = await page.content()
        html_path.write_text(content, encoding="utf-8")
    except Exception:
        html_path = None
    return Evidence(
        screenshot_path=str(screenshot_path) if screenshot_path else None,
        html_path=str(html_path) if html_path else None,
    )


async def _click_option_by_text(page: Any, text: str) -> bool:
    if not text:
        return False
    selectors = [
        page.get_by_label(text, exact=True),
        page.get_by_text(text, exact=True),
        page.locator(f"text={text}"),
    ]
    for locator in selectors:
        try:
            if await locator.count() > 0:
                await locator.first.click(timeout=3000)
                return True
        except Exception:
            continue
    return False


async def fill_questionnaire(page: Any, answers: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    visible: List[str] = []
    question_map = {q["question_id"]: q for q in schema.get("questions", [])}
    for question_id, value in answers.items():
        if value == "not_visible":
            continue
        question = question_map.get(question_id, {})
        values = value if isinstance(value, list) else [value]
        option_labels = {
            opt.get("value"): opt.get("label", opt.get("value"))
            for opt in question.get("options", [])
        }
        clicked_any = False
        for item in values:
            if item in {"none", "not_visible"}:
                continue
            label = option_labels.get(item, str(item))
            clicked_any = await _click_option_by_text(page, label) or clicked_any
        if clicked_any:
            visible.append(question_id)
    return visible


async def click_submit(page: Any) -> bool:
    labels = ["Submit", "Save", "Next", "提交", "保存", "继续", "完成"]
    for label in labels:
        try:
            locator = page.get_by_role("button", name=label)
            if await locator.count() > 0:
                await locator.first.click(timeout=5000)
                return True
        except Exception:
            continue
        try:
            locator = page.get_by_text(label, exact=True)
            if await locator.count() > 0:
                await locator.first.click(timeout=5000)
                return True
        except Exception:
            continue
    return False


async def submit_sample(
    page: Any,
    answers: Dict[str, Any],
    schema: Dict[str, Any],
    strategy: str,
    submit: bool,
    screenshots_dir: str,
    html_dir: str,
    manual_ops_path: str,
) -> SampleRecord:
    record = SampleRecord(strategy=strategy, answers_json=answers)
    text = await _safe_text(page)
    blocker = detect_blocking_condition(text)
    if blocker:
        record.status = "blocked"
        record.error = f"blocking condition detected: {blocker}"
        record.evidence = await save_evidence(page, record.sample_id, screenshots_dir, html_dir)
        append_manual_action(
            "采集被安全检查阻断",
            f"页面疑似出现 `{blocker}`，脚本已停止继续提交。",
            path=manual_ops_path,
            sample_id=record.sample_id,
        )
        return record

    record.questionnaire_version = page_fingerprint(text)
    record.visible_questions = await fill_questionnaire(page, answers, schema)
    record.skipped_questions = [
        q["question_id"] for q in schema.get("questions", [])
        if answers.get(q["question_id"]) == "not_visible"
    ]
    record.evidence = await save_evidence(page, record.sample_id, screenshots_dir, html_dir)

    if not submit:
        record.status = "dry_run"
        record.submit_status = "not_submitted"
        return record

    clicked = await click_submit(page)
    if not clicked:
        record.status = "blocked"
        record.error = "submit button not found"
        append_manual_action(
            "未找到提交按钮",
            "脚本无法定位提交/保存/继续按钮，需要人工确认页面结构。",
            path=manual_ops_path,
            sample_id=record.sample_id,
        )
        return record

    await page.wait_for_timeout(3000)
    result_text = await _safe_text(page)
    blocker = detect_blocking_condition(result_text)
    if blocker:
        record.status = "blocked"
        record.error = f"blocking condition after submit: {blocker}"
        append_manual_action(
            "提交后出现阻断页面",
            f"提交后页面疑似出现 `{blocker}`，需要人工确认。",
            path=manual_ops_path,
            sample_id=record.sample_id,
        )
        return record

    record.submit_status = "submitted"
    record.result_age_rating = parse_age_rating(result_text)
    record.result_region_ratings = parse_region_ratings(result_text)
    record.content_descriptors = parse_list_after_heading(
        result_text, ["Content descriptors", "内容描述符", "Content Descriptors"]
    )
    record.interactive_elements = parse_list_after_heading(
        result_text, ["Interactive elements", "互动元素", "Interactive Elements"]
    )
    record.evidence = await save_evidence(page, record.sample_id, screenshots_dir, html_dir)

    if record.result_age_rating:
        record.status = "success"
    else:
        record.status = "parse_error"
        record.error = "result age rating not parsed"
        append_manual_action(
            "结果页解析失败",
            "提交后未能解析年龄分级，需要人工查看截图/页面并补录。",
            path=manual_ops_path,
            sample_id=record.sample_id,
        )
    return record
