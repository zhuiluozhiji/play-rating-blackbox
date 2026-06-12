#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collector.browser_session import BrowserConfig, launch_persistent_context
from src.collector.key_reader import read_credentials
from src.common import ensure_dir, write_json


QUESTION_EXTRACTION_JS = r"""
() => {
  const normalize = (value) => (value || "").replace(/\s+/g, " ").trim();
  const formRows = Array.from(document.querySelectorAll('console-form-row .form-row[role="group"], .form-row[role="group"]'));
  const sortedRows = Array.from(new Set(formRows)).sort((left, right) => {
    if (left === right) {
      return 0;
    }
    const relation = left.compareDocumentPosition(right);
    if (relation & Node.DOCUMENT_POSITION_FOLLOWING) {
      return -1;
    }
    if (relation & Node.DOCUMENT_POSITION_PRECEDING) {
      return 1;
    }
    return 0;
  });

  const labelForControl = (node) => {
    const candidates = [];
    const aria = normalize(node.getAttribute('aria-label') || "");
    if (aria) {
      candidates.push(aria);
    }
    const labelledBy = node.getAttribute('aria-labelledby');
    if (labelledBy) {
      const parts = labelledBy
        .split(/\s+/)
        .map((id) => document.getElementById(id))
        .filter(Boolean)
        .map((item) => normalize(item.innerText || item.textContent || ""));
      candidates.push(...parts);
    }
    if (node.labels) {
      for (const label of Array.from(node.labels)) {
        candidates.push(normalize(label.innerText || label.textContent || ""));
      }
    }
    const closestLabel = node.closest('label');
    if (closestLabel) {
      candidates.push(normalize(closestLabel.innerText || closestLabel.textContent || ""));
    }
    const parentLabel = node.parentElement ? normalize(node.parentElement.innerText || "") : "";
    if (parentLabel) {
      candidates.push(parentLabel);
    }
    const ownText = normalize(node.innerText || node.textContent || "");
    if (ownText) {
      candidates.push(ownText);
    }
    const filtered = candidates
      .map((text) => normalize(text))
      .filter((text) => text && text.length <= 200);
    return filtered[0] || "";
  };

  const isChecked = (node) => {
    if (!node) {
      return false;
    }
    if (typeof node.checked === 'boolean') {
      return node.checked;
    }
    const ariaChecked = node.getAttribute('aria-checked');
    if (ariaChecked === 'true') {
      return true;
    }
    if (ariaChecked === 'false') {
      return false;
    }
    const host = node.closest('material-checkbox, mat-checkbox, material-radio, mat-radio-button');
    if (host) {
      const hostAria = host.getAttribute('aria-checked');
      if (hostAria === 'true') {
        return true;
      }
      if (hostAria === 'false') {
        return false;
      }
      const classes = normalize(host.className || '');
      if (/\b(selected|checked|mdc-checkbox--selected|mat-mdc-checkbox-checked)\b/i.test(classes)) {
        return true;
      }
    }
    return false;
  };

  const firstDebugId = (element) => {
    if (!element) {
      return '';
    }
    const selfDebug = normalize(element.getAttribute('debug-id') || '');
    if (selfDebug && selfDebug !== 'form-row-title-text') {
      return selfDebug;
    }
    const candidate =
      element.querySelector('[debug-id]:not([debug-id="form-row-title-text"])') ||
      element.querySelector('[debug-id]');
    return candidate ? normalize(candidate.getAttribute('debug-id') || '') : '';
  };

  const rowLabel = (row) => {
    const title = row.querySelector('[debug-id="form-row-title-text"]');
    const ariaLabel = normalize(row.getAttribute('aria-label') || '');
    const titleText = normalize(title ? title.innerText || title.textContent || '' : '');
    return titleText || ariaLabel;
  };

  const dedupeLines = (text) => Array.from(
    new Set(
      (text || '')
        .split(/\n+/)
        .map((line) => normalize(line))
        .filter(Boolean)
    )
  );

  const results = [];
  const auxiliaryControls = [];
  const errorTexts = [];
  let questionOrdinal = 0;
  let auxiliaryOrdinal = 0;

  for (const panel of Array.from(document.querySelectorAll('error-panel'))) {
    const text = normalize(panel.innerText || panel.textContent || '');
    if (text) {
      errorTexts.push(text.slice(0, 400));
    }
  }

  for (const row of sortedRows) {
    const rawText = normalize(row.innerText || "");
    if (!rawText) {
      continue;
    }
    const optionNodes = Array.from(
      row.querySelectorAll('[role="radio"], [role="checkbox"], input[type="radio"], input[type="checkbox"]')
    );
    const inputNodes = Array.from(
      row.querySelectorAll('input:not([type="radio"]):not([type="checkbox"]):not([type="hidden"]), textarea, select')
    );
    const radioNodes = optionNodes.filter((node) => {
      const role = (node.getAttribute('role') || node.type || node.tagName || '').toLowerCase();
      return role.includes('radio');
    });
    const checkboxNodes = optionNodes.filter((node) => {
      const role = (node.getAttribute('role') || node.type || node.tagName || '').toLowerCase();
      return role.includes('checkbox');
    });

    const probeId = `branch-probe-row-${questionOrdinal + auxiliaryOrdinal}`;
    row.setAttribute('data-branch-probe-id', probeId);

    const optionMap = new Map();
    let hasRadio = false;
    let hasCheckbox = false;

    for (const node of optionNodes) {
      const role = (node.getAttribute('role') || node.type || node.tagName || "").toLowerCase();
      if (role.includes('radio')) {
        hasRadio = true;
      }
      if (role.includes('checkbox')) {
        hasCheckbox = true;
      }
      const label = labelForControl(node);
      if (!label) {
        continue;
      }
      if (!optionMap.has(label)) {
        optionMap.set(label, {
          label,
          role,
          selected: isChecked(node),
        });
      }
    }

    if (optionMap.size > 0 && (hasRadio || checkboxNodes.length > 1)) {
      const lines = dedupeLines(row.innerText || '');
      const optionLabels = new Set(Array.from(optionMap.keys()));
      let questionText = rowLabel(row) || lines.find((line) => !optionLabels.has(line)) || lines[0] || rawText;
      if (optionLabels.has(questionText) && rawText.length > questionText.length) {
        questionText = rawText;
      }
      questionText = normalize(questionText).slice(0, 500);

      results.push({
        probe_id: probeId,
        container_ordinal: questionOrdinal,
        stable_id: firstDebugId(row),
        text: questionText,
        raw_text: rawText.slice(0, 1200),
        question_type: hasCheckbox ? 'multi' : hasRadio ? 'single' : 'unknown',
        options: Array.from(optionMap.values()),
      });
      questionOrdinal += 1;
      continue;
    }

    if (checkboxNodes.length === 1) {
      const checkbox = checkboxNodes[0];
      auxiliaryControls.push({
        probe_id: probeId,
        control_ordinal: auxiliaryOrdinal,
        control_type: 'checkbox',
        text: (rowLabel(row) || labelForControl(checkbox) || rawText).slice(0, 500),
        raw_text: rawText.slice(0, 1200),
        option_label: (labelForControl(checkbox) || rowLabel(row) || rawText).slice(0, 500),
        checked: isChecked(checkbox),
        debug_id: firstDebugId(row),
      });
      auxiliaryOrdinal += 1;
      continue;
    }

    if (inputNodes.length > 0) {
      const input = inputNodes[0];
      const inputType = normalize((input.getAttribute('type') || input.tagName || 'text').toLowerCase()) || 'text';
      const currentValue = `${input.value || input.getAttribute('value') || ''}`;
      auxiliaryControls.push({
        probe_id: probeId,
        control_ordinal: auxiliaryOrdinal,
        control_type: inputType,
        text: (rowLabel(row) || labelForControl(input) || rawText).slice(0, 500),
        raw_text: rawText.slice(0, 1200),
        placeholder: normalize(input.getAttribute('placeholder') || ''),
        value_present: currentValue.length > 0,
        value_length: currentValue.length,
        debug_id: firstDebugId(row),
      });
      auxiliaryOrdinal += 1;
    }
  }

  const genericQuestionContainers = Array.from(
    document.querySelectorAll('question[debug-id="question"], [debug-iarc-question-id]')
  ).sort((left, right) => {
    if (left === right) {
      return 0;
    }
    const relation = left.compareDocumentPosition(right);
    if (relation & Node.DOCUMENT_POSITION_FOLLOWING) {
      return -1;
    }
    if (relation & Node.DOCUMENT_POSITION_PRECEDING) {
      return 1;
    }
    return 0;
  });

  const filteredGenericContainers = [];
  for (const container of genericQuestionContainers) {
    if (container.closest('.form-row[role="group"]')) {
      continue;
    }
    if (filteredGenericContainers.some((parent) => parent.contains(container))) {
      continue;
    }
    filteredGenericContainers.push(container);
  }

  for (const container of filteredGenericContainers) {
    const rawText = normalize(container.innerText || "");
    if (!rawText) {
      continue;
    }
    const optionNodes = Array.from(
      container.querySelectorAll('[role="radio"], [role="checkbox"], input[type="radio"], input[type="checkbox"]')
    );
    if (optionNodes.length === 0) {
      continue;
    }

    const optionMap = new Map();
    let hasRadio = false;
    let hasCheckbox = false;

    for (const node of optionNodes) {
      const role = (node.getAttribute('role') || node.type || node.tagName || "").toLowerCase();
      if (role.includes('radio')) {
        hasRadio = true;
      }
      if (role.includes('checkbox')) {
        hasCheckbox = true;
      }
      const label = labelForControl(node);
      if (!label) {
        continue;
      }
      if (!optionMap.has(label)) {
        optionMap.set(label, {
          label,
          role,
          selected: isChecked(node),
        });
      }
    }

    if (optionMap.size === 0) {
      continue;
    }

    const lines = dedupeLines(container.innerText || '');
    const optionLabels = new Set(Array.from(optionMap.keys()));
    let questionText =
      lines.find((line) => !optionLabels.has(line) && !/^learn more$/i.test(line)) ||
      lines[0] ||
      rawText;
    questionText = normalize(questionText.replace(/\s*Learn more$/i, "")).slice(0, 500);

    const probeId = `branch-probe-question-${questionOrdinal}`;
    container.setAttribute('data-branch-probe-id', probeId);

    results.push({
      probe_id: probeId,
      container_ordinal: questionOrdinal,
      stable_id: normalize(container.getAttribute('debug-iarc-question-id') || '') || firstDebugId(container),
      text: questionText,
      raw_text: rawText.slice(0, 1200),
      question_type: hasCheckbox ? 'multi' : hasRadio ? 'single' : 'unknown',
      options: Array.from(optionMap.values()),
    });
    questionOrdinal += 1;
  }

  const buttons = Array.from(document.querySelectorAll('button, [role="button"]'))
    .map((node) => ({
      text: normalize(node.innerText || node.textContent || node.getAttribute('aria-label') || ""),
      disabled: node.hasAttribute('disabled') || node.getAttribute('aria-disabled') === 'true',
    }))
    .filter((item) => item.text && item.text.length <= 120);

  const headings = Array.from(document.querySelectorAll('h1, h2, h3, [role="heading"], legend'))
    .map((node) => normalize(node.innerText || node.textContent || ""))
    .filter(Boolean)
    .slice(0, 10);

  return {
    url: location.href,
    title: document.title || "",
    headings,
    buttons,
    questions: results,
    auxiliary_controls: auxiliaryControls,
    errors: Array.from(new Set(errorTexts)),
  };
}
"""

NEXT_KEYWORDS = (
    "next",
    "continue",
    "review",
    "start",
    "proceed",
    "ok",
    "got it",
    "下一步",
    "继续",
    "下一页",
    "下一",
    "开始",
    "确认",
)
FINAL_KEYWORDS = (
    "submit",
    "save",
    "finish",
    "done",
    "publish",
    "complete",
    "提交",
    "保存",
    "完成",
    "发布",
)
DEFAULT_CONSOLE_URL = "https://play.google.com/console/developers"


class ProbeError(RuntimeError):
    pass


@dataclass
class ProbeConfig:
    start_url: str
    output_dir: Path
    max_states: int = 300
    max_options_per_question: int = 8
    max_multi_combinations: int = 32
    settle_ms: int = 1200
    screenshot: bool = True
    fallback_email: str = ""


def normalize_text(value: str) -> str:
    return " ".join((value or "").split())


def stable_hash(payload: str) -> str:
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:12]


def question_key(
    text: str,
    options: Optional[List[Dict[str, Any]]] = None,
    stable_id: str = "",
) -> str:
    parts = [normalize_text(text).lower()]
    if stable_id:
        parts.insert(0, normalize_text(stable_id).lower())
    if options:
        option_parts = []
        for option in options:
            option_parts.append(
                f"{normalize_text(option.get('role') or '').lower()}:{normalize_text(option.get('label') or '').lower()}"
            )
        parts.append("|".join(option_parts))
    return f"q_{stable_hash('||'.join(parts))}"


def option_key(text: str) -> str:
    return f"o_{stable_hash(normalize_text(text).lower())}"


def control_key(control_type: str, text: str) -> str:
    return f"c_{stable_hash(f'{control_type}|{normalize_text(text).lower()}')}"


def path_id(path: List[Dict[str, Any]]) -> str:
    payload = json.dumps(path, ensure_ascii=False, separators=(",", ":"))
    return f"path_{stable_hash(payload)}"


def classify_button(text: str) -> str:
    lowered = normalize_text(text).lower()
    cleaned = re.sub(r"\s+", " ", lowered).strip()
    final_labels = {keyword.lower() for keyword in FINAL_KEYWORDS}
    next_labels = {keyword.lower() for keyword in NEXT_KEYWORDS}
    if cleaned in final_labels or any(cleaned.startswith(label + " ") for label in final_labels):
        return "final"
    if cleaned in next_labels or any(cleaned.startswith(label + " ") for label in next_labels):
        return "next"
    return "other"


def action_answers_question(action: Dict[str, Any], question_key_value: str) -> bool:
    return action.get("kind") in {"answer", "answer_multi"} and action.get("question_key") == question_key_value


def nonempty_option_subsets(options: List[Dict[str, Any]], max_combinations: int) -> List[List[Dict[str, Any]]]:
    subset_catalog: List[List[Dict[str, Any]]] = []
    option_count = len(options)
    if option_count == 0:
        return subset_catalog

    full_masks = list(range(1, 1 << option_count))
    full_masks.sort(key=lambda mask: (mask.bit_count(), mask))
    for mask in full_masks:
        subset = [options[index] for index in range(option_count) if mask & (1 << index)]
        subset_catalog.append(subset)

    if len(subset_catalog) <= max_combinations:
        return subset_catalog

    prioritized_masks: List[int] = []
    seen_masks = set()

    def push_mask(mask: int) -> None:
        if mask <= 0 or mask in seen_masks:
            return
        seen_masks.add(mask)
        prioritized_masks.append(mask)

    for index in range(option_count):
        push_mask(1 << index)
    push_mask((1 << option_count) - 1)

    for subset_size in range(2, option_count):
        for mask in full_masks:
            if mask.bit_count() == subset_size:
                push_mask(mask)
                if len(prioritized_masks) >= max_combinations:
                    break
        if len(prioritized_masks) >= max_combinations:
            break

    truncated_catalog = [
        [options[index] for index in range(option_count) if mask & (1 << index)]
        for mask in prioritized_masks[:max_combinations]
    ]
    return truncated_catalog


async def prompt_continue(message: str) -> None:
    await asyncio.to_thread(input, message)


async def wait_after_action(page: Any, settle_ms: int) -> None:
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=2000)
    except Exception:
        pass
    try:
        await page.wait_for_load_state("networkidle", timeout=10000)
    except Exception:
        pass

    deadline = monotonic() + max(4.0, settle_ms / 1000 + 1.5)
    stable_hits = 0
    while monotonic() < deadline:
        try:
            body_text = normalize_text(await page.locator("body").inner_text(timeout=1000))
        except Exception:
            body_text = ""
        lowered = body_text.lower()
        looks_loading = (
            not body_text
            or "loading google play console" in lowered
            or lowered == "loading"
        )
        if not looks_loading and len(body_text) >= 40:
            stable_hits += 1
            if stable_hits >= 2:
                break
        else:
            stable_hits = 0
        await page.wait_for_timeout(600)
    await page.wait_for_timeout(settle_ms)


async def extract_snapshot(page: Any) -> Dict[str, Any]:
    raw = await page.evaluate(QUESTION_EXTRACTION_JS)
    body_text = ""
    try:
        body_text = await page.locator("body").inner_text(timeout=2000)
    except Exception:
        body_text = ""

    questions = []
    for item in raw.get("questions", []):
        text = normalize_text(item.get("text") or item.get("raw_text") or "")
        if not text:
            continue
        options = []
        seen_options = set()
        for option in item.get("options", []):
            label = normalize_text(option.get("label") or "")
            if not label or label in seen_options:
                continue
            seen_options.add(label)
            options.append(
                {
                    "option_key": option_key(label),
                    "label": label,
                    "role": normalize_text(option.get("role") or ""),
                    "selected": bool(option.get("selected")),
                }
            )
        if not options:
            continue
        questions.append(
            {
                "question_key": question_key(text, options, normalize_text(item.get("stable_id") or "")),
                "probe_id": item.get("probe_id"),
                "container_ordinal": item.get("container_ordinal"),
                "stable_id": normalize_text(item.get("stable_id") or ""),
                "text": text,
                "raw_text": normalize_text(item.get("raw_text") or "")[:1200],
                "question_type": item.get("question_type") or "unknown",
                "options": options[:20],
            }
        )

    auxiliary_controls = []
    for item in raw.get("auxiliary_controls", []):
        text = normalize_text(item.get("text") or item.get("raw_text") or "")
        if not text:
            continue
        auxiliary_controls.append(
            {
                "control_key": control_key(item.get("control_type") or "unknown", text),
                "probe_id": item.get("probe_id"),
                "control_ordinal": item.get("control_ordinal"),
                "control_type": normalize_text(item.get("control_type") or "unknown"),
                "text": text,
                "raw_text": normalize_text(item.get("raw_text") or "")[:1200],
                "option_label": normalize_text(item.get("option_label") or ""),
                "placeholder": normalize_text(item.get("placeholder") or ""),
                "checked": bool(item.get("checked")),
                "value_present": bool(item.get("value_present")),
                "value_length": int(item.get("value_length") or 0),
                "debug_id": normalize_text(item.get("debug_id") or ""),
            }
        )

    buttons = []
    for button in raw.get("buttons", []):
        text = normalize_text(button.get("text") or "")
        if not text:
            continue
        buttons.append(
            {
                "text": text,
                "kind": classify_button(text),
                "disabled": bool(button.get("disabled")),
            }
        )

    headings = [normalize_text(item) for item in raw.get("headings", []) if normalize_text(item)]
    errors = [normalize_text(item)[:400] for item in raw.get("errors", []) if normalize_text(item)]
    signature_payload = {
        "url": raw.get("url", ""),
        "headings": headings,
        "questions": [
            {
                "question_key": question["question_key"],
                "selected_option_keys": [
                    option["option_key"] for option in question["options"] if option.get("selected")
                ],
                "option_keys": [option["option_key"] for option in question["options"]],
            }
            for question in questions
        ],
        "auxiliary_controls": [
            {
                "control_key": item["control_key"],
                "control_type": item["control_type"],
                "checked": item["checked"],
                "value_present": item["value_present"],
                "value_length": item["value_length"],
            }
            for item in auxiliary_controls
        ],
        "buttons": [button["text"] for button in buttons],
        "errors": errors,
    }
    return {
        "url": raw.get("url") or page.url,
        "title": raw.get("title") or "",
        "headings": headings,
        "questions": questions,
        "auxiliary_controls": auxiliary_controls,
        "buttons": buttons,
        "errors": errors,
        "body_fingerprint": stable_hash(body_text),
        "state_signature": stable_hash(json.dumps(signature_payload, ensure_ascii=False, sort_keys=True)),
        "can_continue": any(button["kind"] == "next" and not button["disabled"] for button in buttons),
        "can_finalize": any(button["kind"] == "final" and not button["disabled"] for button in buttons),
    }


def answered_question_keys(path: List[Dict[str, Any]]) -> List[str]:
    return [
        action["question_key"]
        for action in path
        if action.get("kind") in {"answer", "answer_multi"}
    ]


def first_unanswered_question(snapshot: Dict[str, Any], path: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    answered = set(answered_question_keys(path))
    for question in snapshot.get("questions", []):
        if question["question_key"] not in answered:
            return question
    return None


async def click_option(page: Any, question: Dict[str, Any], option: Dict[str, Any]) -> None:
    probe_id = question.get("probe_id")
    if not probe_id:
        raise ProbeError(f"Question {question['question_key']} has no probe id.")

    option_label = option["label"]
    try:
        clicked = await page.evaluate(
            """
            ({ probeId, optionLabel }) => {
              const normalize = (value) => (value || '').replace(/\\s+/g, ' ').trim();
              const labelForControl = (node) => {
                const candidates = [];
                const aria = normalize(node.getAttribute('aria-label') || '');
                if (aria) candidates.push(aria);
                const labelledBy = node.getAttribute('aria-labelledby');
                if (labelledBy) {
                  const parts = labelledBy
                    .split(/\\s+/)
                    .map((id) => document.getElementById(id))
                    .filter(Boolean)
                    .map((item) => normalize(item.innerText || item.textContent || ''));
                  candidates.push(...parts);
                }
                if (node.labels) {
                  for (const label of Array.from(node.labels)) {
                    candidates.push(normalize(label.innerText || label.textContent || ''));
                  }
                }
                const closestLabel = node.closest('label');
                if (closestLabel) {
                  candidates.push(normalize(closestLabel.innerText || closestLabel.textContent || ''));
                }
                const parentText = node.parentElement ? normalize(node.parentElement.innerText || '') : '';
                if (parentText) candidates.push(parentText);
                const ownText = normalize(node.innerText || node.textContent || '');
                if (ownText) candidates.push(ownText);
                return candidates.find(Boolean) || '';
              };

              const container = document.querySelector(`[data-branch-probe-id="${probeId}"]`);
              if (!container) {
                return false;
              }
              const controls = Array.from(
                container.querySelectorAll('[role="radio"], [role="checkbox"], input[type="radio"], input[type="checkbox"], mat-radio-button, mat-checkbox')
              );
              for (const control of controls) {
                if (labelForControl(control) !== optionLabel) {
                  continue;
                }
                const clickable = control.closest('label, mat-radio-button, mat-checkbox, [role="radio"], [role="checkbox"]') || control;
                clickable.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
                clickable.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
                clickable.click();
                clickable.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
                return true;
              }
              return false;
            }
            """,
            {"probeId": probe_id, "optionLabel": option_label},
        )
        if clicked:
            return
    except Exception as exc:
        last_error = exc
    else:
        last_error = None

    container = page.locator(f"[data-branch-probe-id='{probe_id}']").first
    attempts = [
        container.get_by_label(option_label, exact=True),
        container.get_by_text(option_label, exact=True),
        container.get_by_text(option_label),
        page.get_by_label(option_label, exact=True),
        page.get_by_text(option_label, exact=True),
    ]

    for locator in attempts:
        try:
            if await locator.count() > 0:
                await locator.first.click(timeout=3000, force=True)
                return
        except Exception as exc:
            last_error = exc
            continue

    raise ProbeError(
        f"Failed to click option '{option_label}' for question '{question['text']}'."
    ) from last_error


async def set_multi_answer(page: Any, question: Dict[str, Any], selected_option_keys: List[str]) -> None:
    desired_keys = set(selected_option_keys)
    current_snapshot = await extract_snapshot(page)
    current_question = next(
        (item for item in current_snapshot.get("questions", []) if item["question_key"] == question["question_key"]),
        None,
    )
    if current_question is None:
        raise ProbeError(f"Multi-select question {question['question_key']} is not visible.")

    current_options = {item["option_key"]: item for item in current_question.get("options", [])}
    missing_keys = desired_keys - set(current_options.keys())
    if missing_keys:
        raise ProbeError(
            f"Multi-select question {question['question_key']} is missing options: {sorted(missing_keys)}"
        )

    # First clear anything that should be off, then enable the desired set.
    for option in current_question.get("options", []):
        if option.get("selected") and option["option_key"] not in desired_keys:
            await click_option(page, current_question, option)
            await page.wait_for_timeout(150)
            current_snapshot = await extract_snapshot(page)
            current_question = next(
                (item for item in current_snapshot.get("questions", []) if item["question_key"] == question["question_key"]),
                None,
            )
            if current_question is None:
                raise ProbeError(f"Multi-select question {question['question_key']} disappeared while clearing options.")

    current_option_map = {item["option_key"]: item for item in current_question.get("options", [])}
    for option_key_value in selected_option_keys:
        option = current_option_map.get(option_key_value)
        if option is None:
            raise ProbeError(
                f"Option {option_key_value} is not visible for multi-select question {question['question_key']}."
            )
        if option.get("selected"):
            continue
        await click_option(page, current_question, option)
        await page.wait_for_timeout(150)
        current_snapshot = await extract_snapshot(page)
        current_question = next(
            (item for item in current_snapshot.get("questions", []) if item["question_key"] == question["question_key"]),
            None,
        )
        if current_question is None:
            raise ProbeError(f"Multi-select question {question['question_key']} disappeared while selecting options.")
        current_option_map = {item["option_key"]: item for item in current_question.get("options", [])}


def is_terms_control(control: Dict[str, Any]) -> bool:
    haystack = " ".join(
        [
            normalize_text(control.get("text") or ""),
            normalize_text(control.get("raw_text") or ""),
            normalize_text(control.get("option_label") or ""),
            normalize_text(control.get("debug_id") or ""),
        ]
    ).lower()
    return "iarc-tou-checkbox" in haystack or ("terms" in haystack and ("condition" in haystack or "use" in haystack))


def is_email_control(control: Dict[str, Any]) -> bool:
    control_type = normalize_text(control.get("control_type") or "").lower()
    haystack = " ".join(
        [
            control_type,
            normalize_text(control.get("text") or ""),
            normalize_text(control.get("raw_text") or ""),
            normalize_text(control.get("placeholder") or ""),
            normalize_text(control.get("debug_id") or ""),
        ]
    ).lower()
    return control_type in {"email", "text"} and "email" in haystack


async def set_auxiliary_checkbox(page: Any, control: Dict[str, Any], checked: bool) -> bool:
    probe_id = control.get("probe_id")
    if not probe_id:
        return False

    try:
        clicked = await page.evaluate(
            """
            ({ probeId, checked }) => {
              const row = document.querySelector(`[data-branch-probe-id="${probeId}"]`);
              if (!row) {
                return false;
              }
              const input = row.querySelector('input[type="checkbox"], [role="checkbox"]');
              if (!input) {
                return false;
              }
              const isChecked = (node) => {
                if (typeof node.checked === 'boolean') {
                  return node.checked;
                }
                const ariaChecked = node.getAttribute('aria-checked');
                if (ariaChecked === 'true') {
                  return true;
                }
                if (ariaChecked === 'false') {
                  return false;
                }
                const host = node.closest('material-checkbox, mat-checkbox');
                if (host) {
                  const hostAria = host.getAttribute('aria-checked');
                  if (hostAria === 'true') {
                    return true;
                  }
                  if (hostAria === 'false') {
                    return false;
                  }
                  const classes = `${host.className || ''}`;
                  if (/\b(selected|checked|mdc-checkbox--selected|mat-mdc-checkbox-checked)\b/i.test(classes)) {
                    return true;
                  }
                }
                return false;
              };
              if (isChecked(input) === checked) {
                return true;
              }
              const clickable = input.closest('label, material-checkbox, mat-checkbox, [role="checkbox"]') || input;
              clickable.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
              clickable.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
              clickable.click();
              clickable.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
              return true;
            }
            """,
            {"probeId": probe_id, "checked": checked},
        )
        if clicked:
            return True
    except Exception:
        pass

    locator = page.locator(f"[data-branch-probe-id='{probe_id}']").first.locator(
        "input[type='checkbox'], [role='checkbox']"
    ).first
    try:
        await locator.click(timeout=3000, force=True)
        return True
    except Exception:
        return False


async def fill_auxiliary_input(page: Any, control: Dict[str, Any], value: str) -> bool:
    probe_id = control.get("probe_id")
    if not probe_id or not value:
        return False

    try:
        filled = await page.evaluate(
            """
            ({ probeId, value }) => {
              const row = document.querySelector(`[data-branch-probe-id="${probeId}"]`);
              if (!row) {
                return false;
              }
              const input = row.querySelector(
                'input:not([type="radio"]):not([type="checkbox"]):not([type="hidden"]), textarea, select'
              );
              if (!input) {
                return false;
              }
              const setter =
                Object.getOwnPropertyDescriptor(Object.getPrototypeOf(input), 'value')?.set ||
                Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set ||
                Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;
              input.focus();
              if (setter) {
                setter.call(input, value);
              } else {
                input.value = value;
              }
              input.dispatchEvent(new Event('input', { bubbles: true }));
              input.dispatchEvent(new Event('change', { bubbles: true }));
              input.blur();
              return true;
            }
            """,
            {"probeId": probe_id, "value": value},
        )
        if filled:
            return True
    except Exception:
        pass

    locator = page.locator(f"[data-branch-probe-id='{probe_id}']").first.locator(
        "input:not([type='radio']):not([type='checkbox']):not([type='hidden']), textarea, select"
    ).first
    try:
        await locator.fill(value, timeout=3000)
        return True
    except Exception:
        return False


async def prepare_auxiliary_controls(
    page: Any,
    snapshot: Dict[str, Any],
    fallback_email: str = "",
) -> List[str]:
    actions: List[str] = []
    for control in snapshot.get("auxiliary_controls", []):
        if control.get("control_type") == "checkbox" and is_terms_control(control) and not control.get("checked"):
            if await set_auxiliary_checkbox(page, control, checked=True):
                actions.append(f"checked:{control['text']}")
                await page.wait_for_timeout(250)
            continue

        if is_email_control(control) and not control.get("value_present") and fallback_email:
            if await fill_auxiliary_input(page, control, fallback_email):
                actions.append(f"filled:{control['text']}")
                await page.wait_for_timeout(250)

    return actions


async def click_next_button(page: Any, snapshot: Dict[str, Any]) -> bool:
    labels = [
        button["text"]
        for button in snapshot.get("buttons", [])
        if button["kind"] == "next" and not button["disabled"]
    ]
    seen = set()
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        try:
            clicked = await page.evaluate(
                """
                ({ buttonLabel }) => {
                  const normalize = (value) => (value || '').replace(/\\s+/g, ' ').trim();
                  const elements = Array.from(document.querySelectorAll('button, [role="button"]'));
                  for (const element of elements) {
                    const text = normalize(
                      element.innerText || element.textContent || element.getAttribute('aria-label') || ''
                    );
                    const disabled =
                      element.hasAttribute('disabled') || element.getAttribute('aria-disabled') === 'true';
                    if (!disabled && text === buttonLabel) {
                      element.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
                      element.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
                      element.click();
                      element.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
                      return true;
                    }
                  }
                  return false;
                }
                """,
                {"buttonLabel": label},
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


async def replay_path(
    page: Any,
    start_url: str,
    path: List[Dict[str, Any]],
    settle_ms: int,
    fallback_email: str = "",
) -> Dict[str, Any]:
    try:
        await page.goto(start_url, wait_until="networkidle")
    except Exception:
        await page.goto(start_url, wait_until="domcontentloaded")
    await wait_after_action(page, settle_ms)

    index = 0
    safeguard = 0
    while index < len(path):
        safeguard += 1
        if safeguard > max(100, len(path) * 6 + 10):
            raise ProbeError("Replay exceeded the step safeguard.")

        snapshot = await extract_snapshot(page)
        action = path[index]

        if action["kind"] == "continue":
            await prepare_auxiliary_controls(page, snapshot, fallback_email=fallback_email)
            snapshot = await extract_snapshot(page)
            clicked = await click_next_button(page, snapshot)
            if not clicked:
                raise ProbeError("Replay expected a Next/Continue button but none was clickable.")
            await wait_after_action(page, settle_ms)
            index += 1
            continue

        if action["kind"] == "answer_multi":
            question = next(
                (item for item in snapshot["questions"] if item["question_key"] == action["question_key"]),
                None,
            )
            if question is None:
                raise ProbeError(f"Multi-select question {action['question_key']} is not visible.")
            await set_multi_answer(page, question, action.get("option_keys", []))
            await wait_after_action(page, settle_ms)
            index += 1
            continue

        question = next(
            (item for item in snapshot["questions"] if item["question_key"] == action["question_key"]),
            None,
        )
        if question is None:
            if snapshot["can_continue"] and not snapshot["can_finalize"]:
                await prepare_auxiliary_controls(page, snapshot, fallback_email=fallback_email)
                snapshot = await extract_snapshot(page)
                clicked = await click_next_button(page, snapshot)
                if not clicked:
                    raise ProbeError(
                        f"Question {action['question_key']} is not visible and auto-continue failed."
                    )
                await wait_after_action(page, settle_ms)
                continue
            raise ProbeError(
                f"Question {action['question_key']} is not visible and there is no safe continue button."
            )

        option = next(
            (item for item in question["options"] if item["option_key"] == action["option_key"]),
            None,
        )
        if option is None:
            raise ProbeError(
                f"Option {action['option_key']} is not visible for question {action['question_key']}."
            )
        await click_option(page, question, option)
        await wait_after_action(page, settle_ms)
        index += 1

    return await extract_snapshot(page)


async def save_state_artifacts(
    page: Any,
    output_dir: Path,
    state_id: str,
    snapshot: Dict[str, Any],
    screenshot: bool,
) -> Dict[str, str]:
    artifacts: Dict[str, str] = {}
    html_dir = ensure_dir(output_dir / "html")
    screenshot_dir = ensure_dir(output_dir / "screenshots")

    html_path = html_dir / f"{state_id}.html"
    try:
        content = await page.content()
        html_path.write_text(content, encoding="utf-8")
        artifacts["html"] = str(html_path)
    except Exception:
        pass

    if screenshot:
        screenshot_path = screenshot_dir / f"{state_id}.png"
        try:
            await page.screenshot(path=str(screenshot_path), full_page=True)
            artifacts["screenshot"] = str(screenshot_path)
        except Exception:
            pass

    snapshot_path = ensure_dir(output_dir / "snapshots") / f"{state_id}.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts["snapshot"] = str(snapshot_path)
    return artifacts


def summarize_path(path: List[Dict[str, Any]]) -> str:
    parts = []
    for action in path:
        if action["kind"] == "continue":
            parts.append("[continue]")
        elif action["kind"] == "answer_multi":
            labels = action.get("option_labels") or []
            rendered = ", ".join(labels) if labels else "[none]"
            parts.append(f"{action['question_text']} -> {{{rendered}}}")
        else:
            parts.append(f"{action['question_text']} -> {action['option_label']}")
    return " | ".join(parts)


def build_question_catalog(states: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_id = {state["state_id"]: state for state in states}
    catalog: Dict[str, Dict[str, Any]] = {}

    for state in states:
        for question in state["snapshot"].get("questions", []):
            entry = catalog.setdefault(
                question["question_key"],
                {
                    "question_key": question["question_key"],
                    "text": question["text"],
                    "question_type": question["question_type"],
                    "options": {},
                    "seen_in_states": [],
                    "introduced_examples": [],
                },
            )
            entry["seen_in_states"].append(state["state_id"])
            for option in question.get("options", []):
                entry["options"][option["option_key"]] = {
                    "option_key": option["option_key"],
                    "label": option["label"],
                    "role": option["role"],
                }

    for state in states:
        if not state.get("parent_state_id"):
            continue
        parent = by_id.get(state["parent_state_id"])
        if not parent:
            continue
        current_keys = {item["question_key"] for item in state["snapshot"].get("questions", [])}
        parent_keys = {item["question_key"] for item in parent["snapshot"].get("questions", [])}
        new_keys = sorted(current_keys - parent_keys)
        last_action = state["path"][-1] if state["path"] else None
        if not last_action or last_action["kind"] not in {"answer", "answer_multi"}:
            continue
        for key in new_keys:
            example = {
                "trigger_question_key": last_action["question_key"],
                "trigger_question_text": last_action["question_text"],
                "from_state_id": state["parent_state_id"],
                "to_state_id": state["state_id"],
                "path_summary": summarize_path(state["path"]),
            }
            if last_action["kind"] == "answer":
                example["trigger_option_key"] = last_action["option_key"]
                example["trigger_option_label"] = last_action["option_label"]
            else:
                example["trigger_option_keys"] = last_action.get("option_keys", [])
                example["trigger_option_labels"] = last_action.get("option_labels", [])
            catalog[key]["introduced_examples"].append(example)

    questions = []
    for entry in catalog.values():
        questions.append(
            {
                "question_key": entry["question_key"],
                "text": entry["text"],
                "question_type": entry["question_type"],
                "options": sorted(entry["options"].values(), key=lambda item: item["label"]),
                "seen_in_states": sorted(set(entry["seen_in_states"])),
                "introduced_examples": entry["introduced_examples"][:20],
            }
        )
    questions.sort(key=lambda item: item["question_key"])
    return {"question_count": len(questions), "questions": questions}


async def choose_start_url(page: Any, cli_start_url: Optional[str]) -> str:
    credentials = read_credentials()
    start_url = cli_start_url or credentials.console_url or DEFAULT_CONSOLE_URL

    print("Browser launched. Please use this Playwright window to log in and finish any SMS/MFA checks first.")
    print("You can also navigate manually before the probe starts.")
    await prompt_continue("After login is complete, press Enter here to continue...")

    current_url = page.url
    if (not current_url or current_url == "about:blank") and start_url:
        await page.goto(start_url, wait_until="domcontentloaded")
        print(f"Opened start URL: {start_url}")
        print("Please navigate from there to the questionnaire page if needed.")
    else:
        print(f"Keeping current page: {current_url}")

    await prompt_continue("When the questionnaire page is ready, press Enter here to start probing...")
    current_url = page.url
    if not current_url or current_url == "about:blank":
        raise ProbeError("The page is still blank. Open the questionnaire page before continuing.")
    return current_url


async def run_probe(config: ProbeConfig, browser_config: BrowserConfig) -> Dict[str, Any]:
    ensure_dir(config.output_dir)
    states: List[Dict[str, Any]] = []
    queue: deque[List[Dict[str, Any]]] = deque([[]])
    queued = {path_id([])}
    processed = set()

    async with launch_persistent_context(browser_config) as context:
        page = context.pages[0] if context.pages else await context.new_page()
        start_url = await choose_start_url(page, config.start_url)

        while queue and len(states) < config.max_states:
            path = queue.popleft()
            current_path_id = path_id(path)
            queued.discard(current_path_id)
            if current_path_id in processed:
                continue
            processed.add(current_path_id)
            parent_path = path[:-1]
            parent_state_id = path_id(parent_path) if path else None

            try:
                snapshot = await replay_path(
                    page,
                    start_url,
                    path,
                    config.settle_ms,
                    fallback_email=config.fallback_email,
                )
                artifacts = await save_state_artifacts(
                    page=page,
                    output_dir=config.output_dir,
                    state_id=current_path_id,
                    snapshot=snapshot,
                    screenshot=config.screenshot,
                )
                error = None
            except Exception as exc:
                snapshot = {
                    "url": page.url,
                    "title": "",
                    "headings": [],
                    "questions": [],
                    "auxiliary_controls": [],
                    "buttons": [],
                    "errors": [],
                    "body_fingerprint": "",
                    "state_signature": "",
                    "can_continue": False,
                    "can_finalize": False,
                }
                artifacts = {}
                error = str(exc)

            state = {
                "state_id": current_path_id,
                "parent_state_id": parent_state_id,
                "path": path,
                "path_summary": summarize_path(path),
                "path_depth": len(path),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "snapshot": snapshot,
                "artifacts": artifacts,
                "error": error,
            }
            states.append(state)
            print(f"[{len(states):03d}] {current_path_id} questions={len(snapshot['questions'])} error={bool(error)}")

            if error:
                continue

            next_question = first_unanswered_question(snapshot, path)
            if next_question:
                if next_question.get("question_type") == "multi":
                    subsets = nonempty_option_subsets(
                        next_question["options"][: config.max_options_per_question],
                        config.max_multi_combinations,
                    )
                    for subset in subsets:
                        child_path = path + [
                            {
                                "kind": "answer_multi",
                                "question_key": next_question["question_key"],
                                "question_text": next_question["text"],
                                "option_keys": [option["option_key"] for option in subset],
                                "option_labels": [option["label"] for option in subset],
                            }
                        ]
                        child_id = path_id(child_path)
                        if child_id not in queued and child_id not in processed:
                            queue.append(child_path)
                            queued.add(child_id)
                else:
                    for option in next_question["options"][: config.max_options_per_question]:
                        child_path = path + [
                            {
                                "kind": "answer",
                                "question_key": next_question["question_key"],
                                "question_text": next_question["text"],
                                "option_key": option["option_key"],
                                "option_label": option["label"],
                            }
                        ]
                        child_id = path_id(child_path)
                        if child_id not in queued and child_id not in processed:
                            queue.append(child_path)
                            queued.add(child_id)
                continue

            if snapshot["can_continue"] and not snapshot["can_finalize"]:
                child_path = path + [{"kind": "continue"}]
                child_id = path_id(child_path)
                if child_id not in queued and child_id not in processed:
                    queue.append(child_path)
                    queued.add(child_id)

    catalog = build_question_catalog(states)
    edges = []
    for state in states:
        if not state.get("parent_state_id"):
            continue
        parent = next((item for item in states if item["state_id"] == state["parent_state_id"]), None)
        current_keys = {item["question_key"] for item in state["snapshot"].get("questions", [])}
        parent_keys = {item["question_key"] for item in parent["snapshot"].get("questions", [])} if parent else set()
        edges.append(
            {
                "from_state_id": state["parent_state_id"],
                "to_state_id": state["state_id"],
                "action": state["path"][-1] if state["path"] else None,
                "new_question_keys": sorted(current_keys - parent_keys),
                "removed_question_keys": sorted(parent_keys - current_keys),
            }
        )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "start_url": start_url,
        "state_count": len(states),
        "question_count": catalog["question_count"],
        "states_path": str(config.output_dir / "states.json"),
        "edges_path": str(config.output_dir / "edges.json"),
        "question_catalog_path": str(config.output_dir / "question_catalog.json"),
    }

    write_json(config.output_dir / "states.json", states)
    write_json(config.output_dir / "edges.json", edges)
    write_json(config.output_dir / "question_catalog.json", catalog)
    write_json(config.output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore Google Play questionnaire branches without modifying the existing pipeline."
    )
    parser.add_argument("--start-url", default=None, help="Open this URL before you manually position the page.")
    parser.add_argument("--output-dir", default=None, help="Directory for probe outputs.")
    parser.add_argument("--profile-dir", default="browser_profile/play-console")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-states", type=int, default=300)
    parser.add_argument("--max-options-per-question", type=int, default=8)
    parser.add_argument("--max-multi-combinations", type=int, default=32)
    parser.add_argument("--settle-ms", type=int, default=1200)
    parser.add_argument("--no-screenshot", action="store_true")
    parser.add_argument("--fallback-email", default="")
    return parser.parse_args()


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "questionnaire_probe" / stamp


async def async_main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    config = ProbeConfig(
        start_url=args.start_url or "",
        output_dir=output_dir,
        max_states=args.max_states,
        max_options_per_question=args.max_options_per_question,
        max_multi_combinations=args.max_multi_combinations,
        settle_ms=args.settle_ms,
        screenshot=not args.no_screenshot,
        fallback_email=args.fallback_email,
    )
    browser_config = BrowserConfig(
        profile_dir=args.profile_dir,
        headless=args.headless,
        navigation_timeout_ms=60000,
        selector_timeout_ms=15000,
    )
    summary = await run_probe(config, browser_config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
