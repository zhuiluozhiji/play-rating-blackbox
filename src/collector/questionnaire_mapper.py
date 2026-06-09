from __future__ import annotations

import hashlib
from typing import Any, Dict, List

from src.data.schema import default_question_schema


def page_fingerprint(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()[:12]


async def extract_visible_questions(page: Any) -> List[Dict[str, Any]]:
    candidates = await page.locator("mat-radio-group, mat-checkbox, [role='radiogroup'], [role='group']").all()
    questions: List[Dict[str, Any]] = []
    for index, locator in enumerate(candidates):
        try:
            text = " ".join((await locator.inner_text()).split())
        except Exception:
            continue
        if not text:
            continue
        questions.append(
            {
                "question_id": f"page_q_{index + 1}",
                "text": text[:500],
                "question_type": "unknown",
                "options": [],
            }
        )
    return questions


def fallback_schema() -> Dict[str, Any]:
    return default_question_schema()
