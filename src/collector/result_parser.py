from __future__ import annotations

import re
from typing import Dict, List, Optional


AGE_RE = re.compile(r"\b(3\+|7\+|12\+|16\+|18\+|Everyone|Teen|Mature|Adults only)\b", re.I)


def parse_age_rating(text: str) -> Optional[str]:
    matches = AGE_RE.findall(text or "")
    if not matches:
        return None
    normalized = {
        "everyone": "3+",
        "teen": "12+",
        "mature": "16+",
        "adults only": "18+",
    }
    value = matches[0]
    return normalized.get(value.lower(), value)


def parse_region_ratings(text: str) -> Dict[str, str]:
    regions = ["ESRB", "PEGI", "USK", "IARC", "ACB", "ClassInd", "Google Play"]
    result: Dict[str, str] = {}
    for region in regions:
        pattern = re.compile(rf"{re.escape(region)}[^\n\r:：]*[:：]?\s*([A-Za-z0-9+ ]{{1,20}})", re.I)
        match = pattern.search(text or "")
        if match:
            rating = parse_age_rating(match.group(1)) or match.group(1).strip()
            result[region] = rating
    return result


def parse_list_after_heading(text: str, headings: List[str]) -> List[str]:
    values: List[str] = []
    for heading in headings:
        pattern = re.compile(rf"{heading}[^\n\r]*[:：]\s*(?P<value>[^\n\r]+)", re.I)
        match = pattern.search(text or "")
        if match:
            raw = match.group("value")
            values.extend(
                item.strip(" ,;")
                for item in re.split(r"[,;、]", raw)
                if item.strip(" ,;")
            )
    return sorted(set(values))
