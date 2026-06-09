from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from src.common import project_path


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"https?://[^\s)>\"']+")


@dataclass
class CredentialBundle:
    email: Optional[str] = None
    password: Optional[str] = None
    console_url: Optional[str] = None
    app_id: Optional[str] = None
    package_name: Optional[str] = None

    def redacted(self) -> Dict[str, str]:
        return {
            "email": "<redacted>" if self.email else "",
            "password": "<redacted>" if self.password else "",
            "console_url": self.console_url or "",
            "app_id": self.app_id or "",
            "package_name": self.package_name or "",
        }


def _env_first() -> CredentialBundle:
    return CredentialBundle(
        email=os.getenv("GOOGLE_EMAIL") or None,
        password=os.getenv("GOOGLE_PASSWORD") or None,
        console_url=os.getenv("PLAY_CONSOLE_URL") or None,
        app_id=os.getenv("PLAY_APP_ID") or None,
        package_name=os.getenv("PLAY_PACKAGE_NAME") or None,
    )


def _extract_value_by_label(text: str, labels: list[str]) -> Optional[str]:
    for label in labels:
        pattern = re.compile(
            rf"{label}\s*[:：=]\s*(?P<value>[^\n\r]+)",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            value = match.group("value").strip().strip("`'\"")
            if value:
                return value
    return None


def read_credentials(path: str | Path = "docs/key.md") -> CredentialBundle:
    bundle = _env_first()
    resolved = project_path(path)
    if not resolved.exists():
        return bundle

    text = resolved.read_text(encoding="utf-8", errors="ignore")
    email = bundle.email or _extract_value_by_label(
        text, ["email", "e-mail", "account", "账号", "邮箱", "用户名"]
    )
    if not email:
        match = EMAIL_RE.search(text)
        email = match.group(0) if match else None

    password = bundle.password or _extract_value_by_label(
        text, ["password", "passwd", "pwd", "密码"]
    )
    console_url = bundle.console_url or _extract_value_by_label(
        text, ["play_console_url", "console_url", "url", "链接", "地址"]
    )
    if not console_url:
        urls = URL_RE.findall(text)
        console_url = next((url for url in urls if "play.google.com/console" in url), None)

    app_id = bundle.app_id or _extract_value_by_label(
        text, ["play_app_id", "app_id", "application_id", "应用id", "应用 ID"]
    )
    package_name = bundle.package_name or _extract_value_by_label(
        text, ["package", "package_name", "包名"]
    )
    return CredentialBundle(
        email=email,
        password=password,
        console_url=console_url,
        app_id=app_id,
        package_name=package_name,
    )


def redact_text(text: str) -> str:
    redacted = EMAIL_RE.sub("<redacted-email>", text)
    redacted = re.sub(
        r"(?i)(password|passwd|pwd|密码)\s*[:：=]\s*[^\n\r]+",
        r"\1: <redacted>",
        redacted,
    )
    return redacted
