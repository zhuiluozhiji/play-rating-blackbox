from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from src.common import ensure_parent, project_path


DEFAULT_PATH = "docs/人工操作清单.md"


def append_manual_action(
    title: str,
    detail: str,
    path: str | Path = DEFAULT_PATH,
    sample_id: Optional[str] = None,
) -> Path:
    resolved = ensure_parent(path)
    if not resolved.exists() or resolved.stat().st_size == 0:
        resolved.write_text("# 人工操作清单\n\n## 待处理事项\n\n", encoding="utf-8")
    timestamp = datetime.now().isoformat(timespec="seconds")
    sample_text = f"\n- sample_id: `{sample_id}`" if sample_id else ""
    entry = (
        f"\n### {timestamp} - {title}\n\n"
        f"- 状态: 待人工确认{sample_text}\n"
        f"- 说明: {detail}\n"
    )
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    return resolved


def manual_ops_path(path: str | Path = DEFAULT_PATH) -> Path:
    return project_path(path)
