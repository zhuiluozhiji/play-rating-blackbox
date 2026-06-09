from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Set

from src.common import ensure_parent, project_path


def answer_hash(answers: Dict[str, Any]) -> str:
    payload = json.dumps(answers, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JsonlStore:
    def __init__(self, path: str | Path):
        self.path = project_path(path)
        ensure_parent(self.path)

    def append(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")

    def append_many(self, records: Iterable[Dict[str, Any]]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            for record in records:
                json.dump(record, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {self.path}:{line_no}") from exc

    def read_all(self) -> List[Dict[str, Any]]:
        return list(self.iter_records() or [])

    def answer_hashes(self) -> Set[str]:
        return {
            answer_hash(record.get("answers_json") or {})
            for record in self.iter_records() or []
            if record.get("answers_json")
        }
