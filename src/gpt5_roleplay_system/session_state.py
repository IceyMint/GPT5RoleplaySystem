from __future__ import annotations

import json
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from .memory import MemoryItem


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return cleaned or "default"


def _memory_item_from_dict(raw: Dict[str, Any]) -> MemoryItem:
    return MemoryItem(
        text=str(raw.get("text", "") or ""),
        sender_id=str(raw.get("sender_id", "") or ""),
        sender_name=str(raw.get("sender_name", "") or ""),
        timestamp=float(raw.get("timestamp", 0.0) or 0.0),
        metadata=dict(raw.get("metadata", {}) or {}),
    )


class SessionStateStore:
    def __init__(self, state_dir: str, persona: str, user_id: str) -> None:
        base = Path(state_dir).expanduser()
        base.mkdir(parents=True, exist_ok=True)
        persona_slug = _slug(persona or "persona")
        user_slug = _slug(user_id or "user")
        self._path = base / f"session_state_{persona_slug}_{user_slug}.json"

    def load(self) -> Dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        data["rolling_buffer"] = self._load_items(data.get("rolling_buffer", []))
        memory = data.get("memory", {})
        if isinstance(memory, dict):
            memory["recent"] = self._load_items(memory.get("recent", []))
            data["memory"] = memory
        return data

    def save(self, state: Dict[str, Any]) -> None:
        payload = dict(state or {})
        payload["rolling_buffer"] = self._dump_items(payload.get("rolling_buffer", []))
        memory = payload.get("memory", {})
        if isinstance(memory, dict):
            memory["recent"] = self._dump_items(memory.get("recent", []))
            payload["memory"] = memory
        self._atomic_write(payload)

    def _atomic_write(self, payload: Dict[str, Any]) -> None:
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(self._path.parent),
                prefix="session_state_",
                suffix=".json",
                delete=False,
            ) as handle:
                json.dump(payload, handle, ensure_ascii=True)
                handle.flush()
                tmp_path = handle.name
            Path(tmp_path).replace(self._path)
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _load_items(raw_items: Any) -> List[MemoryItem]:
        if not isinstance(raw_items, list):
            return []
        items: List[MemoryItem] = []
        for raw in raw_items:
            if isinstance(raw, dict):
                items.append(_memory_item_from_dict(raw))
        return items

    @staticmethod
    def _dump_items(items: Any) -> List[Dict[str, Any]]:
        if not isinstance(items, list):
            return []
        dumped: List[Dict[str, Any]] = []
        for item in items:
            if isinstance(item, MemoryItem):
                dumped.append(asdict(item))
            elif isinstance(item, dict):
                dumped.append(item)
        return dumped
