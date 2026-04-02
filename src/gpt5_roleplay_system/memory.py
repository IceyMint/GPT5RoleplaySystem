from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from .models import InboundChat


@dataclass
class MemoryItem:
    text: str
    sender_id: str
    sender_name: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RollingBuffer:
    def __init__(self, max_items: int = 30) -> None:
        self._max_items = max_items
        self._items: List[MemoryItem] = []

    def add_user_message(self, chat: InboundChat) -> None:
        self._items.append(
            MemoryItem(
                text=chat.text,
                sender_id=chat.sender_id,
                sender_name=chat.sender_name,
                timestamp=chat.timestamp,
            )
        )
        self._trim()

    def add_ai_message(self, text: str, persona: str) -> None:
        self._items.append(
            MemoryItem(
                text=text,
                sender_id="ai",
                sender_name=persona,
                timestamp=time.time(),
            )
        )
        self._trim()

    def items(self) -> List[MemoryItem]:
        return list(self._items)

    def snapshot(self) -> List[MemoryItem]:
        return list(self._items)

    def restore(self, items: List[MemoryItem]) -> None:
        if not items:
            self._items = []
            return
        self._items = list(items)[-self._max_items :]

    def trim_to_last(self, keep_last: int) -> None:
        keep = max(0, int(keep_last))
        if keep == 0:
            self._items = []
            return
        if len(self._items) > keep:
            self._items = self._items[-keep:]

    def _trim(self) -> None:
        if len(self._items) > self._max_items:
            self._items = self._items[-self._max_items :]


class MemoryCompressor:
    def compress(self, existing_summary: str, messages: Iterable[MemoryItem]) -> str:
        raise NotImplementedError


class SimpleMemoryCompressor(MemoryCompressor):
    def compress(self, existing_summary: str, messages: Iterable[MemoryItem]) -> str:
        # Fallbacks that generate naive summaries have been removed to prevent memory poisoning.
        # This implementation is now a no-op fallback, meaning only real LLM summaries
        # should update the summary property.
        return existing_summary.strip() if existing_summary else ""


class ConversationMemory:
    def __init__(
        self,
        compressor: MemoryCompressor,
        max_recent: int = 20,
        defer_compression: bool = False,
    ) -> None:
        self._compressor = compressor
        self._max_recent = max_recent
        self._defer_compression = defer_compression
        self._recent: List[MemoryItem] = []
        self._summary = ""
        self._summary_meta: Dict[str, Any] = {}
        self._overflow: List[MemoryItem] = []

    def add_message(self, chat: InboundChat) -> None:
        self._recent.append(
            MemoryItem(
                text=chat.text,
                sender_id=chat.sender_id,
                sender_name=chat.sender_name,
                timestamp=chat.timestamp,
            )
        )
        self._compress_if_needed()

    def add_ai_message(self, text: str, persona: str) -> None:
        self._recent.append(
            MemoryItem(
                text=text,
                sender_id="ai",
                sender_name=persona,
                timestamp=time.time(),
            )
        )
        self._compress_if_needed()

    def recent(self) -> List[MemoryItem]:
        return list(self._recent)

    def summary(self) -> str:
        return self._summary

    def summary_meta(self) -> Dict[str, Any]:
        return dict(self._summary_meta)

    def clamp_summary_range_end(self, max_end_ts: float) -> bool:
        limit = float(max_end_ts or 0.0)
        if limit <= 0.0:
            return False
        range_end = float(self._summary_meta.get("range_end_ts", 0.0) or 0.0)
        if range_end <= 0.0 or range_end <= limit:
            return False
        range_start = float(self._summary_meta.get("range_start_ts", 0.0) or 0.0)
        if range_start > limit and range_start > 0.0:
            range_start = limit
        self._summary_meta["range_start_ts"] = float(range_start or 0.0)
        self._summary_meta["range_end_ts"] = float(limit)
        return True

    def apply_summary(self, summary: str, timestamps: List[float] | None = None) -> None:
        cleaned = summary.strip()
        if not cleaned:
            self._summary = ""
            self._summary_meta = {}
            return
        self._summary = cleaned
        self._update_summary_meta(timestamps or [], time.time())

    def drain_overflow(self) -> List[MemoryItem]:
        overflow = self._overflow
        self._overflow = []
        return overflow

    def requeue_overflow(self, items: List[MemoryItem]) -> None:
        if not items:
            return
        # Preserve chronological order by restoring drained items before any
        # newly accumulated overflow from the current turn.
        self._overflow = list(items) + self._overflow

    def compress_overflow(self, overflow: List[MemoryItem] | None = None) -> None:
        items = list(overflow) if overflow is not None else list(self._overflow)
        if not items:
            return
        previous_summary = self._summary.strip() if self._summary else ""
        candidate_summary = self._compressor.compress(self._summary, items)
        cleaned_summary = candidate_summary.strip() if candidate_summary else ""
        if not cleaned_summary or cleaned_summary == previous_summary:
            # Don't claim timestamp coverage for drained items unless the summary text
            # actually advanced. This keeps no-op/simple compressors from causing
            # permanent overflow data loss.
            if overflow is not None:
                self.requeue_overflow(items)
            return
        self._summary = cleaned_summary
        timestamps = [float(item.timestamp or 0.0) for item in items]
        self._update_summary_meta(timestamps, time.time())
        if overflow is None:
            self._overflow = []

    def snapshot(self) -> Dict[str, Any]:
        return {
            "recent": list(self._recent),
            "summary": self._summary,
            "summary_meta": dict(self._summary_meta),
        }

    def restore(self, state: Dict[str, Any]) -> None:
        recent = state.get("recent", []) if isinstance(state, dict) else []
        summary = state.get("summary", "") if isinstance(state, dict) else ""
        summary_meta = state.get("summary_meta", {}) if isinstance(state, dict) else {}
        self._recent = list(recent)[-self._max_recent :]
        self._summary = str(summary or "")
        self._summary_meta = dict(summary_meta) if isinstance(summary_meta, dict) else {}
        self._overflow = []

    def _compress_if_needed(self) -> None:
        if len(self._recent) <= self._max_recent:
            return
        split_index = len(self._recent) - self._max_recent
        if split_index <= 0:
            return
        boundary_ts = float(self._recent[split_index].timestamp or 0.0)
        while split_index > 0 and float(self._recent[split_index - 1].timestamp or 0.0) == boundary_ts:
            split_index -= 1
        if split_index == 0:
            # Keep tied timestamps together; allow temporary overage instead of unstable split.
            return
        overflow = self._recent[:split_index]
        self._recent = self._recent[split_index:]
        
        # We no longer automatically compress here to avoid generating naive/fake summaries
        # that poison the context window. We accumulate in the overflow buffer instead.
        self._overflow.extend(overflow)
        
        # Enforce hard limit on overflow to avoid unbounded memory growth
        MAX_OVERFLOW = 200
        if len(self._overflow) > MAX_OVERFLOW:
            self._overflow = self._overflow[-MAX_OVERFLOW:]

    def _update_summary_meta(self, timestamps: List[float], now_ts: float) -> None:
        valid = [float(ts) for ts in timestamps if float(ts or 0.0) > 0.0]
        range_start_new = min(valid) if valid else 0.0
        range_end_new = max(valid) if valid else 0.0
        range_start_prev = float(self._summary_meta.get("range_start_ts", 0.0) or 0.0)
        range_end_prev = float(self._summary_meta.get("range_end_ts", 0.0) or 0.0)
        if range_start_new > 0.0:
            range_start = range_start_new if range_start_prev <= 0.0 else min(range_start_prev, range_start_new)
        else:
            range_start = range_start_prev
        if range_end_new > 0.0:
            range_end = range_end_new if range_end_prev <= 0.0 else max(range_end_prev, range_end_new)
        else:
            range_end = range_end_prev
        self._summary_meta = {
            "last_updated_ts": float(now_ts or 0.0),
            "range_start_ts": float(range_start or 0.0),
            "range_end_ts": float(range_end or 0.0),
        }


@dataclass
class ExperienceRecord:
    text: str
    metadata: Dict[str, Any]


class ExperienceStore:
    def __init__(self) -> None:
        self._experiences: List[ExperienceRecord] = []

    def add(self, text: str, metadata: Dict[str, Any], persona_id: str = "") -> ExperienceRecord:
        meta = dict(metadata or {})
        meta.setdefault("experience_id", uuid.uuid4().hex)
        if persona_id:
            meta["persona_id"] = persona_id
        record = ExperienceRecord(text=text, metadata=meta)
        self._experiences.append(record)
        return record

    def all(self) -> List[ExperienceRecord]:
        return list(self._experiences)


class SimilaritySearch:
    def search(self, query: str, experiences: List[ExperienceRecord], top_k: int = 3) -> List[ExperienceRecord]:
        raise NotImplementedError


class SimpleSimilaritySearch(SimilaritySearch):
    def search(self, query: str, experiences: List[ExperienceRecord], top_k: int = 3) -> List[ExperienceRecord]:
        if not query:
            return []
        scored = []
        query_lower = query.lower()
        for experience in experiences:
            text_lower = experience.text.lower()
            score = 1.0 if query_lower in text_lower else 0.0
            scored.append((score, experience))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:top_k] if item[0] > 0.0]


class TokenSimilaritySearch(SimilaritySearch):
    def __init__(self, min_token_len: int = 3) -> None:
        self._min_token_len = min_token_len

    def search(self, query: str, experiences: List[ExperienceRecord], top_k: int = 3) -> List[ExperienceRecord]:
        tokens = _tokenize(query, self._min_token_len)
        if not tokens:
            return []
        scored = []
        for experience in experiences:
            exp_tokens = _tokenize(experience.text, self._min_token_len)
            if not exp_tokens:
                continue
            intersection = tokens & exp_tokens
            if not intersection:
                continue
            score = len(intersection) / max(len(tokens | exp_tokens), 1)
            scored.append((score, experience))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:top_k]]


def _tokenize(text: str, min_len: int) -> set[str]:
    if not text:
        return set()
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    tokens = {token for token in cleaned.split() if len(token) >= min_len}
    return tokens
