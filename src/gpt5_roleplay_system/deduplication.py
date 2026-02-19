from __future__ import annotations

from datetime import datetime
import json
import re
from typing import Any
from zoneinfo import ZoneInfo

from .time_utils import format_pacific_time


def parse_deduped_facts_response(response_text: str) -> list[str]:
    clean_text = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", response_text or "", flags=re.DOTALL)
    start = clean_text.find("{")
    end = clean_text.rfind("}")
    if start != -1 and end != -1:
        clean_text = clean_text[start : end + 1]
    clean_text = clean_text.strip()
    if not clean_text:
        raise ValueError("empty LLM response content")

    data = json.loads(clean_text)
    if isinstance(data, dict):
        refined_raw = data.get("facts", [])
    elif isinstance(data, list):
        refined_raw = data
    else:
        raise ValueError(f"unexpected JSON payload type: {type(data).__name__}")
    if not isinstance(refined_raw, list):
        raise ValueError("facts must be a list")

    seen: set[str] = set()
    refined_facts: list[str] = []
    for item in refined_raw:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        refined_facts.append(text)
    return refined_facts


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_experience_timestamp(raw_value: Any) -> float:
    text = str(raw_value or "").strip()
    if not text or text == "0":
        return 0.0
    normalized = text.replace("T", " ")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        try:
            parsed = datetime.strptime(normalized, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
    return parsed.timestamp()


def extract_experience_window(row: Any, prefix: str) -> tuple[float, float]:
    timestamp = safe_float(row.get(f"{prefix}_timestamp", 0.0))
    start_ts = parse_experience_timestamp(row.get(f"{prefix}_timestamp_start", ""))
    end_ts = parse_experience_timestamp(row.get(f"{prefix}_timestamp_end", ""))
    if timestamp > 0.0:
        start_ts = timestamp if start_ts <= 0.0 else min(start_ts, timestamp)
        end_ts = timestamp if end_ts <= 0.0 else max(end_ts, timestamp)
    if start_ts <= 0.0 and end_ts > 0.0:
        start_ts = end_ts
    if end_ts <= 0.0 and start_ts > 0.0:
        end_ts = start_ts
    if start_ts > 0.0 and end_ts > 0.0 and end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts
    return start_ts, end_ts


def experience_windows_are_close(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
    max_gap_seconds: float,
) -> bool:
    if left_start <= 0.0 or left_end <= 0.0 or right_start <= 0.0 or right_end <= 0.0:
        return False
    if left_end >= right_start and right_end >= left_start:
        return True
    if left_end < right_start:
        gap = right_start - left_end
    else:
        gap = left_start - right_end
    return gap <= max_gap_seconds


class ExperienceDedupePlanner:
    @staticmethod
    def build_plans(
        rows: list[dict[str, Any]],
        similarity_threshold: float,
        max_gap_seconds: float,
    ) -> tuple[list[dict[str, Any]], int]:
        parent: dict[str, str] = {}
        windows: dict[str, dict[str, float]] = {}

        def find(node_id: str) -> str:
            root = parent.get(node_id, node_id)
            if root != node_id:
                root = find(root)
                parent[node_id] = root
            return root

        def union(left_id: str, right_id: str) -> None:
            root_left = find(left_id)
            root_right = find(right_id)
            if root_left == root_right:
                return
            if root_left < root_right:
                parent[root_right] = root_left
            else:
                parent[root_left] = root_right

        def upsert_window(node_id: str, start_ts: float, end_ts: float) -> None:
            if not node_id:
                return
            existing = windows.get(node_id)
            if existing is None:
                windows[node_id] = {"start": start_ts, "end": end_ts}
                return
            existing_start = safe_float(existing.get("start", 0.0))
            existing_end = safe_float(existing.get("end", 0.0))
            if start_ts > 0.0:
                existing_start = start_ts if existing_start <= 0.0 else min(existing_start, start_ts)
            if end_ts > 0.0:
                existing_end = end_ts if existing_end <= 0.0 else max(existing_end, end_ts)
            if existing_start <= 0.0 and existing_end > 0.0:
                existing_start = existing_end
            if existing_end <= 0.0 and existing_start > 0.0:
                existing_end = existing_start
            windows[node_id] = {"start": existing_start, "end": existing_end}

        qualifying_pair_count = 0
        for row in rows:
            left_id = str(row.get("left_id", "") or "")
            right_id = str(row.get("right_id", "") or "")
            if not left_id or not right_id or left_id == right_id:
                continue
            score = safe_float(row.get("score", 0.0))
            if score < similarity_threshold:
                continue
            left_start, left_end = extract_experience_window(row, "left")
            right_start, right_end = extract_experience_window(row, "right")
            if not experience_windows_are_close(left_start, left_end, right_start, right_end, max_gap_seconds):
                continue
            qualifying_pair_count += 1
            upsert_window(left_id, left_start, left_end)
            upsert_window(right_id, right_start, right_end)
            parent.setdefault(left_id, left_id)
            parent.setdefault(right_id, right_id)
            union(left_id, right_id)

        grouped: dict[str, list[str]] = {}
        for node_id in windows:
            root_id = find(node_id)
            grouped.setdefault(root_id, []).append(node_id)

        plans: list[dict[str, Any]] = []
        for members in grouped.values():
            if len(members) < 2:
                continue
            windows_for_members: list[tuple[str, float, float]] = []
            for node_id in members:
                window = windows.get(node_id, {})
                start_ts = safe_float(window.get("start", 0.0))
                end_ts = safe_float(window.get("end", 0.0))
                if start_ts <= 0.0 and end_ts <= 0.0:
                    continue
                if start_ts <= 0.0:
                    start_ts = end_ts
                if end_ts <= 0.0:
                    end_ts = start_ts
                if end_ts < start_ts:
                    start_ts, end_ts = end_ts, start_ts
                windows_for_members.append((node_id, start_ts, end_ts))
            if len(windows_for_members) < 2:
                continue
            canonical_id = max(windows_for_members, key=lambda item: (item[2], item[1], item[0]))[0]
            merged_start = min(item[1] for item in windows_for_members)
            merged_end = max(item[2] for item in windows_for_members)
            duplicate_ids = sorted(item[0] for item in windows_for_members if item[0] != canonical_id)
            if not duplicate_ids:
                continue
            plans.append(
                {
                    "keep_id": canonical_id,
                    "dup_ids": duplicate_ids,
                    "merged_timestamp": merged_end,
                    "merged_timestamp_start": format_pacific_time(merged_start),
                    "merged_timestamp_end": format_pacific_time(merged_end),
                }
            )
        plans.sort(key=lambda item: (float(item.get("merged_timestamp", 0.0)), str(item.get("keep_id", ""))), reverse=True)
        return plans, qualifying_pair_count
