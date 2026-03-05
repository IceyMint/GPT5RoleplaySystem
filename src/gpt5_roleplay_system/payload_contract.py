from __future__ import annotations

import copy
import re
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from .name_utils import name_matches, normalize_for_match, split_display_and_username

_UUID_LIKE_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    flags=re.IGNORECASE,
)


def looks_like_uuid(value: str) -> bool:
    return bool(_UUID_LIKE_RE.match(str(value or "").strip()))


def canonical_identity_key(user_id: str, name: str) -> str:
    uid = str(user_id or "").strip()
    if uid:
        return f"id:{uid}"
    match_key = normalize_for_match(str(name or ""))
    if match_key:
        return f"name:{match_key}"
    return ""


def is_placeholder_self_id(user_id: str) -> bool:
    value = str(user_id or "").strip()
    if not value:
        return True
    if looks_like_uuid(value):
        return False
    return value.casefold() in {
        "default_user",
        "user",
        "self",
        "ai",
        "ai-user",
        "ai-1",
        "ai-config-id",
    }


def normalize_participants(
    participants: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    prepared = [_normalize_participant_entry(entry) for entry in participants if isinstance(entry, dict)]

    by_id: dict[str, dict[str, Any]] = {}
    name_only: list[dict[str, Any]] = []
    for entry in prepared:
        user_id = str(entry.get("user_id", "") or "").strip()
        if user_id:
            key = f"id:{user_id}"
            prior = by_id.get(key)
            by_id[key] = _merge_participant_entries(prior, entry) if prior else dict(entry)
            continue
        name_only.append(dict(entry))

    kept_name_only: list[dict[str, Any]] = []
    for entry in name_only:
        matches = [
            id_key
            for id_key, candidate in by_id.items()
            if _participant_entries_match_name(entry, candidate)
        ]
        if len(matches) == 1:
            key = matches[0]
            by_id[key] = _merge_participant_entries(by_id[key], entry)
            warnings.append(
                {
                    "category": "identity_merge",
                    "reason": "name_only_merged_into_id",
                    "name": _participant_name(entry),
                    "target_id": by_id[key].get("user_id", ""),
                }
            )
            continue
        if len(matches) > 1:
            warnings.append(
                {
                    "category": "identity_merge",
                    "reason": "ambiguous_name_only_match",
                    "name": _participant_name(entry),
                    "candidate_ids": [by_id[key].get("user_id", "") for key in matches],
                }
            )
        kept_name_only.append(entry)

    by_name: dict[str, dict[str, Any]] = {}
    for entry in kept_name_only:
        key = canonical_identity_key("", _participant_name(entry))
        if not key:
            continue
        prior = by_name.get(key)
        if prior:
            by_name[key] = _merge_participant_entries(prior, entry)
            warnings.append(
                {
                    "category": "identity_merge",
                    "reason": "dedupe_name_only_duplicates",
                    "name": _participant_name(entry),
                }
            )
        else:
            by_name[key] = dict(entry)

    merged = list(by_id.values()) + list(by_name.values())
    merged.sort(
        key=lambda entry: (
            str(entry.get("user_id", "") or ""),
            str(entry.get("name", "") or ""),
            str(entry.get("display_name", "") or ""),
        )
    )
    return merged, warnings


def normalize_and_validate_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    repaired = copy.deepcopy(payload if isinstance(payload, dict) else {})
    warnings: list[dict[str, Any]] = []

    participants = repaired.get("participants", [])
    if isinstance(participants, list):
        normalized, participant_warnings = normalize_participants(participants)
        repaired["participants"] = normalized
        warnings.extend(participant_warnings)

    coverage_warnings = _ensure_participant_coverage(repaired)
    warnings.extend(coverage_warnings)
    if coverage_warnings and isinstance(repaired.get("participants"), list):
        normalized, participant_warnings = normalize_participants(repaired["participants"])
        repaired["participants"] = normalized
        warnings.extend(participant_warnings)

    incoming_warnings = _repair_incoming_batch_and_incoming(repaired)
    warnings.extend(incoming_warnings)

    summary_warnings = _repair_summary_meta(repaired)
    warnings.extend(summary_warnings)

    warnings.extend(_self_identity_warnings(repaired))
    return repaired, warnings


def _normalize_participant_entry(entry: dict[str, Any]) -> dict[str, Any]:
    user_id = str(entry.get("user_id", "") or "").strip()
    raw_name = str(entry.get("name", "") or "").strip()
    username = str(entry.get("username", "") or "").strip()
    display_name = str(entry.get("display_name", "") or "").strip()
    full_name = str(entry.get("full_name", "") or "").strip()

    if not full_name:
        if display_name and username and display_name.casefold() != username.casefold():
            full_name = f"{display_name} ({username})"
        else:
            full_name = display_name or username or raw_name
    parsed_display, parsed_username = split_display_and_username(full_name)
    username = username or parsed_username or raw_name or user_id
    display_name = display_name or parsed_display or raw_name or username
    name = raw_name or username or display_name or user_id

    full_name = full_name or name
    return {
        "user_id": user_id,
        "name": name,
        "username": username,
        "display_name": display_name,
        "full_name": full_name,
    }


def _merge_participant_entries(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary or {})
    other = dict(secondary or {})

    if not str(merged.get("user_id", "") or "").strip():
        merged["user_id"] = str(other.get("user_id", "") or "").strip()

    for field_name in ("name", "username", "display_name"):
        merged[field_name] = _pick_richer_text(merged.get(field_name), other.get(field_name))
    merged["full_name"] = _pick_richer_full_name(merged.get("full_name"), other.get("full_name"))

    return _normalize_participant_entry(merged)


def _pick_richer_text(current: Any, candidate: Any) -> str:
    left = str(current or "").strip()
    right = str(candidate or "").strip()
    if not left:
        return right
    if not right:
        return left
    if len(right) > len(left):
        return right
    return left


def _pick_richer_full_name(current: Any, candidate: Any) -> str:
    left = str(current or "").strip()
    right = str(candidate or "").strip()
    if not left:
        return right
    if not right:
        return left
    left_score = (1 if "(" in left and ")" in left else 0, len(left))
    right_score = (1 if "(" in right and ")" in right else 0, len(right))
    return right if right_score > left_score else left


def _participant_names(entry: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key in ("full_name", "display_name", "username", "name"):
        value = str(entry.get(key, "") or "").strip()
        if value:
            names.append(value)
    return names


def _participant_name(entry: dict[str, Any]) -> str:
    names = _participant_names(entry)
    return names[0] if names else ""


def _participant_entries_match_name(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_names = _participant_names(left)
    right_names = _participant_names(right)
    for left_name in left_names:
        for right_name in right_names:
            if name_matches(left_name, right_name):
                return True
    return False


def _ensure_participant_coverage(payload: dict[str, Any]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    participants_raw = payload.get("participants", [])
    if not isinstance(participants_raw, list):
        return warnings

    participants = [_normalize_participant_entry(item) for item in participants_raw if isinstance(item, dict)]
    participant_keys = {
        canonical_identity_key(
            str(participant.get("user_id", "") or ""),
            _participant_name(participant),
        )
        for participant in participants
    }

    persona = str(payload.get("persona", "") or "")
    user_id = str(payload.get("user_id", "") or "")
    active = _collect_active_speakers(payload, persona=persona, user_id=user_id)

    for speaker in active:
        key = canonical_identity_key(speaker["user_id"], speaker["name"])
        if not key or key in participant_keys:
            continue
        participants.append(_normalize_participant_entry(speaker))
        participant_keys.add(key)
        warnings.append(
            {
                "category": "participant_coverage",
                "reason": "added_active_speaker",
                "sender_id": speaker["user_id"],
                "sender_name": speaker["name"],
            }
        )

    payload["participants"] = participants
    return warnings


def _collect_active_speakers(payload: dict[str, Any], *, persona: str, user_id: str) -> list[dict[str, str]]:
    active: dict[str, dict[str, str]] = {}

    def add_sender(sender_id: str, sender_name: str) -> None:
        sid = str(sender_id or "").strip()
        sname = str(sender_name or "").strip()
        if not sid and not sname:
            return
        if _is_self_identity(sid, sname, persona=persona, user_id=user_id):
            return
        key = canonical_identity_key(sid, sname)
        if not key:
            return
        active[key] = {
            "user_id": sid,
            "name": sname or sid,
        }

    for message in payload.get("recent_messages", []) or []:
        if not isinstance(message, dict):
            continue
        add_sender(
            str(message.get("sender_id", "") or ""),
            _message_sender_name(message),
        )
    for message in payload.get("overflow_messages", []) or []:
        if not isinstance(message, dict):
            continue
        add_sender(
            str(message.get("sender_id", "") or ""),
            _message_sender_name(message),
        )
    incoming = payload.get("incoming")
    if isinstance(incoming, dict):
        add_sender(
            str(incoming.get("sender_id", "") or ""),
            _message_sender_name(incoming),
        )
    for entry in payload.get("incoming_batch", []) or []:
        if not isinstance(entry, dict):
            continue
        sender_name = (
            str(entry.get("sender_name", "") or "")
            or str(entry.get("sender_username", "") or "")
            or str(entry.get("sender_display_name", "") or "")
            or str(entry.get("sender_full_name", "") or "")
        )
        add_sender(str(entry.get("sender_id", "") or ""), sender_name)

    return list(active.values())


def _message_sender_name(message: dict[str, Any]) -> str:
    return (
        str(message.get("sender", "") or "")
        or str(message.get("sender_username", "") or "")
        or str(message.get("sender_display_name", "") or "")
        or str(message.get("sender_full_name", "") or "")
    )


def _is_self_identity(sender_id: str, sender_name: str, *, persona: str, user_id: str) -> bool:
    sid = str(sender_id or "").strip()
    sname = str(sender_name or "").strip()
    if sid and user_id and sid == user_id:
        return True
    if sname and persona and name_matches(sname, persona):
        return True
    return False


def _repair_incoming_batch_and_incoming(payload: dict[str, Any]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    raw_batch = payload.get("incoming_batch", [])
    if not isinstance(raw_batch, list):
        return warnings

    ordered = _ordered_incoming_batch(raw_batch)
    payload["incoming_batch"] = ordered
    if not ordered:
        return warnings

    latest = ordered[-1]
    expected_sender_id = str(latest.get("sender_id", "") or "")
    expected_sender = (
        str(latest.get("sender_name", "") or "")
        or str(latest.get("sender_username", "") or "")
        or str(latest.get("sender_display_name", "") or "")
        or str(latest.get("sender_full_name", "") or "")
        or expected_sender_id
    )
    expected_text = str(latest.get("latest_text", "") or "")
    if not expected_text:
        texts = latest.get("texts", [])
        if isinstance(texts, list) and texts:
            expected_text = str(texts[-1] or "")

    incoming = payload.get("incoming")
    if not isinstance(incoming, dict):
        incoming = {}
        payload["incoming"] = incoming

    incoming_sender_id = str(incoming.get("sender_id", "") or "")
    incoming_sender = _message_sender_name(incoming)
    incoming_text = str(incoming.get("text", "") or "")
    sender_matches = (
        incoming_sender_id == expected_sender_id
        if expected_sender_id
        else name_matches(incoming_sender, expected_sender)
    )
    text_matches = incoming_text == expected_text if expected_text else bool(incoming_text)
    if sender_matches and text_matches:
        return warnings

    incoming["sender_id"] = expected_sender_id
    incoming["sender"] = expected_sender
    incoming["text"] = expected_text
    sender_username = str(latest.get("sender_username", "") or "")
    sender_display_name = str(latest.get("sender_display_name", "") or "")
    sender_full_name = str(latest.get("sender_full_name", "") or "")
    if sender_username:
        incoming["sender_username"] = sender_username
    if sender_display_name:
        incoming["sender_display_name"] = sender_display_name
    if sender_full_name:
        incoming["sender_full_name"] = sender_full_name
    if not incoming.get("timestamp"):
        latest_ts = _safe_float(latest.get("last_timestamp", 0.0))
        if latest_ts > 0.0:
            incoming["timestamp"] = latest_ts

    payload["incoming_sender_id"] = expected_sender_id
    people_facts = payload.get("people_facts", {})
    payload["incoming_sender_known"] = bool(
        expected_sender_id and isinstance(people_facts, dict) and expected_sender_id in people_facts
    )
    warnings.append(
        {
            "category": "incoming_repair",
            "reason": "incoming_aligned_with_latest_batch",
            "sender_id": expected_sender_id,
            "sender_name": expected_sender,
        }
    )
    return warnings


def _ordered_incoming_batch(incoming_batch: list[Any]) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    for idx, raw_entry in enumerate(incoming_batch):
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        arrival = _safe_float(entry.get("arrival_order", idx))
        last_ts = _safe_float(entry.get("last_timestamp", 0.0))
        if last_ts <= 0.0:
            last_ts = _safe_float(entry.get("first_timestamp", 0.0))
        if last_ts <= 0.0:
            timestamps = entry.get("timestamps", [])
            if isinstance(timestamps, list):
                valid = [_safe_float(value) for value in timestamps]
                valid = [value for value in valid if value > 0.0]
                if valid:
                    last_ts = max(valid)
        entry["last_timestamp"] = last_ts
        entry["_arrival_order"] = arrival
        sender_key = canonical_identity_key(
            str(entry.get("sender_id", "") or ""),
            str(entry.get("sender_name", "") or "") or str(entry.get("sender_username", "") or ""),
        )
        entry["_sender_key"] = sender_key
        ordered.append(entry)

    ordered.sort(
        key=lambda entry: (
            _safe_float(entry.get("last_timestamp", 0.0)),
            _safe_float(entry.get("_arrival_order", 0.0)),
            str(entry.get("_sender_key", "") or ""),
        )
    )
    for entry in ordered:
        entry.pop("_arrival_order", None)
        entry.pop("_sender_key", None)
    return ordered


def _repair_summary_meta(payload: dict[str, Any]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    summary_meta_raw = payload.get("summary_meta", {})
    if not isinstance(summary_meta_raw, dict):
        summary_meta_raw = {}
    summary_meta = dict(summary_meta_raw)
    now_ts = _parse_timestamp(payload.get("now_timestamp")) or time.time()

    range_start = _safe_float(summary_meta.get("range_start_ts", 0.0))
    range_end = _safe_float(summary_meta.get("range_end_ts", 0.0))
    if range_start > 0.0 and range_end > 0.0 and range_start > range_end:
        range_start, range_end = range_end, range_start
        warnings.append(
            {
                "category": "summary_range_clamp",
                "reason": "range_start_after_range_end",
                "range_start_ts": range_start,
                "range_end_ts": range_end,
            }
        )

    recent_time_range = payload.get("recent_time_range", {})
    recent_start = 0.0
    if isinstance(recent_time_range, dict):
        recent_start = _parse_timestamp(recent_time_range.get("start"))
    if recent_start > 0.0 and range_end > recent_start:
        range_end = recent_start
        if range_start > range_end and range_start > 0.0:
            range_start = range_end
        warnings.append(
            {
                "category": "summary_range_clamp",
                "reason": "range_end_after_recent_start",
                "recent_start_ts": recent_start,
                "range_end_ts": range_end,
            }
        )

    summary_meta["range_start_ts"] = float(range_start or 0.0)
    summary_meta["range_end_ts"] = float(range_end or 0.0)
    last_updated_ts = _safe_float(summary_meta.get("last_updated_ts", 0.0))
    if last_updated_ts > 0.0:
        summary_meta["age_seconds"] = max(0.0, now_ts - last_updated_ts)
    if range_end > 0.0:
        summary_meta["range_age_seconds"] = max(0.0, now_ts - range_end)
    payload["summary_meta"] = summary_meta
    return warnings


def _self_identity_warnings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    user_id = str(payload.get("user_id", "") or "").strip()
    persona = str(payload.get("persona", "") or "").strip()
    if user_id and not looks_like_uuid(user_id):
        warnings.append(
            {
                "category": "self_id_missing_uuid",
                "reason": "user_id_not_uuid",
                "user_id": user_id,
            }
        )

    if not persona:
        return warnings
    for message in (payload.get("recent_messages", []) or []) + (payload.get("overflow_messages", []) or []):
        if not isinstance(message, dict):
            continue
        sender_name = _message_sender_name(message)
        if not sender_name or not name_matches(sender_name, persona):
            continue
        sender_id = str(message.get("sender_id", "") or "").strip()
        if sender_id and not looks_like_uuid(sender_id):
            warnings.append(
                {
                    "category": "self_id_missing_uuid",
                    "reason": "self_message_sender_id_not_uuid",
                    "sender_id": sender_id,
                }
            )
            break
    return warnings


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_timestamp(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text == "0":
        return 0.0
    try:
        return float(text)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
    return float(parsed.timestamp())
