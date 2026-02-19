from __future__ import annotations

import time
from typing import Any

from .models import Action, CommandType, Participant
from .name_utils import normalize_for_match
from .pipeline_state import PipelineRuntimeState
from .time_utils import format_pacific_time


class AutonomyManager:
    def __init__(self, state: PipelineRuntimeState) -> None:
        self._state = state
        now = time.time()
        self._last_inbound_ts = now
        self._last_response_ts = now
        self._current_mood = "neutral"
        self._current_mood_ts = now
        self._current_status = "idle"
        self._current_status_ts = now
        self._autonomy_delay_hint_seconds: float | None = None
        self._autonomy_decision = "wait"
        self._mood_source = "init"
        self._status_source = "init"

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "mood": self._current_mood,
            "mood_ts": self._current_mood_ts,
            "mood_source": self._mood_source,
            "status": self._current_status,
            "status_ts": self._current_status_ts,
            "status_source": self._status_source,
            "last_inbound_ts": self._last_inbound_ts,
            "last_response_ts": self._last_response_ts,
            "autonomy_decision": self._autonomy_decision,
            "autonomy_delay_hint_seconds": self._autonomy_delay_hint_seconds,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        mood = state.get("mood", self._current_mood)
        status = state.get("status", self._current_status)
        self._current_mood = str(mood or self._current_mood)
        self._current_status = str(status or self._current_status)
        self._current_mood_ts = float(state.get("mood_ts", self._current_mood_ts) or self._current_mood_ts)
        self._current_status_ts = float(state.get("status_ts", self._current_status_ts) or self._current_status_ts)
        self._mood_source = str(state.get("mood_source", self._mood_source) or self._mood_source)
        self._status_source = str(state.get("status_source", self._status_source) or self._status_source)
        self._last_inbound_ts = float(state.get("last_inbound_ts", self._last_inbound_ts) or self._last_inbound_ts)
        self._last_response_ts = float(state.get("last_response_ts", self._last_response_ts) or self._last_response_ts)
        decision = self._normalize_autonomy_decision_value(state.get("autonomy_decision"))
        if decision is not None:
            self._autonomy_decision = decision
        self._autonomy_delay_hint_seconds = self._sanitize_autonomy_delay_hint(
            state.get("autonomy_delay_hint_seconds")
        )

    def mark_inbound_activity(self, now_ts: float | None = None) -> None:
        self._last_inbound_ts = float(now_ts if now_ts is not None else time.time())

    def mark_response_activity(self, now_ts: float | None = None) -> None:
        self._last_response_ts = float(now_ts if now_ts is not None else time.time())

    def update_from_bundle(self, bundle: Any, *, source: str) -> None:
        now = time.time()
        mood = self._clean_optional_text(getattr(bundle, "mood", None))
        status = self._clean_optional_text(getattr(bundle, "status", None))
        if mood:
            self._current_mood = mood
            self._current_mood_ts = now
            self._mood_source = source
        status_candidate = status or self._derive_status_from_actions(getattr(bundle, "actions", []), getattr(bundle, "text", ""))
        if status_candidate:
            self._current_status = status_candidate
            self._current_status_ts = now
            self._status_source = source

    def apply_scheduler_override(
        self,
        decision_value: Any,
        delay_value: Any,
    ) -> tuple[str | None, float | None]:
        decision = self._normalize_autonomy_decision_value(decision_value)
        delay_hint = self._sanitize_autonomy_delay_hint(delay_value)
        if decision is None and delay_hint is None:
            return None, None
        if decision is not None:
            self._autonomy_decision = decision
            if delay_hint is None:
                self._autonomy_delay_hint_seconds = None
        if delay_hint is not None:
            self._autonomy_delay_hint_seconds = delay_hint
        return decision, delay_hint

    def filter_autonomous_actions(
        self,
        actions: list[Action],
        participants: list[Participant],
    ) -> list[Action]:
        if not actions:
            return []
        persona_key = normalize_for_match(self._state.persona)
        participant_keys = {
            normalize_for_match(participant.user_id or participant.name)
            for participant in participants
            if participant.user_id or participant.name
        }
        participant_keys.discard("")
        wait_tokens = ("wait", "waiting", "waited")
        filtered: list[Action] = []
        for action in actions:
            if action.command_type not in {CommandType.CHAT, CommandType.EMOTE}:
                filtered.append(action)
                continue
            content = (action.content or "").strip()
            if not content:
                continue
            content_lower = content.casefold()
            content_key = normalize_for_match(content)
            mentions_persona = bool(persona_key and persona_key in content_key)
            mentions_wait = any(token in content_lower for token in wait_tokens)
            mentions_other = any(key and key in content_key for key in participant_keys)
            if mentions_persona and mentions_wait and not mentions_other:
                continue
            filtered.append(action)
        return filtered

    @staticmethod
    def chat_texts_from_actions(actions: list[Action]) -> list[str]:
        texts: list[str] = []
        for action in actions:
            if action.command_type not in {CommandType.CHAT, CommandType.EMOTE}:
                continue
            content = (action.content or "").strip()
            if content:
                texts.append(content)
        return texts

    def seconds_since_activity(self, now: float | None = None) -> float:
        current = float(now if now is not None else time.time())
        last_activity = max(self._last_inbound_ts, self._last_response_ts)
        return max(0.0, current - last_activity)

    def activity_snapshot(
        self,
        recent_activity_window_seconds: float,
        *,
        recent_messages_count: int,
    ) -> dict[str, Any]:
        seconds = self.seconds_since_activity()
        return {
            "seconds_since_activity": seconds,
            "last_inbound_ts": self._last_inbound_ts,
            "last_response_ts": self._last_response_ts,
            "recent_messages": int(recent_messages_count),
            "recent_activity_window_seconds": recent_activity_window_seconds,
            "mood": self._current_mood,
            "mood_ts": self._current_mood_ts,
            "mood_source": self._mood_source,
            "status": self._current_status,
            "status_ts": self._current_status_ts,
            "status_source": self._status_source,
            "autonomy_decision": self._autonomy_decision,
            "autonomy_delay_hint_seconds": self._autonomy_delay_hint_seconds,
        }

    def agent_state(self, now_ts: float | None = None) -> dict[str, Any]:
        current = float(now_ts if now_ts is not None else time.time())
        mood_ts = float(self._current_mood_ts or 0.0)
        status_ts = float(self._current_status_ts or 0.0)
        mood_seconds_ago = (current - mood_ts) if mood_ts > 0.0 else None
        status_seconds_ago = (current - status_ts) if status_ts > 0.0 else None
        return {
            "mood": self._current_mood,
            "mood_ts": format_pacific_time(mood_ts),
            "mood_seconds_ago": mood_seconds_ago,
            "mood_source": self._mood_source,
            "status": self._current_status,
            "status_ts": format_pacific_time(status_ts),
            "status_seconds_ago": status_seconds_ago,
            "status_source": self._status_source,
            "last_message_received_at": format_pacific_time(self._last_inbound_ts),
            "last_ai_response_at": format_pacific_time(self._last_response_ts),
            "seconds_since_activity": self.seconds_since_activity(current),
            "autonomy_decision": self._autonomy_decision,
            "autonomy_delay_hint_seconds": self._autonomy_delay_hint_seconds,
        }

    def consume_delay_hint_seconds(self) -> float | None:
        hint = self._autonomy_delay_hint_seconds
        self._autonomy_delay_hint_seconds = None
        return hint

    @property
    def last_inbound_ts(self) -> float:
        return self._last_inbound_ts

    @property
    def last_response_ts(self) -> float:
        return self._last_response_ts

    @property
    def mood(self) -> str:
        return self._current_mood

    @property
    def mood_ts(self) -> float:
        return self._current_mood_ts

    @property
    def mood_source(self) -> str:
        return self._mood_source

    @property
    def status(self) -> str:
        return self._current_status

    @property
    def status_ts(self) -> float:
        return self._current_status_ts

    @property
    def status_source(self) -> str:
        return self._status_source

    @property
    def autonomy_decision(self) -> str:
        return self._autonomy_decision

    def set_activity_from_timestamp(self, timestamp: float) -> None:
        if timestamp > 0.0:
            self._last_inbound_ts = timestamp
            self._last_response_ts = timestamp

    @staticmethod
    def normalize_autonomy_decision(value: Any, has_actions: bool) -> str:
        if isinstance(value, str):
            decision = value.strip().lower()
            if decision in {"act", "wait", "sleep"}:
                if has_actions and decision != "act":
                    return "act"
                if decision == "act" and not has_actions:
                    return "wait"
                return decision
        return "act" if has_actions else "wait"

    @staticmethod
    def _sanitize_autonomy_delay_hint(value: Any) -> float | None:
        try:
            delay = float(value)
        except (TypeError, ValueError):
            return None
        if delay <= 0.0:
            return None
        return delay

    @staticmethod
    def _normalize_autonomy_decision_value(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        decision = value.strip().lower()
        if decision in {"act", "wait", "sleep"}:
            return decision
        return None

    @staticmethod
    def _clean_optional_text(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        cleaned = value.strip()
        return cleaned if cleaned else ""

    def _derive_status_from_actions(self, actions: list[Action], text: str) -> str:
        if actions:
            return self._status_from_command(actions[0].command_type)
        if text:
            return "chatting"
        return ""

    @staticmethod
    def _status_from_command(command_type: CommandType) -> str:
        mapping = {
            CommandType.CHAT: "chatting",
            CommandType.EMOTE: "emoting",
            CommandType.MOVE: "moving",
            CommandType.TOUCH: "interacting",
            CommandType.SIT: "sitting",
            CommandType.STAND: "standing",
            CommandType.FACE_TARGET: "facing",
        }
        return mapping.get(command_type, "active")
