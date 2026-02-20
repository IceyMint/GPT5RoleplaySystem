from __future__ import annotations

from typing import Any

from .models import Action, CommandType


class ResponseMapper:
    def bundle_from_structured(self, parsed: Any, mode: str = "chat"):
        from .llm import LLMResponseBundle, ParticipantHint

        text = getattr(parsed, "text", "") or ""
        actions: list[Action] = []
        for action in getattr(parsed, "actions", []) or []:
            actions.extend(self._actions_from_structured(action))
        if not text and actions:
            text = self._first_chat_text(actions)

        facts = self.facts_from_structured(parsed)
        hints = [
            ParticipantHint(
                user_id=getattr(hint, "user_id", ""),
                name=getattr(hint, "name", ""),
            )
            for hint in getattr(parsed, "participant_hints", []) or []
        ]

        summary_update = getattr(parsed, "summary_update", None)
        summary = summary_update.strip() if isinstance(summary_update, str) and summary_update.strip() else None
        mood = getattr(parsed, "mood", None)
        status = getattr(parsed, "status", None)
        raw_decision = getattr(parsed, "autonomy_decision", None)
        if raw_decision is None:
            raw_decision = getattr(parsed, "decision", None)
        decision = self._normalize_autonomy_decision(raw_decision)
        next_delay_seconds = self._optional_positive_float(getattr(parsed, "next_delay_seconds", None))
        if mode == "autonomous":
            if actions:
                decision = "act"
            elif decision is None:
                decision = "wait"

        return LLMResponseBundle(
            text=text,
            actions=actions,
            facts=facts,
            participant_hints=hints,
            summary=summary,
            mood=mood.strip() if isinstance(mood, str) and mood.strip() else None,
            status=status.strip() if isinstance(status, str) and status.strip() else None,
            autonomy_decision=decision,
            next_delay_seconds=next_delay_seconds,
        )

    def state_update_from_structured(self, parsed: Any):
        from .llm import LLMStateUpdate, ParticipantHint

        facts = self.facts_from_structured(parsed)
        hints = [
            ParticipantHint(
                user_id=getattr(hint, "user_id", ""),
                name=getattr(hint, "name", ""),
            )
            for hint in getattr(parsed, "participant_hints", []) or []
        ]
        summary_update = getattr(parsed, "summary_update", None)
        summary = summary_update.strip() if isinstance(summary_update, str) and summary_update.strip() else None
        mood = getattr(parsed, "mood", None)
        status = getattr(parsed, "status", None)
        raw_decision = getattr(parsed, "autonomy_decision", None)
        if raw_decision is None:
            raw_decision = getattr(parsed, "decision", None)
        decision = self._normalize_autonomy_decision(raw_decision)
        next_delay_seconds = self._optional_positive_float(getattr(parsed, "next_delay_seconds", None))
        return LLMStateUpdate(
            facts=facts,
            participant_hints=hints,
            summary_update=summary,
            mood=mood.strip() if isinstance(mood, str) and mood.strip() else None,
            status=status.strip() if isinstance(status, str) and status.strip() else None,
            autonomy_decision=decision,
            next_delay_seconds=next_delay_seconds,
        )

    def facts_from_structured(self, parsed: Any):
        from .llm import ExtractedFact

        extracted = []
        for fact in getattr(parsed, "facts", []) or []:
            extracted.append(
                ExtractedFact(
                    user_id=getattr(fact, "user_id", ""),
                    name=getattr(fact, "name", ""),
                    facts=list(getattr(fact, "facts", []) or []),
                )
            )
        return extracted

    def _actions_from_structured(self, action: Any) -> list[Action]:
        primary = self._command_from_structured(action)
        if primary is None:
            return []
        param_type_raw = str(primary.parameters.get("type", "") or "").upper()
        if not param_type_raw or param_type_raw == primary.command_type.value:
            primary.parameters.pop("type", None)
            return [primary]
        primary.parameters.pop("type", None)
        try:
            param_command = CommandType(param_type_raw)
        except ValueError:
            return [primary]
        secondary_params = dict(primary.parameters)
        secondary_content = self._normalize_text_action_content(param_command, primary.content)
        secondary = Action(
            command_type=param_command,
            content=secondary_content,
            x=primary.x,
            y=primary.y,
            z=primary.z,
            target_uuid=primary.target_uuid,
            parameters=secondary_params,
        )
        if param_command in {CommandType.CHAT, CommandType.EMOTE} and secondary.content:
            secondary.parameters["content"] = secondary.content
        return [secondary, primary]

    def _command_from_structured(self, action: Any) -> Action | None:
        action_type = getattr(action, "type", "")
        if isinstance(action_type, CommandType):
            command_type = action_type
        else:
            raw_type = str(action_type).upper()
            try:
                command_type = CommandType(raw_type)
            except ValueError:
                return None
        parameters_raw = getattr(action, "parameters", {})
        parameters = dict(parameters_raw) if isinstance(parameters_raw, dict) else {}
        content = str(getattr(action, "content", ""))
        if command_type in {CommandType.CHAT, CommandType.EMOTE} and not content:
            fallback_content = parameters.get("content", parameters.get("text"))
            if fallback_content is not None:
                content = str(fallback_content)
        if command_type in {CommandType.CHAT, CommandType.EMOTE} and content:
            content = self._normalize_text_action_content(command_type, content)
            parameters["content"] = content
        if command_type in {CommandType.MOVE, CommandType.FACE_TARGET}:
            parameters.setdefault("x", str(getattr(action, "x", 0.0)))
            parameters.setdefault("y", str(getattr(action, "y", 0.0)))
            parameters.setdefault("z", str(getattr(action, "z", 0.0)))
        return Action(
            command_type=command_type,
            content=content,
            x=float(getattr(action, "x", 0.0) or 0.0),
            y=float(getattr(action, "y", 0.0) or 0.0),
            z=float(getattr(action, "z", 0.0) or 0.0),
            target_uuid=str(getattr(action, "target_uuid", "")),
            parameters=parameters,
        )

    @staticmethod
    def _first_chat_text(actions: list[Action]) -> str:
        for action in actions:
            if action.command_type == CommandType.CHAT and action.content:
                return action.content
        return ""

    @staticmethod
    def _strip_emote_wrapping_asterisks(content: str) -> str:
        cleaned = content.strip()
        while len(cleaned) >= 2 and cleaned.startswith("*") and cleaned.endswith("*"):
            cleaned = cleaned[1:-1].strip()
        return cleaned

    def _normalize_text_action_content(self, command_type: CommandType, content: str) -> str:
        text = str(content or "")
        if command_type != CommandType.EMOTE:
            return text
        return self._strip_emote_wrapping_asterisks(text)

    @staticmethod
    def _normalize_autonomy_decision(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        decision = value.strip().lower()
        if decision in {"act", "wait", "sleep"}:
            return decision
        return None

    @staticmethod
    def _optional_positive_float(value: Any) -> float | None:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            return None
        if candidate <= 0.0:
            return None
        return candidate
