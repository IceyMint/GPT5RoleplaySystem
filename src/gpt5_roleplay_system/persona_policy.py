from __future__ import annotations


def is_chat_and_state_restricted_persona(persona: str) -> bool:
    return isinstance(persona, str) and persona.strip().casefold() == "defaultpersona"
