from __future__ import annotations

from dataclasses import dataclass, field

from .models import EnvironmentSnapshot, Participant


@dataclass
class PipelineRuntimeState:
    persona: str
    user_id: str
    llm_chat_enabled: bool = True
    environment: EnvironmentSnapshot = field(default_factory=EnvironmentSnapshot)
    participant_hints: list[Participant] = field(default_factory=list)
    display_names_by_id: dict[str, str] = field(default_factory=dict)
