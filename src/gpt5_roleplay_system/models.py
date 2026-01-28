from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CommandType(str, Enum):
    CHAT = "CHAT"
    EMOTE = "EMOTE"
    MOVE = "MOVE"
    TOUCH = "TOUCH"
    SIT = "SIT"
    STAND = "STAND"
    LOOK_AT = "LOOK_AT"
    WALK_TO = "WALK_TO"
    TURN_TO = "TURN_TO"
    GESTURE = "GESTURE"
    FOLLOW = "FOLLOW"


@dataclass
class Action:
    command_type: CommandType
    content: str = ""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    target_uuid: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InboundChat:
    text: str
    sender_id: str
    sender_name: str
    timestamp: float
    raw: Dict[str, Any]


@dataclass
class Participant:
    user_id: str
    name: str


@dataclass
class EnvironmentSnapshot:
    agents: List[Dict[str, Any]] = field(default_factory=list)
    objects: List[Dict[str, Any]] = field(default_factory=list)
    location: str = ""
    avatar_position: str = ""


@dataclass
class ConversationContext:
    persona: str
    user_id: str
    environment: EnvironmentSnapshot
    participants: List[Participant]
    people_facts: Dict[str, Any]
    recent_messages: List[InboundChat]
    summary: str
    related_experiences: List[Dict[str, Any]]
    summary_meta: Dict[str, Any] = field(default_factory=dict)
    agent_state: Dict[str, Any] = field(default_factory=dict)
    persona_instructions: str = ""
