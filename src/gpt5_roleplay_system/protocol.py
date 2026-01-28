from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict

from .models import Action
from .time_utils import get_pacific_timestamp


@dataclass
class InboundMessage:
    msg_type: str
    data: Dict[str, Any]
    raw: Dict[str, Any]


def _normalize_data(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except json.JSONDecodeError:
            return {"value": data}
    return {"value": data}


def decode_message(line: str) -> InboundMessage:
    line = line.strip()
    if not line:
        raise ValueError("empty line")
    raw = json.loads(line)
    msg_type = raw.get("type")
    if not msg_type:
        raise ValueError("missing message type")
    data = _normalize_data(raw.get("data", {}))
    return InboundMessage(msg_type=msg_type, data=data, raw=raw)


def _now_ts() -> int:
    return get_pacific_timestamp()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def encode_message(msg_type: str, data: Dict[str, Any], message_id: str | None = None) -> str:
    payload = {
        "type": msg_type,
        "data": data,
        "id": message_id or _new_id("msg"),
        "timestamp": _now_ts(),
    }
    return json.dumps(payload, ensure_ascii=True) + "\n"


def build_chat_response(actions: list[Action]) -> Dict[str, Any]:
    commands = []
    timestamp = _now_ts()
    for action in actions:
        commands.append(
            {
                "type": action.command_type.value,
                "content": action.content,
                "timestamp": timestamp,
                "x": action.x,
                "y": action.y,
                "z": action.z,
                "target_uuid": action.target_uuid,
                "parameters": action.parameters or {},
            }
        )
    return {"commands": commands}
