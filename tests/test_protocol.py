import json

from gpt5_roleplay_system.models import Action, CommandType
from gpt5_roleplay_system.protocol import build_chat_response, decode_message, encode_message


def test_decode_message_with_json_string_data():
    payload = {
        "type": "set_persona",
        "data": json.dumps({"persona": "Test"}),
        "id": "msg_1",
        "timestamp": 1,
    }
    msg = decode_message(json.dumps(payload))
    assert msg.msg_type == "set_persona"
    assert msg.data["persona"] == "Test"


def test_encode_message_includes_envelope():
    line = encode_message("status", {"status": "ok"}, message_id="msg_99")
    assert line.endswith("\n")
    payload = json.loads(line)
    assert payload["type"] == "status"
    assert payload["id"] == "msg_99"
    assert "timestamp" in payload


def test_build_chat_response():
    actions = [
        Action(command_type=CommandType.CHAT, content="Hello", parameters={"content": "Hello"})
    ]
    data = build_chat_response(actions)
    assert "commands" in data
    assert data["commands"][0]["type"] == "CHAT"
