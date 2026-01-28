# NDJSON Protocol Reference

Transport is newline-delimited JSON (NDJSON). Each message is a single JSON
object followed by `\n`.

Envelope fields are preserved:

- `type`: message type string
- `data`: message payload (object recommended)
- `id`: message id (server generates if not provided)
- `timestamp`: unix seconds (server generates if not provided)

Core implementation:

- Decode: `src/gpt5_roleplay_system/protocol.py:decode_message`
- Encode: `src/gpt5_roleplay_system/protocol.py:encode_message`

## Inbound Messages (Viewer -> Server)

### `process_chat`

Minimal shape:

```json
{"type":"process_chat","data":{"text":"Hello","from_name":"Evie","from_id":"uuid","timestamp":1738000000}}
```

Common fields:

- `text`: chat content
- `from_name`: speaker display name
- `from_id`: speaker UUID (preferred)
- `timestamp`: unix seconds (float or int)
- `participants`: optional list of `{user_id,name}` hints
- `logged_in_agent`: optional AI persona name (used for self-filtering)

### `environment_update`

Minimal shape:

```json
{"type":"environment_update","data":{"agents":[],"objects":[],"location":"Region","avatar_position":"(0,0,0)","timestamp":1738000000}}
```

Notes:

- Agents and objects are free-form dicts but should include:
  - `name`
  - `position`
  - `uuid` or `target_uuid`
- The viewer should exclude the AI's own avatar from `agents`.

### `set_user_id`

Sets the AI's own UUID for the connection:

```json
{"type":"set_user_id","data":{"user_id":"ai-uuid"}}
```

Only affects the sending TCP session.

## Outbound Messages (Server -> Viewer)

### `chat_response`

The server returns commands in the established Firestorm-friendly layout:

```json
{
  "type": "chat_response",
  "data": {
    "commands": [
      {
        "type": "CHAT",
        "content": "Hello there.",
        "timestamp": 1738000000,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "target_uuid": "",
        "parameters": {"channel": -9001}
      }
    ]
  }
}
```

Important:

- `parameters` is the extension point for extra fields.
- Channel routing uses `parameters.channel`.
- Viewer code must parse `parameters` in the `chat_response` handler.

### `status`

Periodic status payload (mainly for diagnostics/UI):

```json
{
  "type": "status",
  "data": {
    "session_id": "session_1",
    "persona": "DefaultPersona",
    "user_id": "ai-uuid",
    "seconds_since_activity": 12.3,
    "recent_messages": 8,
    "recent_activity_window_seconds": 45.0,
    "last_inbound_ts": 1700000000.0,
    "last_response_ts": 1700000012.0,
    "mood": "neutral",
    "mood_ts": 1700000012.0,
    "status": "idle",
    "status_ts": 1700000012.0,
    "autonomy_enabled": true
  }
}
```

The viewer may ignore this, display it in UI, or route it to a dedicated chat
channel on its side.

## Internal Messages

The server also schedules internal queue messages:

- `autonomous_tick`: used by the autonomy loop

These are not required viewer message types.
