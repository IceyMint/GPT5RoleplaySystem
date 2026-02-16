# Firestorm Viewer Integration Contract

This server assumes the viewer runs as a TCP client and speaks the NDJSON
protocol described in `docs/PROTOCOL.md`.

Primary viewer files (outside this repo):

- `/run/media/snow/Games/Software/source/phoenix-firestorm/indra/newview/fsaiintegration.cpp`
- `/run/media/snow/Games/Software/source/phoenix-firestorm/indra/newview/fsaiintegration.h`
- `/run/media/snow/Games/Software/source/phoenix-firestorm/indra/newview/python_bridge/py_ai_process_bridge.cpp`
- `/run/media/snow/Games/Software/source/phoenix-firestorm/indra/newview/python_bridge/py_ai_process_bridge.h`
- `/run/media/snow/Games/Software/source/phoenix-firestorm/indra/newview/fsfloaternearbychat.cpp`

## Required Behavior

### 1) Viewer must be the TCP client

- The server listens on `host:port` (default `0.0.0.0:9999`)
- Viewer should connect to `127.0.0.1:9999`
- The viewer should not try to open a TCP server on that port

### 2) Enable chat output after connect

Each new server session starts with LLM chat output disabled.

Viewer requirement:

- Send `{"type":"set_llm_chat_enabled","data":{"enabled":true}}` after establishing the socket.

Why it matters:

- Without this handshake, no `chat_response` actions are emitted for normal chat.

### 3) `chat_response.parameters` must be parsed

The server uses `parameters` to pass extra command metadata, especially
`parameters.channel`.

Viewer requirements:

- In the `chat_response` handler, parse `cmd["parameters"]` into the command
  object passed to command handlers.

Why it matters:

- Status/mood routing uses `parameters.channel`
- Future metadata will also travel through `parameters`

### 4) Channel routing must be honored

Command handlers should honor `parameters.channel`:

- For `CHAT` and `EMOTE`, pass the resolved channel into:
  - `FSNearbyChat::sendChatFromViewer(wtext, out_text, type, animate, channel)`

Recommended behavior:

- If `parameters.channel` is missing or invalid, fall back to channel `0`

### 5) Exclude the AI's own avatar from nearby agents

The viewer should not include the AI's own avatar in the nearby agents list.
This prevents confusing participant extraction and context pollution.

In `gatherNearbyAgents(...)`:

- Compare agent UUIDs to `gAgent.getID()`
- Skip the matching UUID

### 6) Support `FACE_TARGET` for avatar body rotation

The server may emit `FACE_TARGET` when the AI should rotate to face someone or
something without moving.

Viewer requirements:

- Accept `type: "FACE_TARGET"` in `chat_response.commands`.
- Resolve facing target from:
  - `target_uuid` when available, otherwise
  - `x`/`y`/`z` coordinates.
- Rotate avatar body orientation in place (do not treat this as `FOLLOW`).

## Optional Behavior

### `status` message handling

The server emits `status` messages for UI/diagnostics. The viewer can:

- ignore them
- display them in UI
- route them to a dedicated channel

Status payloads include activity metrics plus mood/status fields and timestamps.

Status channel routing can also be done on the server via
`autonomy.status_channel`.

## Known Viewer Pitfalls and Fixes

### Name collision with `shutdown(...)`

If you see a compile error like:

- "no matching function for call to AIProcessBridge::shutdown(socket_t&, ...)"

Fix:

- Call the global function explicitly: `::shutdown(...)`

### Viewer freeze on disable toggle

Disable should:

- stop AI processing
- disconnect TCP client
- avoid blocking the UI thread

If freezes happen, check for blocking joins or socket calls on the UI thread.
