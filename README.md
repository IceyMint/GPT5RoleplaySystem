# GPT5RoleplaySystem

Clean-room, server-first rewrite of the Genesis AI roleplay system. This version runs as a TCP **server** so multiple Firestorm viewers can connect concurrently. The code is modular and testable, with individual components designed for unit testing.

Docs index: `docs/INDEX.md`

## Goals

- Multi-client TCP server (NDJSON protocol compatible with Firestorm).
- Pluggable controller and message pipeline.
- Memory + Neo4j knowledge store with similarity search hooks.
- Episodic experience storage (not per-message).
- Memory compression for older conversation context.
- Optional Weights & Biases tracing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional dependencies (install if you plan to use them):

```bash
pip install wandb neo4j
```

LLM structured outputs (OpenRouter via OpenAI client):

```bash
pip install openai pydantic
```

## Run the server

```bash
python -m gpt5_roleplay_system.server --host 0.0.0.0 --port 9999 --config /path/to/config.yaml
```

## Test all parts

```bash
python scripts/test_all_parts.py
```

## Protocol

The server speaks newline-delimited JSON. It preserves the envelope fields `type`, `data`, `id`, `timestamp` and supports the `chat_response` commands schema from the existing Firestorm protocol.

Message types used by the server:

- `process_chat`: inbound chat messages from the viewer
- `environment_update`: environment snapshots from the viewer
- `chat_response`: outbound commands for the viewer to execute
- `status`: outbound status payloads for UI/diagnostics

## Session model (multi-client)

- Each TCP connection maps to an isolated session with its own controller and memory.
- `set_user_id` applies only to the connection that sent it; persona is fixed per session.
- `process_chat` can include optional `participants` (list of `{user_id,name}`) to avoid LLM-based name detection.
- Environment updates are stored per session; only the session that sent them sees them.
- Session state (rolling buffer + summary) can persist across reconnects when enabled.

## Config

- If `--config` is not provided, the server checks `GPT5_ROLEPLAY_CONFIG`, then `./config.yaml`.
- For compatibility, it will also look for a sibling `qwenRoleplayAISystem/config.yaml` when no local file exists.
- Use `config.example.yaml` as a template; avoid committing real API keys.
- Backpressure settings: `GPT5_ROLEPLAY_QUEUE_MAX` and `GPT5_ROLEPLAY_QUEUE_DROP` (`drop_oldest` or `drop_newest`).
- Chat batching settings: `GPT5_ROLEPLAY_BATCH_WINDOW_MS` and `GPT5_ROLEPLAY_BATCH_MAX`.
- Episodic experience settings live under `episode_summary`.
- Autonomy status chat routing is controlled by `autonomy.status_channel` (non-zero recommended).

Key config blocks:

- `knowledge_storage`
  - `experience_similar_limit`, `experience_score_min`, `experience_score_delta`
- `episode_summary`
  - `enabled`: episodic experience consolidation on/off
  - `min_messages`, `max_messages`: thresholds for episode boundaries
  - `inactivity_seconds`: idle time that can trigger an episode boundary
  - `forced_interval_seconds`: maximum time between episode boundaries
  - `overlap_messages`: trailing messages kept after an episode flush
  - `persist_state`: persist rolling buffer + summary across reconnects
  - `state_dir`: directory for persisted session state (default: `.episode_state`)
- `autonomy`
  - `enabled`: autonomy loop + status updates on/off
  - `status_interval_seconds`: status emission cadence
  - `status_channel`: when set, emits a `CHAT` command to that channel

Persisted session state:

- When `episode_summary.persist_state: true`, the server saves and restores:
  - rolling buffer contents
  - conversation memory summary + recent items
  - episode watermark metadata
- Files are stored at:
  - `.episode_state/session_state_<persona>_<user_id>.json`

## Structure

- `src/gpt5_roleplay_system/server.py` - TCP server, multi-client connection handling.
- `src/gpt5_roleplay_system/session.py` - Per-connection state + routing.
- `src/gpt5_roleplay_system/controller.py` - Orchestrates pipeline and persona/user settings.
- `src/gpt5_roleplay_system/pipeline.py` - Message handling, context building, LLM integration.
- `src/gpt5_roleplay_system/memory.py` - Rolling buffer and compressed memory.
- `src/gpt5_roleplay_system/session_state.py` - Session state persistence across reconnects.
- `src/gpt5_roleplay_system/neo4j_store.py` - Knowledge store interface and in-memory fallback.
- `src/gpt5_roleplay_system/llm.py` - LLM interface + rule-based fallback.
- `src/gpt5_roleplay_system/protocol.py` - NDJSON encode/decode + envelope.
- `src/gpt5_roleplay_system/observability.py` - Optional wandb tracing.

## Notes on behavior

- Addressed-to-me classification includes conversation summary and recent messages, not just the latest text.
- Experiences are created on episodic boundaries (buffer size / inactivity / forced interval), then embedded and indexed.
- Status updates can optionally be routed to a non-local chat channel via `autonomy.status_channel`.
