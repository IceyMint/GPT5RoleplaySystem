# Architecture Overview

This rewrite is server-first and session-isolated. Each TCP connection gets its
own controller, memory, rolling buffer, and episodic state.

Key entry points:

- Server: `src/gpt5_roleplay_system/server.py`
- Session routing + batching: `src/gpt5_roleplay_system/session.py`
- Orchestration + persistence: `src/gpt5_roleplay_system/controller.py`
- Core pipeline: `src/gpt5_roleplay_system/pipeline.py`

## High-Level Flow (Per Connection)

1) TCP accept -> create session objects

- `GPT5RoleplayServer._handle_client(...)` creates:
  - a `SessionController`
  - a `ClientSession`
  - an inbound `asyncio.Queue`
  - a worker task: `session.process_queue(queue)`

2) Inbound messages -> queue

- The server reads NDJSON lines, decodes via `decode_message(...)`, and calls
  `_enqueue_message(...)`.
- Backpressure policy:
  - queue size: `queue_max_size`
  - drop policy: `drop_oldest` (default) or `drop_newest`

3) Queue worker -> batching -> pipeline

`ClientSession.process_queue(...)` handles two modes:

- Non-batched:
  - non-`process_chat` messages
  - or batching disabled (`batch_window_seconds <= 0` or `batch_max_size <= 1`)
- Batched:
  - collects `process_chat` messages during a short window
  - runs one `process_chat_batch(...)`

4) Pipeline stages

At a high level, `MessagePipeline.process_chat_batch(...)` does:

- Build normalized `InboundChat` objects.
- Filter self messages (prevents self-reply loops).
- Update rolling buffer + short-term memory.
- Build context (participants, environment, facts, summary, experiences).
- Addressed-to-me classification.
- Generate structured bundle (text + actions + facts + hints + summary update).
- Apply summary compression and schedule fact storage.
- Schedule episodic experience detection.

## Autonomy and Status Loops

When enabled (`autonomy.enabled: true`), the server starts:

- Autonomy loop: `_autonomy_loop(...)`
  - schedules `autonomous_tick` messages onto the session queue
  - uses activity-aware delay backoff
- Status loop: `_status_loop(...)`
  - emits `status` payloads
  - optionally emits a `CHAT` action to `autonomy.status_channel`

Autonomy runs through the same queue + session pipeline entry points as regular
chat, which keeps ordering and write locking consistent.

## Session Isolation Model

Each TCP connection is fully isolated:

- Dedicated controller + pipeline
- Dedicated rolling buffer and memory summary
- Dedicated episodic state (watermark / timers)
- Shared knowledge store driver is fine, but in-memory state is per session

This means a reconnect normally resets state, unless session persistence is
enabled via `episode_summary.persist_state` (see `docs/STATE_PERSISTENCE.md`).

