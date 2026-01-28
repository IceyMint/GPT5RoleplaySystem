# Session State Persistence (Reconnect Continuity)

Goal: keep short-term state across viewer disconnect/reconnects.

Core code:

- State store: `src/gpt5_roleplay_system/session_state.py`
- Snapshot/restore: `src/gpt5_roleplay_system/pipeline.py:snapshot_state`
- Wiring: `src/gpt5_roleplay_system/controller.py`
- Flush on disconnect: `src/gpt5_roleplay_system/server.py`

## What Is Persisted

When `episode_summary.persist_state: true`, the server saves and restores:

- Rolling buffer contents
- Conversation memory:
  - recent messages
  - summary text
- Episode watermark metadata:
  - `last_episode_ts`
  - `last_episode_size`

## Where It Is Stored

File pattern:

- `<episode_summary.state_dir>/session_state_<persona>_<user_id>.json`

Slug rules:

- Non-alphanumeric characters are replaced with `_`
- Empty persona/user_id falls back to `default`

Default location:

- `.episode_state/session_state_<persona>_<user_id>.json`

## When It Saves

State is scheduled to save:

- after `process_chat`
- after `process_chat_batch`

State is flushed on disconnect:

- `SessionController.flush_state()` is called in the server's disconnect path

## Important Notes

- Persistence is per persona + user_id.
- Changing persona or user_id resets the state store key and reloads state for
  that new key.
- The pipeline resets activity timestamps to the latest restored message to
  avoid immediate inactivity-triggered episode flushes on reconnect.

