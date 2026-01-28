# Development Workflows and "Where to Look"

When something feels off, this page should help you jump directly to the right
file and function.

## Run + Test

- Run server:
  - `python -m gpt5_roleplay_system.server --host 0.0.0.0 --port 9999 --config /path/to/config.yaml`
- Run tests:
  - `python -m pytest -q`
- Full script:
  - `python scripts/test_all_parts.py`

## Search Tips

Fast code search:

- `rg "keyword" src tests docs -S`
- `rg -n "function_name|event_name|message_type" src -S`

## Feature Map (File -> Responsibility)

### TCP server + session lifecycle

- `src/gpt5_roleplay_system/server.py`
  - session creation
  - queue wiring
  - autonomy/status loops
  - disconnect cleanup and state flush

### Message routing + batching

- `src/gpt5_roleplay_system/session.py`
  - `process_queue(...)`
  - batching behavior
  - write locking (`_write_lock`)

### Orchestration + persistence

- `src/gpt5_roleplay_system/controller.py`
  - session state persistence
  - persona/user_id lifecycle
  - entry points used by `ClientSession`

### Core behavior (most changes land here)

- `src/gpt5_roleplay_system/pipeline.py`
  - self filtering
  - context building
  - addressed-to-me classification
  - LLM prompt/response logging
  - episodic experience creation

### Protocol layout

- `src/gpt5_roleplay_system/protocol.py`
  - envelope
  - `chat_response` command schema

### LLM behavior + schemas

- `src/gpt5_roleplay_system/llm.py`
  - structured bundle schema
  - address classifier prompt
  - autonomy prompt

### Memory and state

- `src/gpt5_roleplay_system/memory.py`
  - rolling buffer
  - conversation memory
- `src/gpt5_roleplay_system/session_state.py`
  - persistence format and file location

### Knowledge + Neo4j + vectors

- `src/gpt5_roleplay_system/neo4j_store.py`
  - facts and person lookups
- `src/gpt5_roleplay_system/neo4j_experience_vector.py`
  - Neo4j vector index integration
- `src/gpt5_roleplay_system/experience_vector.py`
  - embedding client and local vector index

### Observability

- `src/gpt5_roleplay_system/observability.py`
  - W&B tracer
  - Weave initialization

## Common Change Recipes

### "AI responds when it shouldn't"

Check in this order:

1) `llm_address_check` and `llm_address_result` events in W&B
2) address classifier prompt in `llm.py`
3) participant extraction in `pipeline.py`

### "Experiences are wrong"

- Episode boundaries: `pipeline.py:_maybe_finalize_episode`
- Episode settings: `config.yaml:episode_summary`
- Session state: `docs/STATE_PERSISTENCE.md`

### "Status channel not honored"

- Server: `server.py:_status_loop`
- Protocol: `protocol.py:build_chat_response`
- Viewer: see `docs/VIEWER_INTEGRATION.md`

