# Config Reference (`config.yaml`)

Config loading lives in `src/gpt5_roleplay_system/config.py:load_config`.

Resolution order:

1) `--config /path/to/config.yaml`
2) `GPT5_ROLEPLAY_CONFIG`
3) `./config.yaml`
4) `../qwenRoleplayAISystem/config.yaml` (compat fallback)

## Top-Level Keys

- `ai_name`: persona display name used by the server
- `user_id`: AI's own UUID (can also be set at runtime via `set_user_id`)
- `queue_max_size`: inbound queue size per session
- `queue_drop_policy`: `drop_oldest` (default) or `drop_newest`
- `chat_batch_window_ms`: batching window for rapid chat bursts
- `chat_batch_max`: max number of chat messages batched into one LLM call

## `api_keys`

Supported keys:

- `openrouter_api_key`: used as the primary LLM key
- `openai_api_key`: used for embeddings / Neo4j GenAI token
- `wandb_api_key`: used for W&B logging

Environment variables override these:

- `OPENROUTER_API_KEY`
- `OPENAI_API_KEY`
- `WANDB_API_KEY`

## `llm`

Primary LLM settings:

- `base_url`: default `https://openrouter.ai/api/v1`
- `thinking_model`: preferred; used when present
- `models`: compatibility fallback map; first useful entry is selected
- `address_model`: cheap/fast model for addressed-to-me classification
- `embedding_model`: embedding model id (for similarity search)
- `embedding_dimensions`: vector dimension (must match DB index)
- `max_tokens`: completion cap (kept at 500 by design)
- `temperature`: generation temperature
- `timeout`: request timeout seconds

Environment overrides:

- `GPT5_ROLEPLAY_LLM_BASE_URL`
- `GPT5_ROLEPLAY_LLM_MODEL`
- `GPT5_ROLEPLAY_LLM_ADDRESS_MODEL`
- `GPT5_ROLEPLAY_LLM_EMBEDDING_MODEL`
- `GPT5_ROLEPLAY_LLM_EMBEDDING_DIMENSIONS`
- `GPT5_ROLEPLAY_LLM_MAX_TOKENS`
- `GPT5_ROLEPLAY_LLM_TEMPERATURE`
- `GPT5_ROLEPLAY_LLM_TIMEOUT`

## `knowledge_storage`

Experience retrieval knobs:

- `context_window_size`: used as a fallback for `memory.max_recent_messages`
- `experience_similar_limit`: top-k retrieved experiences
- `experience_score_min`: minimum similarity score gate
- `experience_score_delta`: max allowed drop from the top score

## `episode_summary`

This drives episodic experience creation and session persistence.

- `enabled`: episodic experience consolidation on/off
- `min_messages`: minimum rolling-buffer size before an episode can trigger
- `max_messages`: hard trigger when rolling buffer reaches this size
- `inactivity_seconds`: idle time that can trigger episode completion
- `forced_interval_seconds`: force a boundary after this much time
- `overlap_messages`: keep this many trailing messages after episode flush
- `persist_state`: persist rolling buffer + memory summary across reconnects
- `state_dir`: directory for persisted session files

Persisted session file pattern:

- `<state_dir>/session_state_<persona>_<user_id>.json`

## `autonomy`

Autonomy + status loop:

- `enabled`: turn autonomy loop on/off
- `base_delay_seconds`: default autonomy cadence
- `min_delay_seconds`, `max_delay_seconds`: clamp range
- `recent_activity_window_seconds`: "recent activity" window
- `recent_activity_multiplier`: delay multiplier during recent activity
- `status_interval_seconds`: cadence for status emissions
- `status_channel`: when set, also emits a `CHAT` command to that channel

## `database`

Neo4j connection:

- `uri`: e.g., `bolt://localhost:7687`
- `user`: Neo4j username
- `password`: Neo4j password
- `name`: Neo4j database name (no underscores)

Environment overrides:

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`

## `wandb`

W&B logging:

- `enabled`: enable/disable tracer
- `project`: W&B project name
- `run_name`: optional run name

When enabled and installed, Weave is also initialized automatically.

