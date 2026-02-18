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
- `max_environment_participants`: cap for nearby agents included in prompt context
- `persona_profiles`: optional map of persona name -> profile text (case-insensitive keying)

## `api_keys`

Supported keys:

- `openrouter_api_key`: used as the primary LLM key
- `embedding_api_key`: optional embedding key (falls back to `openai_api_key`, then `OPENAI_API_KEY`)
- `openai_api_key`: default OpenAI key used by Neo4j GenAI (and as embedding fallback)
- `wandb_api_key`: used for W&B logging

Environment variables override these:

- `OPENROUTER_API_KEY`
- `GPT5_ROLEPLAY_LLM_EMBEDDING_API_KEY`
- `OPENAI_API_KEY`
- `NEO4J_GENAI_API_KEY`
- `NEO4J_GENAI_PROVIDER`
- `WANDB_API_KEY`

## `llm`

Primary LLM settings:

- `base_url`: default `https://openrouter.ai/api/v1`
- `embedding_base_url`: optional base URL override for embeddings client
- `thinking_model`: preferred; used when present
- `models`: compatibility fallback map; first useful entry is selected
- `bundle_model`: optional override used only for `generate_bundle` (main chat reply path)
- `summary_model`: optional override used only for conversation summarization
- `facts_model`: optional override used for durable fact extraction and facts deduplication refinement
- `address_model`: cheap/fast model for addressed-to-me classification
- `provider.order`: optional OpenRouter provider priority list for roleplay/autonomy model calls (example: `["siliconflow"]`)
- `provider.allow_fallbacks`: optional OpenRouter provider fallback flag (`false` keeps routing pinned to `provider.order`)
- `embedding_model`: embedding model id (for similarity search)
- `neo4j_genai_api_key`: optional explicit key for Neo4j `genai.vector.encode` (defaults to `openai_api_key` / `OPENAI_API_KEY`)
- `neo4j_genai_provider`: provider passed to Neo4j GenAI (default: `OpenAI`)
- `neo4j_genai_only`: when `true`, disables external embedding fallback and requires Neo4j GenAI to be available
- `embedding_dimensions`: vector dimension (must match DB index)
- `max_tokens`: completion cap for model responses
- `temperature`: generation temperature
- `timeout`: request timeout seconds
- `reasoning`: optional OpenRouter reasoning effort level (for supported models)

Environment overrides:

- `GPT5_ROLEPLAY_LLM_BASE_URL`
- `GPT5_ROLEPLAY_LLM_MODEL`
- `GPT5_ROLEPLAY_LLM_BUNDLE_MODEL`
- `GPT5_ROLEPLAY_LLM_SUMMARY_MODEL`
- `GPT5_ROLEPLAY_LLM_FACTS_MODEL`
- `GPT5_ROLEPLAY_LLM_ADDRESS_MODEL`
- `GPT5_ROLEPLAY_LLM_EMBEDDING_BASE_URL`
- `GPT5_ROLEPLAY_LLM_EMBEDDING_MODEL`
- `GPT5_ROLEPLAY_LLM_EMBEDDING_API_KEY`
- `GPT5_ROLEPLAY_LLM_EMBEDDING_DIMENSIONS`
- `GPT5_ROLEPLAY_LLM_MAX_TOKENS`
- `GPT5_ROLEPLAY_LLM_TEMPERATURE`
- `GPT5_ROLEPLAY_LLM_TIMEOUT`
- `GPT5_ROLEPLAY_LLM_REASONING`
- `GPT5_ROLEPLAY_LLM_PROVIDER_ORDER` (comma-separated, e.g. `siliconflow`)
- `GPT5_ROLEPLAY_LLM_PROVIDER_ALLOW_FALLBACKS`
- `NEO4J_GENAI_API_KEY`
- `NEO4J_GENAI_PROVIDER`
- `NEO4J_GENAI_ONLY`

## `knowledge_storage`

Experience retrieval knobs:

- `context_window_size`: used as a fallback for `memory.max_recent_messages`
- `experience_similar_limit`: top-k retrieved experiences
- `experience_score_min`: minimum similarity score gate
- `experience_score_delta`: max allowed drop from the top score
- `near_duplicate_collapse_enabled`: collapse near-duplicate `related_experiences` into one representative
- `near_duplicate_similarity`: similarity threshold (0.0-1.0) used for near-duplicate clustering
- `routine_summary_enabled`: include compact routine-pattern summaries in `related_experiences`
- `routine_summary_limit`: max routine summary lines added to context
- `routine_summary_min_count`: minimum matching experiences required before a routine summary is emitted

## `facts`

Durable-fact extraction behavior:

- `enabled`: turn fact extraction on/off
- `mode`: `periodic` (recommended) or `per_message`
- `interval_seconds`: minimum time between periodic sweeps
- `evidence_max_messages`: max messages sent to the facts extractor per sweep chunk
- `in_bundle`: include facts extraction in primary chat bundle path
- `min_pending_messages`: flush queued messages once this many are waiting
- `max_pending_age_seconds`: flush queued messages when oldest queued age reaches this threshold
- `flush_on_overflow`: when `true`, memory overflow can force immediate fact flush

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

## `facts_deduplication`

Periodic post-processing loop that collapses redundant facts in Neo4j.

- `enabled`: enables/disables the background deduplication loop
- `interval_hours`: time between deduplication passes

Notes:

- This loop runs only when both Neo4j storage and `OpenRouterLLMClient` are active.

## `experience_deduplication`

Periodic post-processing loop that merges near-identical, near-in-time `Experience` nodes in Neo4j.

- `enabled`: enables/disables the background deduplication loop
- `dry_run`: when `true`, logs merge plans without deleting/updating nodes
- `interval_hours`: time between deduplication passes
- `similarity_threshold`: minimum vector similarity required to consider a merge
- `max_time_gap_hours`: maximum non-overlapping time gap between experiences to allow merge
- `neighbor_k`: nearest-neighbor candidates checked per experience node

Notes:

- This loop runs only when Neo4j storage is active.
- Merges keep one canonical node, widen `timestamp_start`/`timestamp_end`, set `timestamp` to merged end, and delete duplicates.

## `database`

Neo4j connection:

- `uri`: e.g., `bolt://localhost:7687`
- `user`: Neo4j username
- `password`: Neo4j password
- `name`: Neo4j database name (hyphenated names are supported; creation is identifier-quoted)

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

Environment override:

- `GPT5_ROLEPLAY_WANDB_ENABLED`

## Additional Environment Overrides

- `GPT5_ROLEPLAY_HOST`, `GPT5_ROLEPLAY_PORT`
- `GPT5_ROLEPLAY_PERSONA`, `GPT5_ROLEPLAY_USER_ID`
- `GPT5_ROLEPLAY_QUEUE_MAX`, `GPT5_ROLEPLAY_QUEUE_DROP`
- `GPT5_ROLEPLAY_BATCH_WINDOW_MS`, `GPT5_ROLEPLAY_BATCH_MAX`
- `GPT5_ROLEPLAY_MAX_PARTICIPANTS`
- `GPT5_ROLEPLAY_SUMMARY` (`simple` or `llm`)
