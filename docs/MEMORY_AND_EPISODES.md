# Memory Model and Episodic Experiences

This rewrite separates three concerns:

1) Short-term context
2) Compressed summary
3) Long-term episodic experiences

Core code:

- Memory types: `src/gpt5_roleplay_system/memory.py`
- Episode boundaries: `src/gpt5_roleplay_system/pipeline.py:_maybe_finalize_episode`

## Short-Term Context

### Rolling buffer (episodic input)

- Class: `RollingBuffer`
- Stored in-memory per session
- Used to detect episode boundaries and build episode summaries

Writes:

- User messages: `MessagePipeline.process_chat_batch(...)`
- AI responses: when a bundle returns text

### Conversation memory (prompt context)

- Class: `ConversationMemory`
- Provides:
  - `recent()` messages
  - `summary()` compressed history

Compression:

- The controller sets `defer_compression=True`
- The pipeline drains overflow and compresses it after the LLM call

## Long-Term Experiences (Episodic)

Experiences are no longer created per message or per overflow. They are created
on episodic boundaries.

Trigger conditions (episode boundary):

- `rolling_buffer` size reaches `episode_summary.max_messages`
- inactivity exceeds `episode_summary.inactivity_seconds`
- time since last episode exceeds `episode_summary.forced_interval_seconds`

Guardrails:

- Requires at least `episode_summary.min_messages`
- Requires that at least `min_messages` new entries have appeared since the
  last episode watermark

When an episode triggers:

1) The pipeline summarizes the rolling buffer snapshot
   - prefers LLM summary: `LLMClient.summarize(...)`
   - falls back to `SimpleMemoryCompressor`
2) It stores one experience in `ExperienceStore`
3) If vector search is enabled, it schedules embedding/indexing
4) It trims the rolling buffer down to `episode_summary.overlap_messages`

## Experience Retrieval

Experience retrieval combines:

- lexical similarity: `TokenSimilaritySearch`
- semantic vector search (if enabled)

Vector results are gated by:

- `knowledge_storage.experience_score_min`
- `knowledge_storage.experience_score_delta`

Only gated results are merged into `related_experiences` in the prompt context.

## Why Episodic > Overflow

Overflow boundaries are token-budget artifacts, not semantic boundaries.
Episodes better match "memorable experiences across multiple messages."

