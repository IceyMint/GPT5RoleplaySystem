# State of Play — Orientation

**Last updated:** 2026-08-18

Read this first when returning to the project after time away. It records what is
*verified true* about the running system, what is fixed, what is still broken, and
which older docs will mislead you.

---

## 1. What this system does (one paragraph)

A TCP server that gives an AI roleplay persona memory. Each connection gets its own
session. Incoming chat flows through `MessagePipeline`, which keeps short-term context
(`recent`), a compressed `summary` of older context, and long-term *episodic memories*
stored as vector-embedded `Experience` nodes in Neo4j. On each turn it semantically
searches those memories for ones related to the incoming message and puts them in the
prompt.

---

## 2. The memory model (the part that causes most bugs)

Three layers, in time order. **The critical invariant is that they must tile without
gaps or overlap:**

```
[ summary ] [ overflow ] [ recent ]
   oldest  ->               newest
```

- **`recent`** — last N messages, sent to the model verbatim. Bounded by `max_recent`.
- **`overflow`** — messages that fell out of `recent` but are **not yet summarized**.
  This is the *only copy* of that history. **Never truncate it.** It grows without
  bound during an LLM outage and drains when summarization succeeds.
- **`summary`** — LLM-written prose covering everything older. Carries
  `summary_meta.range_end_ts`, a watermark meaning "summary covers up to here."

**The trap:** `_filter_overflow_against_summary()` discards any overflow message at or
below `range_end_ts`. So advancing that watermark *without* genuinely summarizing
destroys history permanently. Only advance the watermark when the summary text
actually changed.

**Episodes** are separate from all of the above. The `rolling_buffer` accumulates
messages; on an episode boundary it is summarized into one `Experience` and embedded
into Neo4j for semantic recall.

---

## 3. Where the code actually lives (IMPORTANT)

`pipeline.py` (~3000 lines, 135 methods) contains **all live logic**.

`context_builder.py`, `fact_manager.py`, and `episode_manager.py` look authoritative
but are **unused abstractions**. Verified by runtime tracing of a full turn:

| module | functions that executed | notes |
|---|---|---|
| `pipeline.py` | 84 | all real work |
| `context_builder.py` | 5 | ~2 lines each — pure forwarders |
| `fact_manager.py` | 1 | pure forwarder |
| `episode_manager.py` | **0** | never entered |

They delegate back via `if self._owner is not None: return self._owner._x(...)`, and
`pipeline.py` always passes `owner=self`.

> **Unused abstraction ≠ unused feature.** Every feature these modules name is *live*
> — it just runs in `pipeline.py`. `EpisodeManager` is the episodic-memory feature
> (Neo4j semantic recall) and that feature **works**; only the wrapper class is unused.

**Their copies have drifted and are worse than the live ones.** If you ever wire them
up, port these first:

- `fact_manager.store_facts` — missing near-duplicate filtering, missing
  `_mark_mention_index_dirty()`
- `fact_manager._enqueue_fact_messages` — missing self/ignored-message filtering
- `episode_manager` — missing ignored-user filtering (would pollute Neo4j embeddings
  with system users), no compressor fallback
- `context_builder.update_environment` — missing ignored-user filter and
  reappearance-signal tracking; **unreachable** (all callers route to the pipeline)

---

## 4. Stale docs — do not trust

- `docs/REFACTORING_PLAN.md` — **obsolete.** Describes a refactor that was abandoned
  half-done. The result is section 3 above.
- `pipeline.py:574` comment *"Ensure deferred compression doesn't accumulate unbounded
  while silent"* — **stale.** It described `compress_overflow()` as the bounding
  mechanism, but the same commit that added a cap (`a16995d`, 2026-04-02) also made
  `SimpleMemoryCompressor` a **no-op** to prevent summary poisoning. That call has been
  inert since. Verified: `overflow before=48, after=48`.

Lesson: date-check comments against `git log -S` before trusting them as invariants.

---

## 5. Fixed this session (all with regression tests)

| # | Bug | Fix |
|---|---|---|
| L | `summarize_overflow` returned the **old summary** on failure. Truthy → watermark advanced → history destroyed on every provider hiccup. | Return `""` on empty response (`llm.py`) |
| L2 | Both overflow call sites applied any truthy summary. | New `_apply_overflow_summary()` — only advances if text changed; catches exceptions; requeues otherwise; logs `overflow_summary_stalled` |
| C | Overflow cap **destroyed history** (kept newest 200, dropped oldest un-summarized). | Cap removed entirely. `OVERFLOW_WARN_THRESHOLD=200` now only *warns*. Overflow must grow during outages. |
| P | Overflow was **not persisted** — a restart mid-outage dropped the whole backlog. | Added to `snapshot()`/`restore()` **and** `SessionStateStore` (both required — raw `MemoryItem`s are not JSON serializable and `save()` would throw on a background thread) |
| A | Tied timestamps disabled the `recent` bound entirely → unbounded prompt growth. | Tie preservation kept, but capped at `2× max_recent`, then forced deterministic split |
| BG | 7 of 17 background tasks held **no reference** (GC-able mid-flight, per CPython docs) and **all** failures were invisible — a failing Neo4j/embedding/fact write surfaced nowhere. | New `background.py` → `spawn_supervised()`: holds a strong ref, logs failures with a label. Converted all 7 unreferenced sites. |
| BG2 | `_autonomy_loop` had **zero** exception handling — one bad tick killed autonomy for the whole connection while the socket stayed open looking healthy. | try/except + backoff, matching what `_status_loop` already did |
| D | Episode summarization awaits the LLM, then trimmed the **live** buffer by position — messages arriving during the await were discarded unsummarized. | New `RollingBuffer.trim_after_episode(snapshot, keep_last)` trims by **identity**, so only actually-summarized items are removed. Watermark now records what was really retained. |
| F | Fact sweep removed evidence from the pending queue **before** the LLM call; a provider error destroyed it permanently. | Chunk is dropped only **after** success; failures requeue via `_requeue_fact_evidence()` (chronological order preserved) + `FACTS_RETRY_COOLDOWN_SECONDS=60` so a broken provider doesn't retry per message |

**Verification approach:** `scripts/verify_overflow_regressions.sh`,
`scripts/verify_overflow_persistence.sh`,
`scripts/verify_background_supervision.sh`, and
`scripts/verify_episode_facts_recovery.sh` re-introduce each original bug and confirm a
test catches it. Run these — a passing suite alone does not prove the tests are
load-bearing.

Full suite: **206 passing.**

---

## 5a. Background tasks (read before touching concurrency)

Backgrounding Neo4j writes, embeddings, fact storage and state saves is **deliberate
and correct** — it keeps ~150 ms/turn off the response path. Measured:

```
work on the response path : 150.2 ms
backgrounded              :   0.0 ms
spawn + supervision        :   0.01 ms   <- supervision is free
```

**Always use `spawn_supervised(coro, label=...)` from `background.py`** for
fire-and-forget work. Never bare `asyncio.create_task` for work whose handle you
discard: asyncio keeps only a *weak* reference, so the task can be garbage collected
mid-execution, and its exception surfaces nowhere the caller can see.

- `quiet=True` — for tasks that are themselves part of error reporting
  (`error_broadcast`), to avoid a logging feedback loop.
- **10 sites intentionally still use bare `create_task`**: `_facts_task`,
  `_episode_task`, `_state_task`, the two dedup services, and worker/autonomy/status.
  These hold their handles and are awaited (`wait_for_idle`, `flush_state`) or
  cancelled (`_cancel_task`). Leave them.

**Useful signal:** grep logs for `Background task <label> failed`. Labels in use:
`experience_embedding`, `store_facts`, `last_seen_update`, `error_broadcast`.
If `experience_embedding` or `store_facts` appear, bugs D/F below are firing for real.

---

## 6. Still open (not fixed)

| # | Bug | Where |
|---|---|---|
| G | `set_user_id()` reloads disk state **over** live session state | `controller._reset_state_store` |
| B | `restore()` truncates `recent` to `max_recent` with no summary compensation | `memory.restore` |
| J/H | No startup validation that `episode.max_messages < max_rolling_buffer` (both are 30 — the trigger is borderline unreachable) | config |

---

## 7. Operational facts (verified against the live system)

- **Neo4j** 2026.07.0, `bolt://localhost:7687`, db `gpt5-roleplay`. GenAI plugin
  installed. Vector index `experience_embedding_idx` ONLINE, 3072-dim, COSINE.
- **539 Experience nodes**, all embedded. 572 Person, 6 Persona.
- **Semantic search silently died** when the OpenAI account ran out of credits
  (`429 insufficient_quota`). `neo4j_genai_only: true` forbids the OpenRouter fallback,
  so `_build_experience_index()` returned `None` and the whole feature switched off.
  Credits restored 2026-08-18 → verified **LIVE** again.
  - The probe runs **once at startup** — you must restart the server after fixing billing.
  - `_check_genai_available` does `except Exception: return False`, so a *billing*
    failure is reported as *"genai.vector.encode is unavailable"*. Misleading. This is
    why it went unnoticed from April to August.

### Two tuning issues worth attention

1. **`experience_score_min: 0.78` is probably too high.** Real matches on live data
   scored **0.70 / 0.699** — below the gate. `text-embedding-3-large` cosine scores for
   related text typically sit in 0.6–0.8. Query *"bedtime routine"* returned **0 hits**
   despite 296 relevant episodes. Try 0.60–0.65.

2. **Persona ID case split (stale, not active).** `ReneAllardyce` (97 experiences,
   through 2026-04-06) vs `reneallardyce` (4, written only on 2026-03-06). Writes
   converged after the normalization work; Person nodes have **0** case collisions.
   Only 4 orphaned rows. Note `set_persona()` still stores `extract_username()` output
   verbatim (case-preserving), so divergence remains *possible* — normalize at the
   storage boundary if you want it airtight.

---

## 8. How to work on this safely

The codebase is ~100% AI-written and has accumulated duplication. It does **not** need
a rewrite — the test suite (6.5k lines vs 12.9k source) encodes real behavioral
contracts and is the most valuable asset here. A rewrite would discard the oracle.

Method that works (learned the hard way this session):

1. **Trace the whole subsystem before editing** — callers, callees, persistence,
   lifecycle, error paths.
2. **Extract invariants from evidence** — comments, tests, `git log -S`, docs.
   Date-check them; stale comments caused a wrong fix this session.
3. **Separate evidence from inference.** Say which is which.
4. **Check the fix against the whole invariant**, not just the symptom.
   (A correct local `_overflow` fix still broke the restart guarantee.)
5. **Write tests from the requirement, not the implementation.** Asserting that code
   does what you just wrote is confirmation bias, not verification. Prove the test
   fails against the old behavior.
6. Only ask a human when the repo genuinely lacks or conflicts on evidence.
