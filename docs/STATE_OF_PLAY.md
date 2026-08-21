# State of Play — Orientation

**Last updated:** 2026-08-21

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

**The trap:** `_filter_overflow_against_summary()` discards overflow already covered by
the summary. Advancing that watermark *without* genuinely summarizing destroys history
permanently. Only advance the watermark when the summary text actually changed.

**Timestamps are not identities.** The protocol accepts integer unix seconds and a
batch shares one timestamp, so `ts <= watermark` cannot mean "already covered". Every
coverage check (summary, episodes, facts) is keyed by `MemoryItem.item_id` /
`_fact_message_key`, with the timestamp watermark only as the coarse fallback. An item
sitting *exactly* on a watermark is ambiguous and must be proven covered before it is
dropped.

**Episodes** are separate from all of the above. The `rolling_buffer` accumulates
messages; on an episode boundary it is summarized into one `Experience` and embedded
into Neo4j for semantic recall. `max_rolling_buffer` is a *retention target*, not a
hard cap — the buffer is the only copy of un-summarized episodic input, so it is
drained only by `trim_after_episode()` after real summarization. Appending never
discards.

**The general rule (the one that keeps being violated):** overflow, the rolling buffer,
and the pending fact-evidence queue are each the *only copy* of un-processed history.
None of them may be capped, truncated, or dropped on a failure path. They shrink only
after their contents have actually been summarized or stored. Every bug in §5b was a
violation of this one sentence.


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
| — | Persistence is keyed by **persona only** (`session_state_<persona>.json`); `user_id` no longer participates. Two user_ids under one persona share a file. This is deliberate (`tests/test_controller.py`) but contradicts `docs/STATE_PERSISTENCE.md`. Decide which is authoritative. | `session_state.py:29-36` |
| — | `_slug()` is non-injective: `Alice/Bob` and `Alice?Bob` both become `Alice_Bob`, so distinct personas can collide on one state file. | `session_state.py:13-15` |
| — | A stale `_apply_overflow_summary` completion can overwrite a newer summary. Requires concurrent direct pipeline calls; the server queue serializes chat, so it is not reachable in normal operation. | `pipeline.py` `_apply_overflow_summary` |
| — | `drain_background_tasks()` snapshots the task set once, so it can return while a task spawned *during* the drain is still running. | `background.py:103-104` |

---

## 5b. Fixed in the data-loss audit (2026-08-21)

All twelve were confirmed with an executed repro against live code, then proven
load-bearing by re-introducing the original bug and watching the test go red.
Regression suite: `tests/test_data_loss_regressions.py`. Full suite: **218 passing.**

**The unifying invariant:** *a queue that is the only copy of un-processed history
(overflow, rolling buffer, pending fact evidence) must never be truncated, and must
never be dropped on a failure path. It may only shrink after the content has actually
been summarized or stored.* Every bug below was a violation of it.

| # | Bug | Fix |
|---|---|---|
| V1 | **Critical.** Overflow was drained at `pipeline.py:592`, then any later raise (`build_context`, address check, `generate_bundle`) destroyed it — only `_apply_overflow_summary` requeued. A provider 500 silently ate history. | The whole turn is wrapped in `try/finally` with an `overflow_settled` flag; unowned overflow is requeued on every abnormal exit. |
| V2 | **Critical.** `set_user_id()` → `_reset_state_store()` reloaded disk state *over* live memory, and the stale state then overwrote the good file on flush. (open bug G) | `set_user_id` now only rotates where saves land (`_rebind_state_store(load=False)`), matching `_sync_user_id_from_pipeline`. |
| V3 | Switching to a persona with no state file left the previous persona's history live, then wrote it into the new persona's file. Privacy + correctness. | `set_persona` saves under the *outgoing* persona's key first, then loads the new one, clearing when absent. |
| V4/V12 | `RollingBuffer._trim()` discarded the oldest un-summarized messages on every append. With `max_rolling_buffer == episode.max_messages == 30`, the max-messages trigger lost a message *every time it fired*; batches lost more. | `_trim()` no longer discards — it only warns. Draining happens exclusively in `trim_after_episode()`, after real summarization. |
| V5 | `ConversationMemory.restore()` sliced `recent` to `max_recent`, destroying the remainder with no summary compensation. Reachable at unchanged config via legitimate tie overage. (open bug B) | Displaced items move to the end of the overflow backlog instead of being dropped. |
| V6 | Fact worker only requeued on *extraction* errors. Any later raise (tracer, cancellation) destroyed the un-processed tail, and `finally` cleared the handle so the exception was unobservable. | Added `except BaseException` that requeues `remaining` before re-raising. |
| V7 | `_schedule_state_save()` returned silently while a save was in flight, with no dirty flag — newer state never reached disk until a graceful flush. | `_state_save_pending` flag; the in-flight task performs a trailing save. |
| V8 | Episode "new items" used `count - last_episode_size`. Once the buffer sat at retention size, arrivals *replaced* overlap without changing length, so new traffic never registered and episodes stalled permanently. | Counted by message identity against `_episode_covered_ids` (persisted in snapshot). |
| V9 | Fact cursor compared timestamps only (`<=`), so a distinct message sharing the cursor's timestamp was skipped as "already swept". Integer unix seconds make ties normal. | `<` plus an explicit `_facts_swept_keys` check for the ambiguous equal case. |
| V10 | Pending fact evidence was capped at `3 × evidence_max_messages`, **keeping the newest** — the classic backlog-vs-cache inversion. A long outage destroyed the oldest un-swept evidence. | Cap removed; `FACTS_PENDING_WARN_THRESHOLD` warns instead. |
| V11 | `_filter_overflow_against_summary` dropped everything `<= range_end_ts`, so later messages tied to the watermark were treated as summarized and discarded. | Items *exactly* at the watermark are only dropped when `memory.is_summarized(item)` confirms real coverage. |

Supporting change: `MemoryItem` gained a stable `item_id` (persisted through
`session_state.py`). Timestamps cannot identify a message — the protocol accepts
integer unix seconds and batches share one timestamp — so all coverage bookkeeping
(summary, episodes, facts) is keyed by identity, with the timestamp watermark kept
only as the coarse fallback.

Two pre-existing tests asserted the buggy behavior and were rewritten to assert the
requirement instead: `test_rolling_buffer_trims` (asserted the destructive trim) and
`test_pipeline_filters_overflow_already_covered_by_summary_meta` (faked coverage by
writing `_summary_meta` directly; now records coverage through `apply_summary`).

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
