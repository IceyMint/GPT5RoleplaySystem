# CODE MAP — GPT5RoleplaySystem

**Status: SOURCE OF TRUTH for the 2026-11 cleanup. No code changes precede this map.**
Mapped at commit `dd2a9e5` (main, 2026-08-22) on branch `cleanup/code-map-2026-11-06`.
Method: every source file below was read firsthand, line by line, by Muse on 2026-11-06.
Reference counts are machine-extracted (Python AST inventory + name search across
`src/`, `tests/`, `scripts/`); the raw machine index is `docs/code_map_index.json`.
Classifications were assigned from the firsthand read, not from the counts alone.

## How to read this map

| Tag | Meaning |
|---|---|
| **LIVE** | Runs on a production path (server → controller → pipeline → llm/store). |
| **ENTRY** | Process/protocol entry point. Zero in-repo callers is normal. |
| **DELEGATE** | Exists only to forward to the real implementation elsewhere. |
| **FALLBACK-ONLY** | Runs only when constructed without an owner / on a degraded path. |
| **DEAD** | No caller anywhere in src, tests, or scripts (name search = 0, or shadowed). |
| **TEST-ONLY** | Referenced only from tests (or test doubles living in production files). |
| **DOUBLE** | A test double (fake LLM/store) — legitimate, but know it is not production. |

Counts in the appendix tables are name occurrences outside the definition line.
They undercount dynamic dispatch (`getattr`, protocol calls) and overcount
same-named methods on other classes; every DEAD claim in §5 was additionally
verified by a targeted grep for attribute/import usage.

---

## 1. What actually runs

```
start-gpt5-roleplay.sh
  → python -m gpt5_roleplay_system.server  (server.py: main → run_server)
      builders: _build_knowledge_store / _build_llm_client / _build_tracer / _build_experience_index
      GPT5RoleplayServer.start()
        ├─ FactsDeduplicationService.run_forever()      (background_tasks.py, if enabled + Neo4j + OpenRouter)
        ├─ ExperienceDeduplicationService.run_forever() (background_tasks.py, if enabled + Neo4j)
        └─ per TCP connection (_handle_client):
             SessionController (controller.py) ── owns ──► MessagePipeline (pipeline.py)
             ClientSession.process_queue (session.py) — batching, protocol dispatch
             _autonomy_loop / _status_loop (server.py, if autonomy.enabled)
```

**The one-paragraph truth:** `MessagePipeline` (pipeline.py, 3,200 lines, 138
functions) contains effectively all live behavior: chat processing, context
building, fact extraction/storage, episode summarization, autonomy, posture,
state snapshot/restore. `OpenRouterLLMClient` (llm.py) contains all live LLM
behavior, using `PromptManager` (llm_prompts.py) for every prompt and payload
format. Several other modules *look* like the home of these behaviors and are
not — see §4. Storage is `Neo4jKnowledgeStore` (neo4j_store.py) in production,
`InMemoryKnowledgeStore` when Neo4j is unconfigured/unavailable (announced by
log). Experiences persist through `Neo4jExperienceVectorIndex`
(neo4j_experience_vector.py); the in-memory `ExperienceVectorIndex`
(experience_vector.py) serves the non-Neo4j path.

## 2. Module inventory (37 source files, 13,474 lines)

| File | Lines | Status | What it really is |
|---|---|---|---|
| `pipeline.py` | 3,200 | LIVE | The system. See §4.1. |
| `llm.py` | 2,049 | LIVE | All LLM clients + structured schemas + live copies of parser/mapper/transport logic. See §4.2. |
| `llm_prompts.py` | 999 | LIVE | `PromptManager`: every system prompt + prompt payload formatter. Genuinely used by llm.py (via `_prompt_manager_instance()`) and twice by pipeline.py directly. |
| `context_builder.py` | 754 | DELEGATE + FALLBACK-ONLY | Satellite of pipeline. §4.1. |
| `neo4j_store.py` | 663 | LIVE | KnowledgeStore ABC, InMemory + Neo4j implementations, dedupe/runtime-marker queries. |
| `payload_contract.py` | 582 | LIVE (warnings only) | Payload normalization/validation. Pipeline calls `normalize_and_validate_payload` and **discards the repaired payload** — only the warnings are logged (§7). |
| `server.py` | 575 | LIVE + DEAD pockets | Entry point, builders, loops; 4 dead wrapper functions (§5). |
| `config_loader.py` | 504 | LIVE | The real config parser (`ConfigLoader`). |
| `memory.py` | 484 | LIVE | MemoryItem, RollingBuffer, ConversationMemory, ExperienceStore, similarity searches. Clean module. |
| `fact_manager.py` | 465 | DELEGATE + FALLBACK-ONLY | Satellite of pipeline. §4.1. |
| `background_tasks.py` | 248 | LIVE | The two dedupe services (the real loops; server.py's same-named wrappers are dead). |
| `config.py` | 217 | LIVE (dataclasses) + DEAD pockets | Config dataclasses + `load_config` delegate + 5 dead helper functions (§5). |
| `deduplication.py` | 210 | LIVE | `parse_deduped_facts_response`, `ExperienceDedupePlanner`. |
| `llm_response_mapper.py` | 197 | DEAD (prod) / TEST-ONLY | Decoy copy of llm.py's module-level mapping functions. §4.2. |
| `episode_manager.py` | 193 | DORMANT | Constructed by pipeline, never called. §4.1. |
| `llm_structured_parser.py` | 186 | DEAD | Decoy copy of llm.py's `_request_structured`. Instantiated, never used. §4.2. |
| `experience_vector.py` | 186 | LIVE | Embedding clients + in-memory vector index (non-Neo4j path). |
| `neo4j_experience_vector.py` | 266 | LIVE | Neo4j vector index (GenAI encode + external embedder paths). §7 notes silent spots. |
| `autonomy_manager.py` | 294 | DORMANT | Constructed by pipeline, handed to FactManager, never invoked. Holds duplicate mood/status state. §4.1. |
| `controller.py` | 293 | LIVE | SessionController: honest forwarder + state persistence binding. No facades. |
| `llm_transport.py` | 130 | DEAD | Decoy copy of llm.py's request/extra-body logic. Instantiated, never used. §4.2. |
| `session_state.py` | 131 | LIVE | SessionStateStore: atomic JSON state files, persona-keyed, legacy filename migration. |
| `session.py` | 125 | LIVE | ClientSession: protocol dispatch + chat batching. |
| `background.py` | 106 | LIVE | `spawn_supervised` — supervised fire-and-forget tasks. Model citizen. |
| `name_utils.py` | 91 | LIVE | Display/username splitting + name matching. |
| `protocol.py` | 79 | LIVE | NDJSON decode/encode, chat_response builder. |
| `models.py` | 70 | LIVE | Core dataclasses + CommandType. |
| `observability.py` | 59 | LIVE | Tracer ABC, NoOpTracer, WandbTracer. |
| `config_validation.py` | 48 | **DEAD (whole file)** | Pydantic raw-config models. Nothing imports them (§5). |
| `time_utils.py` | 22 | LIVE | Pacific-time helpers (note: `get_pacific_timestamp` returns Pacific-digit-shifted epoch — read its docstring before "fixing"). |
| `pipeline_state.py` | 20 | LIVE | `PipelineRuntimeState` — the small shared state the satellites actually share. |
| `__init__.py` | 15 | LIVE | Package marker; `__all__` lists only 11 of 37 modules. |
| `cypher_utils.py` | 8 | LIVE | `quote_cypher_identifier`. |
| `persona_policy.py` | 5 | LIVE | `is_chat_and_state_restricted_persona` (the "DefaultPersona" gate). |

## 3. Entry points and who calls whom (production)

- `server.main()` → `run_server()` → builders → `GPT5RoleplayServer.start()`.
- Per connection: server builds one `SessionController` + one `ClientSession`.
- `ClientSession.handle_message` dispatches protocol types: `process_chat`,
  `environment_update`, `set_user_id`, `set_llm_chat_enabled`, `autonomous_tick`.
- `SessionController` forwards everything to `MessagePipeline` and owns state
  save/load binding (`SessionStateStore`), including persona-switch save-then-rebind.
- `MessagePipeline.process_chat` / `process_chat_batch` / `generate_autonomous_actions`
  are the three behavior entry points. Everything else hangs off them.
- Background: `spawn_supervised` (background.py) for fact storage, experience
  embedding, state saves; dedupe services run as server-level asyncio tasks.

---

## 4. The satellite systems (read this before editing anything)

### 4.1 pipeline.py satellites

`MessagePipeline.__init__` constructs four helper objects. Three of them do not
own their behavior:

**ContextBuilder (context_builder.py)** — constructed with `owner=pipeline`.
Its hot-path methods (`build_chat_context`, `merge_participants`,
`build_context`, `collapse_incoming_batch`, `fact_sweep_*`, `store_facts`, …)
each begin with an owner check and delegate straight back into pipeline
privates (`_build_context`, `_merge_participants`, …). The fallback bodies
only execute when an owner-less instance is constructed — which **never
happens anywhere**: no production code, no test, and no script constructs
`ContextBuilder`, `FactManager`, `EpisodeManager`, or `AutonomyManager`
directly (verified by grep across src/tests/scripts, 2026-11-06).
Fallback drift (verified firsthand): fallback `build_context` is lexical-only
(no people resolution, no semantic gating, no routine summaries); its
near-duplicate scorer uses 2 signals vs the pipeline's 3; its fact-key
normalizer skips NFKC.


**FactManager (fact_manager.py)** — constructed with `owner=pipeline` plus a
facts config and the (dormant) AutonomyManager. Six methods delegate to the
pipeline (`enqueue_fact_messages`, `recover_pending_from_memory`,
`run_facts_worker`, `maybe_run_periodic_fact_sweep`, `store_facts`,
`sweep_evidence_messages` — sweep delegates *indirectly* via the pipeline's
extraction callable). Fallback drift (verified firsthand): the fallback
`_enqueue_fact_messages` **caps the pending queue and silently drops the
oldest evidence**; the fallback worker slices messages off before extracting
and never requeues on failure. The pipeline's own versions do the opposite,
with comments explaining why. `FactManager.recover_pending_from_memory`
specifically: zero callers even as a delegate — the pipeline calls its own
`_recover_pending_facts_from_memory` in `restore_state`.

**EpisodeManager (episode_manager.py)** — constructed **without** an owner and
never called by anything (grep-verified: `schedule_check` / `maybe_finalize`
have no call sites). Episodic memory is alive — inside pipeline
(`_schedule_episode_check`, `_maybe_finalize_episode`, `_summarize_episode`,
called from `process_chat_batch` and `generate_autonomous_actions`). The
dormant manager still detects "new messages" by length arithmetic
(`count - last_episode_size`), the method pipeline comments say "silently
under-counts… so episodes stalled forever"; the pipeline uses identity
tracking (`EPISODE_COVERED_ID_LIMIT = 5000`).

**AutonomyManager (autonomy_manager.py)** — constructed, handed to FactManager,
never invoked. Autonomy is alive in pipeline + server loops. The manager holds
its own mood/status state (`_mood`, `_agent_status`, delay hint) that diverges
from the pipeline's from the moment of construction. 9 of its methods have zero
references anywhere (§5).

**Why they look modular:** the shared state never moved. `PipelineRuntimeState`
(pipeline_state.py, 20 lines) holds only persona/user_id/environment/posture/
display names; the fact queue, episode coverage, mood, and cursors all live in
`MessagePipeline`. A module that doesn't own its state can't own its behavior —
hence the `owner=` escape hatch.

### 4.2 llm.py satellites

`OpenRouterLLMClient.__init__` instantiates four components. Only one is real:

| Component | Status | Evidence |
|---|---|---|
| `PromptManager` (llm_prompts.py) | **LIVE** | Every prompt/format method in llm.py routes through `_prompt_manager_instance()`. |
| `OpenRouterTransport` (llm_transport.py) | **DEAD** | Constructed (llm.py:717); zero method calls anywhere. llm.py uses its own `_request_text_with_model`, `_apply_extra_body`, `_provider_payload`. |
| `StructuredParser` (llm_structured_parser.py) | **DEAD** | Constructed (llm.py:716); zero calls. llm.py uses its own `_request_structured`, `_structured_parse_key`, `_clean_json`. |
| `ResponseMapper` (llm_response_mapper.py) | **DEAD in prod, TEST-ONLY** | Constructed (llm.py:725); zero production calls — llm.py uses module-level `_bundle_from_structured` etc. Its only caller is `tests/test_nonfinite_autonomy_delay.py`. |

Drift example: `ResponseMapper._optional_positive_float` rejects non-finite
values (`math.isfinite`); llm.py's live module-level `_optional_positive_float`
**does not** — an infinite `next_delay_seconds` from the live path is only
caught later by the pipeline's own guard.

### 4.3 Thinner delegates (harmless, but know them)

- `config.load_config()` → one-line delegate to `ConfigLoader().load()`.
- `server._parse_deduped_facts_response` / `_facts_deduplication_loop` /
  `_experience_deduplication_loop` → dead wrappers (server.start builds the
  services directly); `_build_experience_dedupe_plans` → live-code delegate
  kept alive **only by two tests**.
- `payload_contract.normalize_and_validate_payload` → pipeline uses only its
  warnings; the repaired payload is discarded (see §7).

---

## 5. Dead and dormant inventory (evidence-backed)

### 5.1 Whole files / classes with no production path
| Item | Evidence |
|---|---|
| `config_validation.py` (entire file) | `RawServerConfigModel` referenced nowhere outside its own file (grep, 2026-11-06). |
| `llm_transport.OpenRouterTransport` | Imported + constructed in llm.py; no method call in src/tests/scripts. |
| `llm_structured_parser.StructuredParser` | Same pattern. |
| `llm_response_mapper.ResponseMapper` | Production: same pattern. One test uses `bundle_from_structured`. |
| `episode_manager.EpisodeManager` | Constructed once (pipeline), no method ever called. |
| `autonomy_manager.AutonomyManager` | Constructed once (pipeline), no method ever called. |
| `memory.SimpleSimilaritySearch` | No references outside memory.py. (TokenSimilaritySearch is LIVE via pipeline.) |
| `llm.EchoLLMClient` | DOUBLE: used by tests only; lives in the production module. |

### 5.2 Zero-reference functions (name appears nowhere else in src/tests/scripts)
Verified by the machine index; each is a deletion candidate pending the
ChatGPT cross-check:
- autonomy_manager: `mark_inbound_activity`, `mark_response_activity`,
  `update_from_bundle`, `apply_scheduler_override`, `filter_autonomous_actions`,
  `chat_texts_from_actions`, `consume_delay_hint_seconds`,
  `set_activity_from_timestamp`, `normalize_autonomy_decision` (9)
- `ContextBuilder.environment_payload` (context_builder.py:344)
- `EpisodeManager.schedule_check` (episode_manager.py:68)
- `FactManager.recover_pending_from_memory` (fact_manager.py:108)
- `OpenRouterLLMClient._state_system_prompt` (llm.py:1136)
- `ResponseMapper.state_update_from_structured` (llm_response_mapper.py:54)
- `StructuredParser.request_structured` (llm_structured_parser.py:15)
- `OpenRouterTransport.request_text` (llm_transport.py:31)
- `MessagePipeline._state_update_payload` (pipeline.py:2682)
- `MessagePipeline._looks_like_uuid` (pipeline.py:2867) — duplicate of
  `payload_contract.looks_like_uuid`, which is the live one.
- server: `_parse_deduped_facts_response`, `_facts_deduplication_loop`,
  `_experience_deduplication_loop`

### 5.3 Dead by shadowing / duplication (has references, still dead)
- `RuleBasedLLMClient.generate_autonomous_bundle` is defined **twice**
  (llm.py:588 and :595); the first definition is unreachable.
- config.py module-level helpers `_resolve_default_config_path`, `_load_yaml`,
  `_env_bool`, `_parse_optional_bool`, `_pick_model`: nothing imports them;
  `ConfigLoader` carries its own copies. (config.py's `load_config` is live.)
- `OpenRouterLLMClient._request_text` (llm.py:1441): defined, never called;
  all callers use `_request_text_with_model`.
- llm.py's private copies of PromptManager helpers (`_serialize_payload`,
  `_canonicalize_for_prompt`, `_stable_participants_payload`,
  `_entity_prompt_payload`, `_stable_entity_list`, `_chat_payload`,
  `_participant_payload`, `_filter_facts_entities`,
  `_minimal_people_fact_profile`, `_profile_mentioned_in_text`,
  `_name_mentioned_in_text`, `_normalize_name`, `_name_in_text`): the live
  prompt paths delegate to PromptManager's copies; per-function liveness is in
  the appendix table (refs column) — several are kept alive only by tests.

### 5.4 Test-only production functions
- `background.drain_background_tasks`, `background.pending_background_tasks`
  (diagnostics/tests by design — documented as such).
- `OpenRouterLLMClient._system_prompt`, `._autonomous_system_prompt` (tests only).
- `ResponseMapper.bundle_from_structured` (one test).
- `server._build_experience_dedupe_plans` (two tests).

---

## 6. Duplication register (same logic, multiple homes)

| Logic | Homes |
|---|---|
| Structured→bundle mapping | llm.py module functions (LIVE) + ResponseMapper (dead) |
| Structured request + JSON repair | llm.py `_request_structured`/`_clean_json` (LIVE) + StructuredParser (dead) |
| HTTP request + provider payload | llm.py `_request_text_with_model`/`_provider_payload`/`_apply_extra_body` (LIVE) + OpenRouterTransport (dead) |
| Coerce message content to text | llm.py `_coerce_text_content`, StructuredParser._coerce_text_content, OpenRouterTransport._coerce_text_content (3 copies) |
| Context building / fact sweeps | pipeline.py (LIVE) + ContextBuilder/FactManager fallbacks (fallback-only, drifted) |
| Episode finalization | pipeline.py (LIVE) + EpisodeManager (dormant, stale method) |
| Autonomy state/decisions | pipeline.py (LIVE) + AutonomyManager (dormant) |
| Prompt payload helpers (serialize, canonicalize, participant/chat payloads, facts filtering, name matching) | llm_prompts.PromptManager (LIVE) + llm.py private copies (mixed liveness) |
| Config parsing helpers | ConfigLoader methods (LIVE) + config.py module functions (dead) |
| `ExperienceIndexProtocol` | Defined 3×: pipeline.py, context_builder.py, episode_manager.py |
| `looks_like_uuid` | payload_contract (LIVE) + pipeline dead copy + `_UUID_LIKE_RE` re-implemented in llm.py and llm_prompts.py |
| Fact-key normalization (NFKC + casefold + alnum) | pipeline `_normalize_fact_key`, neo4j_store `_normalize_fact_key`, context_builder fallback variant (no NFKC) |
| `safe_float` | deduplication.safe_float, payload_contract._safe_float, llm_prompts._try_float, llm._to_float |
| Deduped-facts response parsing | deduplication.parse_deduped_facts_response (LIVE) + server dead wrapper |
| `_normalize_name` | name_utils.normalize_display_name (different rules!) + llm.py + llm_prompts.py copies |

## 7. Silent-degradation register (per the no-silent-fallbacks principle)

Live-path spots where degradation is **not** announced (contrast: the pipeline's
own overflow/fact paths requeue + warn, server builders log their fallbacks,
and OpenRouterLLMClient records quality-fallback events with warnings):

1. `Neo4jExperienceVectorIndex._search_with_genai` — on exception, sets
   `_genai_available = False` and returns `[]` with **no log**. (Startup probe
   failure is likewise silent in `_check_genai_available`; server logs only in
   strict `neo4j_genai_only` mode.)
2. `Neo4jExperienceVectorIndex._add_with_genai` — runtime GenAI failure flips
   to the external embedder silently.
3. `Neo4jExperienceVectorIndex._ensure_vector_index` — swallows all exceptions
   (`pass`, comment says version compat).
4. `payload_contract` repairs — pipeline computes a repaired payload and
   discards it; only warnings are logged. The LLM receives the unrepaired
   payload.
5. `SimpleMemoryCompressor.compress` — deliberate no-op (documented in code);
   combined with `summary_strategy="simple"` the summary simply never advances.
   Announced in docs, silent at runtime.
6. `RuleBasedLLMClient.generate_response` returns empty by design (comment:
   "Silent fallback: avoid immersion-breaking canned acknowledgements") — this
   one is an intentional product choice, recorded here so it is a decision,
   not an accident.

## 8. Test map (20 files, 7,620 lines, 218 tests per STATE_OF_PLAY)

- Tests construct **real production objects** (MessagePipeline, SessionController,
  ConversationMemory, OpenRouterLLMClient) with test doubles for the LLM
  (`EchoLLMClient`, plus 19 hand-rolled fake LLM/store classes in
  test_pipeline.py alone) and `InMemoryKnowledgeStore`.
- **No test references ContextBuilder, FactManager, EpisodeManager, or
  AutonomyManager at all** (grep-verified). The facade fallbacks are untested
  *and* uncalled.
- Tests that certify non-production paths (cleanup must re-point or remove):
  - `test_nonfinite_autonomy_delay.py` → `ResponseMapper().bundle_from_structured`
    (the dead mapper; the live path is llm.py's `_bundle_from_structured`).
  - `test_server_experience_index.py` (2 tests) → `server._build_experience_dedupe_plans`
    wrapper instead of `ExperienceDedupePlanner.build_plans` directly.
  - `test_llm.py` prompt tests → `OpenRouterLLMClient._system_prompt` /
    `_autonomous_system_prompt` (test-only wrappers over PromptManager).
- Data-loss regression suites (`test_data_loss_regressions.py`,
  `test_episode_and_facts_recovery.py`, `test_overflow_*`,
  `test_recent_tie_bounding.py`) test the **live** pipeline/memory paths and
  encode the V1–V12 fixes — these are the crown jewels; cleanup must keep them
  green verbatim.

## 9. Split candidates (functions ≥ 60 lines)

| Span | Where |
|---|---|
| 160 | `MessagePipeline.__init__` (pipeline.py) |
| 157 | `MessagePipeline._build_context` |
| 145 | `MessagePipeline.process_chat_batch` |
| 122 | `MessagePipeline.restore_state` |
| 116 | `ConfigLoader.load` |
| 106 | `ExperienceDedupePlanner.build_plans` |
| 102 | `FactsDeduplicationService.run_once` |
| 92 | `ConfigLoader.load_llm_config` |
| 87 | `ExperienceDeduplicationService.run_once` |
| 82 | `GPT5RoleplayServer._handle_client` |
| 82 | `OpenRouterLLMClient._request_structured` |
| 77 | `MessagePipeline._bundle_prompt_payload` |
| 77 | `Neo4jKnowledgeStore.upsert_person_facts` |
| 75 | `FactManager.store_facts` (fallback copy; pipeline's is the live one) |
| 74 | `normalize_participants`, `_repair_incoming_batch_and_incoming` (payload_contract) |
| 71 | `SessionController.__init__` |
| 69 | `MessagePipeline._facts_worker`, `PromptManager.format_address_check`, `OpenRouterLLMClient.__init__` |
| 67 | `Neo4jKnowledgeStore.update_last_seen` |

## 10. Naming hazards (same name, different thing / name points at the wrong home)

- `build_context`: ContextBuilder's fallback vs pipeline's `_build_context` (live).
- `environment_payload` (ContextBuilder, dead) vs `_environment_payload` (pipeline, live).
- `recover_pending_from_memory` (FactManager, dead) vs `_recover_pending_facts_from_memory` (pipeline, live).
- `store_facts`: three implementations (pipeline live, FactManager delegate+fallback).
- `_state_system_prompt` (llm, dead) vs `state_system_prompt_for_context` (llm→PromptManager, live).
- `load_config`: config.py delegate vs ConfigLoader.load (real).
- `_facts_deduplication_loop` (server, dead) vs `FactsDeduplicationService.run_forever` (live).
- `normalize_display_name` (name_utils) vs `_normalize_name` (llm/llm_prompts) — different normalization rules under near-identical names.
- `ExperienceStore` (memory.py, in-memory list) vs Neo4j Experience nodes — same word, different stores; persistence happens via the vector index, not the store class.

---

## Appendix — full function index (machine-extracted, hand-classified above)

Refs = name occurrences outside the definition, split src/tests/scripts.

#### `src/gpt5_roleplay_system/__init__.py` — 15 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|

#### `src/gpt5_roleplay_system/autonomy_manager.py` — 294 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `AutonomyManager.__init__` | 13 | 13 | 26/112/3 |
| `AutonomyManager.snapshot_state` | 27 | 13 | 7/3/0 |
| `AutonomyManager.restore_state` | 41 | 19 | 6/3/0 |
| `AutonomyManager.mark_inbound_activity` | 61 | 2 | 0/0/0 |
| `AutonomyManager.mark_response_activity` | 64 | 2 | 0/0/0 |
| `AutonomyManager.update_from_bundle` | 67 | 13 | 0/0/0 |
| `AutonomyManager.apply_scheduler_override` | 81 | 16 | 0/0/0 |
| `AutonomyManager.filter_autonomous_actions` | 98 | 32 | 0/0/0 |
| `AutonomyManager.chat_texts_from_actions` | 132 | 9 | 0/0/0 |
| `AutonomyManager.seconds_since_activity` | 142 | 4 | 15/9/0 |
| `AutonomyManager.activity_snapshot` | 147 | 22 | 6/6/0 |
| `AutonomyManager.agent_state` | 170 | 21 | 20/16/0 |
| `AutonomyManager.consume_delay_hint_seconds` | 192 | 4 | 0/0/0 |
| `AutonomyManager.last_inbound_ts` | 198 | 2 | 9/3/0 |
| `AutonomyManager.last_response_ts` | 202 | 2 | 9/3/0 |
| `AutonomyManager.mood` | 206 | 2 | 36/16/0 |
| `AutonomyManager.mood_ts` | 210 | 2 | 14/2/0 |
| `AutonomyManager.mood_source` | 214 | 2 | 9/2/0 |
| `AutonomyManager.status` | 218 | 2 | 48/22/1 |
| `AutonomyManager.status_ts` | 222 | 2 | 14/2/0 |
| `AutonomyManager.status_source` | 226 | 2 | 9/3/0 |
| `AutonomyManager.autonomy_decision` | 230 | 2 | 35/14/1 |
| `AutonomyManager.set_activity_from_timestamp` | 233 | 4 | 0/0/0 |
| `AutonomyManager.normalize_autonomy_decision` | 239 | 10 | 0/0/0 |
| `AutonomyManager._sanitize_autonomy_delay_hint` | 251 | 8 | 5/1/0 |
| `AutonomyManager._normalize_autonomy_decision_value` | 261 | 7 | 4/0/0 |
| `AutonomyManager._clean_optional_text` | 270 | 5 | 5/0/0 |
| `AutonomyManager._derive_status_from_actions` | 276 | 6 | 3/0/0 |
| `AutonomyManager._status_from_command` | 284 | 11 | 3/0/0 |

#### `src/gpt5_roleplay_system/background.py` — 106 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `spawn_supervised` | 35 | 28 | 12/8/0 |
| `_report` | 65 | 27 | 1/0/0 |
| `pending_background_tasks` | 94 | 3 | 0/3/0 |
| `drain_background_tasks` | 99 | 8 | 0/7/0 |

#### `src/gpt5_roleplay_system/background_tasks.py` — 248 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `FactsDeduplicationService.__init__` | 18 | 11 | 26/112/3 |
| `FactsDeduplicationService.run_forever` | 30 | 8 | 5/0/1 |
| `FactsDeduplicationService.run_once` | 39 | 102 | 3/14/0 |
| `ExperienceDeduplicationService.__init__` | 144 | 8 | 26/112/3 |
| `ExperienceDeduplicationService.run_forever` | 153 | 8 | 5/0/1 |
| `ExperienceDeduplicationService.run_once` | 162 | 87 | 3/14/0 |

#### `src/gpt5_roleplay_system/config.py` — 217 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `_resolve_default_config_path` | 150 | 10 | 2/0/0 |
| `_load_yaml` | 162 | 6 | 3/0/0 |
| `_env_bool` | 170 | 10 | 2/0/0 |
| `_parse_optional_bool` | 182 | 16 | 5/0/0 |
| `_pick_model` | 200 | 12 | 2/0/0 |
| `load_config` | 214 | 4 | 2/10/8 |

#### `src/gpt5_roleplay_system/config_loader.py` — 504 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `_represent_multiline_str` | 34 | 3 | 1/0/0 |
| `ConfigLoader.__init__` | 43 | 2 | 26/112/3 |
| `ConfigLoader.load` | 46 | 116 | 9/5/1 |
| `ConfigLoader.ensure_persona_profile` | 163 | 38 | 1/1/0 |
| `ConfigLoader.load_llm_config` | 202 | 92 | 1/0/0 |
| `ConfigLoader.load_memory_config` | 295 | 24 | 1/0/0 |
| `ConfigLoader.load_knowledge_config` | 320 | 31 | 1/0/0 |
| `ConfigLoader.load_facts_config` | 352 | 14 | 1/0/0 |
| `ConfigLoader.load_episode_config` | 367 | 14 | 1/0/0 |
| `ConfigLoader.load_autonomy_config` | 382 | 26 | 1/0/0 |
| `ConfigLoader.load_neo4j_config` | 409 | 8 | 1/0/0 |
| `ConfigLoader._resolve_default_config_path` | 418 | 10 | 2/0/0 |
| `ConfigLoader._load_yaml` | 430 | 6 | 3/0/0 |
| `ConfigLoader._write_yaml` | 438 | 11 | 1/0/0 |
| `ConfigLoader._find_case_insensitive_key` | 451 | 8 | 2/0/0 |
| `ConfigLoader._env_bool` | 460 | 10 | 2/0/0 |
| `ConfigLoader._parse_optional_bool` | 472 | 16 | 5/0/0 |
| `ConfigLoader._pick_model` | 490 | 12 | 2/0/0 |
| `ConfigLoader._getenv` | 503 | 2 | 44/0/0 |

#### `src/gpt5_roleplay_system/config_validation.py` — 48 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|

#### `src/gpt5_roleplay_system/context_builder.py` — 754 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `ExperienceIndexProtocol.is_enabled` | 27 | 2 | 13/3/1 |
| `ExperienceIndexProtocol.search` | 30 | 2 | 11/4/1 |
| `ContextBuilder.__init__` | 35 | 40 | 26/112/3 |
| `ContextBuilder.upsert_persona_profile` | 76 | 6 | 3/0/0 |
| `ContextBuilder.update_environment` | 83 | 31 | 4/15/0 |
| `ContextBuilder.build_chat` | 115 | 22 | 1/0/0 |
| `ContextBuilder.merge_participants` | 138 | 9 | 1/0/0 |
| `ContextBuilder.participants_from_messages` | 148 | 26 | 3/0/0 |
| `ContextBuilder.participants_for_autonomy` | 175 | 22 | 1/0/0 |
| `ContextBuilder.build_context` | 198 | 38 | 2/0/0 |
| `ContextBuilder.build_facts_context` | 237 | 22 | 1/0/0 |
| `ContextBuilder.collapse_batch` | 260 | 32 | 1/0/0 |
| `ContextBuilder.participant_payload` | 293 | 12 | 3/0/0 |
| `ContextBuilder.chat_payload` | 306 | 37 | 9/0/0 |
| `ContextBuilder.environment_payload` | 344 | 8 | 0/0/0 |
| `ContextBuilder.snapshot_state` | 353 | 16 | 7/3/0 |
| `ContextBuilder.restore_state` | 370 | 31 | 6/3/0 |
| `ContextBuilder._resolve_participants` | 402 | 41 | 3/0/0 |
| `ContextBuilder._memory_items_to_chats` | 444 | 22 | 8/0/0 |
| `ContextBuilder._people_facts_for_participants` | 467 | 33 | 4/0/0 |
| `ContextBuilder._record_display_name` | 501 | 8 | 18/0/0 |
| `ContextBuilder._full_name_for` | 510 | 4 | 13/0/0 |
| `ContextBuilder._update_last_seen_cache_from_agents` | 515 | 14 | 3/0/0 |
| `ContextBuilder._update_last_seen_from_agents` | 530 | 15 | 5/0/0 |
| `ContextBuilder._schedule_update_last_seen_from_agents` | 546 | 12 | 3/0/0 |
| `ContextBuilder._participant_key` | 559 | 5 | 12/0/0 |
| `ContextBuilder._normalize_participant_records` | 565 | 12 | 3/0/0 |
| `ContextBuilder._is_self_participant` | 578 | 6 | 22/0/0 |
| `ContextBuilder._is_self_message` | 585 | 8 | 6/0/0 |
| `ContextBuilder._chat_full_name` | 594 | 8 | 3/0/0 |
| `ContextBuilder._normalize_name` | 604 | 2 | 18/0/0 |
| `ContextBuilder._normalize_name_for_match` | 608 | 2 | 18/0/0 |
| `ContextBuilder._name_key` | 611 | 8 | 12/0/0 |
| `ContextBuilder._canonical_name` | 621 | 9 | 11/0/0 |
| `ContextBuilder._persona_instructions` | 631 | 5 | 3/0/0 |
| `ContextBuilder._dedupe_preserve_order` | 638 | 10 | 15/0/0 |
| `ContextBuilder._normalize_fact_key` | 650 | 5 | 15/0/0 |
| `ContextBuilder._experiences_for_persona` | 656 | 11 | 3/0/0 |
| `ContextBuilder._collapse_near_duplicate_experiences` | 668 | 44 | 3/0/0 |
| `ContextBuilder._normalize_experience_text` | 714 | 3 | 3/0/0 |
| `ContextBuilder._near_duplicate_similarity_score` | 719 | 10 | 3/0/0 |
| `ContextBuilder._token_jaccard` | 731 | 7 | 5/0/0 |
| `ContextBuilder._normalize_entities` | 740 | 15 | 5/0/0 |

#### `src/gpt5_roleplay_system/controller.py` — 293 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `SessionController.__init__` | 21 | 71 | 26/112/3 |
| `SessionController.process_chat` | 93 | 6 | 6/79/1 |
| `SessionController.process_chat_batch` | 100 | 6 | 4/16/0 |
| `SessionController.update_environment` | 107 | 4 | 4/15/0 |
| `SessionController.set_persona` | 112 | 12 | 3/3/0 |
| `SessionController.set_user_id` | 125 | 9 | 4/3/0 |
| `SessionController.set_llm_chat_enabled` | 135 | 2 | 5/12/0 |
| `SessionController.llm_chat_enabled` | 138 | 2 | 6/4/0 |
| `SessionController.knowledge_store` | 141 | 2 | 30/82/4 |
| `SessionController.memory_summary` | 144 | 2 | 0/1/1 |
| `SessionController.generate_autonomous_actions` | 147 | 2 | 3/6/0 |
| `SessionController.consume_autonomy_delay_hint_seconds` | 150 | 2 | 3/7/0 |
| `SessionController.activity_snapshot` | 153 | 2 | 6/6/0 |
| `SessionController.persona` | 156 | 2 | 138/146/8 |
| `SessionController.user_id` | 159 | 2 | 481/182/9 |
| `SessionController.flush_state` | 162 | 5 | 1/6/0 |
| `SessionController._reset_state_store` | 168 | 4 | 1/0/0 |
| `SessionController._rebind_state_store` | 173 | 21 | 3/0/0 |
| `SessionController._schedule_state_save` | 195 | 13 | 3/2/0 |
| `SessionController._start_state_save` | 209 | 13 | 1/0/0 |
| `SessionController._save_state_now` | 223 | 19 | 2/0/0 |
| `SessionController._sync_user_id_from_pipeline` | 243 | 15 | 3/0/0 |
| `SessionController._maybe_update_persona_from_payloads` | 259 | 17 | 3/0/0 |
| `SessionController._ensure_persona_profile` | 277 | 17 | 1/0/0 |

#### `src/gpt5_roleplay_system/cypher_utils.py` — 8 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `quote_cypher_identifier` | 3 | 6 | 4/8/0 |

#### `src/gpt5_roleplay_system/deduplication.py` — 210 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `parse_deduped_facts_response` | 12 | 31 | 5/0/0 |
| `safe_float` | 45 | 5 | 6/0/0 |
| `parse_experience_timestamp` | 52 | 15 | 2/0/0 |
| `extract_experience_window` | 69 | 14 | 2/0/0 |
| `experience_windows_are_close` | 85 | 16 | 1/0/0 |
| `ExperienceDedupePlanner.build_plans` | 105 | 106 | 2/0/0 |

#### `src/gpt5_roleplay_system/episode_manager.py` — 193 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `ExperienceIndexProtocol.is_enabled` | 18 | 2 | 13/3/1 |
| `ExperienceIndexProtocol.add_record_async` | 21 | 2 | 5/3/0 |
| `EpisodeManager.__init__` | 26 | 29 | 26/112/3 |
| `EpisodeManager.snapshot_state` | 56 | 5 | 7/3/0 |
| `EpisodeManager.restore_state` | 62 | 5 | 6/3/0 |
| `EpisodeManager.schedule_check` | 68 | 18 | 0/0/0 |
| `EpisodeManager.maybe_finalize` | 87 | 53 | 1/0/0 |
| `EpisodeManager.wait_for_idle` | 141 | 8 | 1/0/0 |
| `EpisodeManager._summarize_episode` | 150 | 21 | 3/0/0 |
| `EpisodeManager._episode_metadata` | 173 | 21 | 3/0/0 |

#### `src/gpt5_roleplay_system/experience_vector.py` — 186 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `EmbeddingClient.embed` | 19 | 2 | 5/4/0 |
| `EmbeddingClient.is_available` | 22 | 2 | 4/3/0 |
| `NullEmbeddingClient.embed` | 27 | 2 | 5/4/0 |
| `NullEmbeddingClient.is_available` | 30 | 2 | 4/3/0 |
| `OpenAIEmbeddingClient.__init__` | 35 | 17 | 26/112/3 |
| `OpenAIEmbeddingClient.is_available` | 53 | 2 | 4/3/0 |
| `OpenAIEmbeddingClient.embed` | 56 | 29 | 5/4/0 |
| `OpenAIEmbeddingClient._get_cached` | 86 | 10 | 1/0/0 |
| `OpenAIEmbeddingClient._set_cached` | 97 | 7 | 1/0/0 |
| `ExperienceVectorIndex.__init__` | 113 | 5 | 26/112/3 |
| `ExperienceVectorIndex.is_enabled` | 119 | 2 | 13/3/1 |
| `ExperienceVectorIndex.add_record_async` | 122 | 4 | 5/3/0 |
| `ExperienceVectorIndex.search` | 127 | 7 | 11/4/1 |
| `ExperienceVectorIndex._add_record_sync` | 135 | 9 | 3/0/0 |
| `ExperienceVectorIndex._embed_query` | 145 | 5 | 2/0/0 |
| `ExperienceVectorIndex._search_with_embedding` | 151 | 16 | 3/0/0 |
| `ExperienceVectorIndex._trim_locked` | 168 | 6 | 1/0/0 |
| `_cosine_similarity` | 176 | 11 | 1/0/0 |

#### `src/gpt5_roleplay_system/fact_manager.py` — 465 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `FactManager.__init__` | 18 | 34 | 26/112/3 |
| `FactManager.snapshot_state` | 53 | 16 | 7/3/0 |
| `FactManager.restore_state` | 70 | 37 | 6/3/0 |
| `FactManager.recover_pending_from_memory` | 108 | 12 | 0/0/0 |
| `FactManager.maybe_schedule_periodic_sweep` | 121 | 14 | 2/0/0 |
| `FactManager.schedule_store_facts` | 136 | 14 | 2/0/0 |
| `FactManager.store_facts` | 151 | 75 | 4/1/0 |
| `FactManager.wait_for_idle` | 227 | 11 | 1/0/0 |
| `FactManager._enqueue_fact_messages` | 239 | 23 | 5/7/0 |
| `FactManager._fact_message_key` | 263 | 6 | 8/0/0 |
| `FactManager._facts_pending_age_seconds` | 270 | 5 | 5/0/0 |
| `FactManager._fact_flush_reason` | 276 | 13 | 3/0/0 |
| `FactManager._maybe_start_facts_worker` | 290 | 27 | 5/8/0 |
| `FactManager._facts_worker` | 318 | 43 | 3/0/0 |
| `FactManager._extract_and_store_facts` | 362 | 35 | 3/0/0 |
| `FactManager._serialize_inbound_chat_state` | 399 | 9 | 3/0/0 |
| `FactManager._deserialize_inbound_chat_state` | 410 | 11 | 3/0/0 |
| `FactManager._memory_items_to_chats` | 423 | 13 | 8/0/0 |
| `FactManager._participant_key` | 438 | 4 | 12/0/0 |
| `FactManager._normalize_name_for_match` | 444 | 3 | 18/0/0 |
| `FactManager._normalize_fact_key` | 449 | 5 | 15/0/0 |
| `FactManager._dedupe_preserve_order` | 456 | 10 | 15/0/0 |

#### `src/gpt5_roleplay_system/llm.py` — 2049 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `_field_as_dict` | 40 | 32 | 7/0/2 |
| `_field_get` | 74 | 6 | 30/0/7 |
| `_to_int` | 82 | 5 | 15/0/0 |
| `_to_float` | 89 | 5 | 1/0/0 |
| `_coerce_text_content` | 96 | 24 | 16/0/0 |
| `_message_content_text` | 121 | 2 | 11/0/3 |
| `_truncate_text` | 125 | 6 | 2/0/0 |
| `_json_safe` | 133 | 21 | 7/0/2 |
| `_extract_reasoning_tokens` | 156 | 11 | 1/0/0 |
| `PromptCacheUsageSample.uncached_prompt_tokens` | 179 | 2 | 4/1/1 |
| `_extract_prompt_cache_usage` | 183 | 35 | 1/3/2 |
| `PromptCacheStats.__init__` | 221 | 12 | 26/112/3 |
| `PromptCacheStats.record` | 234 | 15 | 62/3/4 |
| `PromptCacheStats.snapshot` | 250 | 24 | 35/36/11 |
| `prompt_cache_stats_snapshot` | 279 | 2 | 1/0/0 |
| `log_prompt_cache_summary` | 283 | 23 | 2/0/0 |
| `LLMClient.is_addressed_to_me` | 449 | 9 | 5/18/2 |
| `LLMClient.generate_response` | 459 | 2 | 6/16/0 |
| `LLMClient.extract_facts` | 462 | 2 | 4/16/2 |
| `LLMClient.summarize_overflow` | 465 | 2 | 6/22/0 |
| `LLMClient.summarize_episode` | 468 | 2 | 6/8/0 |
| `LLMClient.generate_bundle` | 471 | 19 | 8/17/3 |
| `LLMClient.generate_state_update` | 491 | 16 | 3/2/0 |
| `LLMClient.generate_autonomous_bundle` | 508 | 6 | 6/2/2 |
| `LLMClient.extract_facts_from_evidence_sync` | 515 | 7 | 3/3/0 |
| `LLMClient.consume_reasoning_trace` | 523 | 2 | 4/4/0 |
| `LLMClient.consume_quality_fallback_events` | 526 | 2 | 2/5/0 |
| `RuleBasedLLMClient.is_addressed_to_me` | 531 | 27 | 5/18/2 |
| `RuleBasedLLMClient.generate_response` | 559 | 3 | 6/16/0 |
| `RuleBasedLLMClient.extract_facts` | 563 | 2 | 4/16/2 |
| `RuleBasedLLMClient.summarize_overflow` | 566 | 2 | 6/22/0 |
| `RuleBasedLLMClient.summarize_episode` | 569 | 2 | 6/8/0 |
| `RuleBasedLLMClient.generate_bundle` | 572 | 15 | 8/17/3 |
| `RuleBasedLLMClient.generate_autonomous_bundle` | 588 | 6 | 6/2/2 |
| `RuleBasedLLMClient.generate_autonomous_bundle` | 595 | 17 | 6/2/2 |
| `EchoLLMClient.is_addressed_to_me` | 615 | 9 | 5/18/2 |
| `EchoLLMClient.generate_response` | 625 | 4 | 6/16/0 |
| `EchoLLMClient.extract_facts` | 630 | 2 | 4/16/2 |
| `EchoLLMClient.summarize_overflow` | 633 | 2 | 6/22/0 |
| `EchoLLMClient.summarize_episode` | 636 | 2 | 6/8/0 |
| `EchoLLMClient.generate_bundle` | 639 | 15 | 8/17/3 |
| `OpenRouterLLMClient.__init__` | 657 | 69 | 26/112/3 |
| `OpenRouterLLMClient._prompt_manager_instance` | 727 | 7 | 16/0/0 |
| `OpenRouterLLMClient._reasoning_trace_store` | 735 | 7 | 3/0/0 |
| `OpenRouterLLMClient._clear_reasoning_trace` | 743 | 4 | 5/0/0 |
| `OpenRouterLLMClient.consume_reasoning_trace` | 748 | 5 | 4/4/0 |
| `OpenRouterLLMClient._quality_fallback_event_store` | 754 | 7 | 2/0/0 |
| `OpenRouterLLMClient._quality_fallback_warned_store` | 762 | 7 | 1/0/0 |
| `OpenRouterLLMClient._quality_fallback_lock_instance` | 770 | 7 | 2/0/0 |
| `OpenRouterLLMClient._record_quality_fallback` | 778 | 31 | 11/0/0 |
| `OpenRouterLLMClient.consume_quality_fallback_events` | 810 | 5 | 2/5/0 |
| `OpenRouterLLMClient._record_reasoning_trace` | 816 | 55 | 11/0/0 |
| `OpenRouterLLMClient._record_cache_usage` | 872 | 19 | 11/9/0 |
| `OpenRouterLLMClient._provider_payload` | 892 | 18 | 3/0/2 |
| `OpenRouterLLMClient._apply_extra_body` | 911 | 18 | 6/0/0 |
| `OpenRouterLLMClient.is_addressed_to_me` | 930 | 26 | 5/18/2 |
| `OpenRouterLLMClient.generate_response` | 957 | 3 | 6/16/0 |
| `OpenRouterLLMClient.extract_facts` | 961 | 3 | 4/16/2 |
| `OpenRouterLLMClient.summarize_overflow` | 965 | 37 | 6/22/0 |
| `OpenRouterLLMClient.summarize_episode` | 1003 | 30 | 6/8/0 |
| `OpenRouterLLMClient.generate_bundle` | 1034 | 39 | 8/17/3 |
| `OpenRouterLLMClient.generate_state_update` | 1074 | 28 | 3/2/0 |
| `OpenRouterLLMClient.generate_autonomous_bundle` | 1103 | 23 | 6/2/2 |
| `OpenRouterLLMClient._system_prompt` | 1127 | 2 | 0/4/0 |
| `OpenRouterLLMClient._system_prompt_for_context` | 1130 | 2 | 1/3/0 |
| `OpenRouterLLMClient._facts_system_prompt` | 1133 | 2 | 1/2/0 |
| `OpenRouterLLMClient._state_system_prompt` | 1136 | 2 | 0/0/0 |
| `OpenRouterLLMClient._state_system_prompt_for_context` | 1139 | 2 | 1/1/0 |
| `OpenRouterLLMClient._autonomous_system_prompt` | 1142 | 2 | 0/2/0 |
| `OpenRouterLLMClient._autonomous_system_prompt_for_context` | 1145 | 2 | 1/1/0 |
| `OpenRouterLLMClient._structured_parse_cache` | 1148 | 6 | 1/1/0 |
| `OpenRouterLLMClient._structured_parse_key` | 1156 | 13 | 3/1/0 |
| `OpenRouterLLMClient._is_no_endpoint_not_found` | 1171 | 13 | 3/0/0 |
| `OpenRouterLLMClient._request_structured_raw_create` | 1185 | 18 | 5/0/0 |
| `OpenRouterLLMClient._request_structured` | 1204 | 82 | 2/9/0 |
| `OpenRouterLLMClient._request_structured_with_trace` | 1287 | 16 | 4/0/0 |
| `OpenRouterLLMClient._request_bundle` | 1304 | 19 | 1/3/0 |
| `OpenRouterLLMClient._request_state_update` | 1324 | 19 | 1/2/0 |
| `OpenRouterLLMClient._request_autonomous_bundle` | 1344 | 17 | 1/1/0 |
| `OpenRouterLLMClient._request_facts_from_chat` | 1362 | 13 | 1/2/0 |
| `OpenRouterLLMClient._request_facts_from_messages` | 1376 | 39 | 2/2/0 |
| `OpenRouterLLMClient.extract_facts_from_evidence_sync` | 1416 | 24 | 3/3/0 |
| `OpenRouterLLMClient._request_text` | 1441 | 30 | 0/0/2 |
| `OpenRouterLLMClient._request_text_with_model` | 1472 | 36 | 5/12/0 |
| `OpenRouterLLMClient._serialize_payload` | 1510 | 3 | 7/0/0 |
| `OpenRouterLLMClient._canonicalize_for_prompt` | 1515 | 9 | 22/0/0 |
| `OpenRouterLLMClient._stable_participants_payload` | 1526 | 10 | 5/0/0 |
| `OpenRouterLLMClient._entity_prompt_payload` | 1538 | 10 | 7/0/0 |
| `OpenRouterLLMClient._stable_entity_list` | 1550 | 29 | 4/0/0 |
| `OpenRouterLLMClient._format_address_check` | 1580 | 9 | 1/2/0 |
| `OpenRouterLLMClient._format_context` | 1590 | 8 | 2/7/0 |
| `OpenRouterLLMClient._format_facts_context_from_messages` | 1599 | 21 | 1/3/0 |
| `OpenRouterLLMClient._filter_facts_entities` | 1622 | 48 | 2/0/0 |
| `OpenRouterLLMClient._minimal_people_fact_profile` | 1672 | 19 | 3/0/0 |
| `OpenRouterLLMClient._profile_mentioned_in_text` | 1693 | 6 | 3/0/0 |
| `OpenRouterLLMClient._name_mentioned_in_text` | 1701 | 17 | 5/0/0 |
| `OpenRouterLLMClient._format_autonomous_context` | 1719 | 2 | 1/3/0 |
| `OpenRouterLLMClient._chat_payload` | 1723 | 25 | 4/0/0 |
| `OpenRouterLLMClient._participant_payload` | 1750 | 16 | 4/0/0 |
| `_bundle_from_structured` | 1768 | 43 | 2/14/0 |
| `_state_update_from_structured` | 1813 | 28 | 1/2/0 |
| `_facts_from_structured` | 1843 | 11 | 4/0/0 |
| `_strip_emote_wrapping_asterisks` | 1856 | 5 | 3/0/0 |
| `_normalize_text_action_content` | 1863 | 5 | 5/0/0 |
| `_actions_from_structured` | 1870 | 28 | 3/0/0 |
| `_command_from_structured` | 1900 | 33 | 3/0/0 |
| `_first_chat_text` | 1935 | 5 | 3/0/0 |
| `_normalize_name` | 1942 | 5 | 18/0/0 |
| `_name_in_text` | 1949 | 10 | 5/0/0 |
| `_clean_json` | 1961 | 25 | 3/2/0 |
| `_collect_other_names` | 1988 | 34 | 1/0/0 |
| `_parse_bool_response` | 2024 | 7 | 1/0/0 |
| `_normalize_autonomy_decision` | 2033 | 7 | 7/0/0 |
| `_optional_positive_float` | 2042 | 8 | 5/0/0 |

#### `src/gpt5_roleplay_system/llm_prompts.py` — 999 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `PromptManager.system_prompt_for_context` | 20 | 13 | 1/0/0 |
| `PromptManager.state_system_prompt_for_context` | 34 | 13 | 1/0/0 |
| `PromptManager.autonomous_system_prompt_for_context` | 48 | 13 | 1/0/0 |
| `PromptManager.facts_system_prompt` | 62 | 14 | 1/0/0 |
| `PromptManager.address_check_system_prompt` | 78 | 9 | 1/0/0 |
| `PromptManager.continuity_summary_system_prompt` | 89 | 22 | 1/0/0 |
| `PromptManager.episodic_summary_system_prompt` | 113 | 8 | 1/0/0 |
| `PromptManager.format_address_check` | 122 | 69 | 1/0/0 |
| `PromptManager.format_context` | 192 | 48 | 1/0/0 |
| `PromptManager.format_facts_context_from_messages` | 241 | 31 | 1/0/0 |
| `PromptManager.format_continuity_summary_context` | 273 | 18 | 1/0/0 |
| `PromptManager.format_episode_summary_context` | 292 | 12 | 1/0/0 |
| `PromptManager.format_autonomous_context` | 305 | 44 | 1/0/0 |
| `PromptManager.chat_payload` | 351 | 25 | 9/0/0 |
| `PromptManager._summary_message_payload` | 378 | 7 | 2/0/0 |
| `PromptManager._summary_sender_label` | 387 | 20 | 1/0/0 |
| `PromptManager.participant_payload` | 409 | 16 | 3/0/0 |
| `PromptManager._serialize_payload` | 427 | 2 | 7/0/0 |
| `PromptManager.split_people_facts_prompt_sections` | 431 | 43 | 3/0/0 |
| `PromptManager.split_agent_state_prompt_sections` | 476 | 27 | 3/0/0 |
| `PromptManager._canonicalize_for_prompt` | 505 | 9 | 22/0/0 |
| `PromptManager._stable_participants_payload` | 516 | 10 | 5/0/0 |
| `PromptManager._entity_prompt_payload` | 528 | 10 | 7/0/0 |
| `PromptManager._stable_entity_list` | 540 | 29 | 4/0/0 |
| `PromptManager._stable_objects_catalog_payload` | 571 | 14 | 2/0/0 |
| `PromptManager._object_proximity_overlay` | 587 | 33 | 2/0/0 |
| `PromptManager._extract_object_distance` | 622 | 23 | 1/0/0 |
| `PromptManager._is_dynamic_object_key` | 647 | 7 | 1/0/0 |
| `PromptManager._parse_xyz_triplet` | 656 | 28 | 3/0/0 |
| `PromptManager._try_float` | 686 | 5 | 12/0/0 |
| `PromptManager._round_recency_seconds` | 693 | 18 | 3/0/0 |
| `PromptManager._recency_bucket_label` | 713 | 18 | 6/0/0 |
| `PromptManager._relative_day_label` | 733 | 9 | 1/0/0 |
| `PromptManager._filter_facts_entities` | 744 | 48 | 2/0/0 |
| `PromptManager._minimal_people_fact_profile` | 794 | 19 | 3/0/0 |
| `PromptManager._profile_mentioned_in_text` | 815 | 6 | 3/0/0 |
| `PromptManager._name_in_text` | 823 | 10 | 5/0/0 |
| `PromptManager._normalize_name` | 835 | 5 | 18/0/0 |
| `PromptManager._name_mentioned_in_text` | 842 | 17 | 5/0/0 |
| `PromptManager.system_prompt` | 861 | 54 | 10/26/7 |
| `PromptManager.state_system_prompt` | 917 | 18 | 2/0/0 |
| `PromptManager.autonomous_system_prompt` | 937 | 56 | 2/0/0 |
| `PromptManager._merge_prompt_sections` | 995 | 5 | 5/0/0 |

#### `src/gpt5_roleplay_system/llm_response_mapper.py` — 197 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `ResponseMapper.bundle_from_structured` | 10 | 43 | 0/1/0 |
| `ResponseMapper.state_update_from_structured` | 54 | 29 | 0/0/0 |
| `ResponseMapper.facts_from_structured` | 84 | 13 | 2/0/0 |
| `ResponseMapper._actions_from_structured` | 98 | 27 | 3/0/0 |
| `ResponseMapper._command_from_structured` | 126 | 33 | 3/0/0 |
| `ResponseMapper._first_chat_text` | 161 | 5 | 3/0/0 |
| `ResponseMapper._strip_emote_wrapping_asterisks` | 168 | 5 | 3/0/0 |
| `ResponseMapper._normalize_text_action_content` | 174 | 5 | 5/0/0 |
| `ResponseMapper._normalize_autonomy_decision` | 181 | 7 | 7/0/0 |
| `ResponseMapper._optional_positive_float` | 190 | 8 | 5/0/0 |

#### `src/gpt5_roleplay_system/llm_structured_parser.py` — 186 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `StructuredParser.__init__` | 12 | 2 | 26/112/3 |
| `StructuredParser.request_structured` | 15 | 77 | 0/0/0 |
| `StructuredParser._request_structured_raw_create` | 94 | 20 | 5/0/0 |
| `StructuredParser._structured_parse_key` | 116 | 13 | 3/1/0 |
| `StructuredParser._is_no_endpoint_not_found` | 131 | 13 | 3/0/0 |
| `StructuredParser._message_content_text` | 146 | 2 | 11/0/3 |
| `StructuredParser._coerce_text_content` | 150 | 24 | 16/0/0 |
| `StructuredParser.clean_json` | 175 | 12 | 3/0/0 |

#### `src/gpt5_roleplay_system/llm_transport.py` — 130 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `OpenRouterTransport.__init__` | 7 | 23 | 26/112/3 |
| `OpenRouterTransport.request_text` | 31 | 34 | 0/0/0 |
| `OpenRouterTransport.apply_extra_body` | 66 | 18 | 1/0/0 |
| `OpenRouterTransport._provider_payload` | 85 | 16 | 3/0/2 |
| `OpenRouterTransport._coerce_text_content` | 103 | 24 | 16/0/0 |
| `OpenRouterTransport._message_content_text` | 129 | 2 | 11/0/3 |

#### `src/gpt5_roleplay_system/memory.py` — 484 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `RollingBuffer.__init__` | 61 | 3 | 26/112/3 |
| `RollingBuffer.add_user_message` | 65 | 10 | 2/9/0 |
| `RollingBuffer.add_ai_message` | 76 | 10 | 7/4/0 |
| `RollingBuffer.items` | 87 | 2 | 125/15/1 |
| `RollingBuffer.snapshot` | 90 | 2 | 35/36/11 |
| `RollingBuffer.restore` | 93 | 9 | 4/7/0 |
| `RollingBuffer.trim_to_last` | 103 | 7 | 2/0/0 |
| `RollingBuffer.trim_after_episode` | 111 | 22 | 3/2/0 |
| `RollingBuffer._trim` | 134 | 11 | 3/0/0 |
| `RollingBuffer._warn_if_oversized` | 146 | 10 | 2/0/0 |
| `MemoryCompressor.compress` | 159 | 2 | 4/0/0 |
| `SimpleMemoryCompressor.compress` | 164 | 5 | 4/0/0 |
| `ConversationMemory.__init__` | 172 | 14 | 26/112/3 |
| `ConversationMemory.add_message` | 187 | 10 | 2/38/0 |
| `ConversationMemory.add_ai_message` | 198 | 10 | 7/4/0 |
| `ConversationMemory.recent` | 209 | 2 | 40/54/0 |
| `ConversationMemory.summary` | 212 | 2 | 100/114/4 |
| `ConversationMemory.summary_meta` | 215 | 2 | 47/34/0 |
| `ConversationMemory.clamp_summary_range_end` | 218 | 13 | 1/0/0 |
| `ConversationMemory.apply_summary` | 232 | 15 | 1/4/0 |
| `ConversationMemory.drain_overflow` | 248 | 4 | 1/19/0 |
| `ConversationMemory.requeue_overflow` | 253 | 6 | 4/1/0 |
| `ConversationMemory.compress_overflow` | 260 | 20 | 4/1/0 |
| `ConversationMemory.snapshot` | 281 | 14 | 35/36/11 |
| `ConversationMemory.restore` | 296 | 30 | 4/7/0 |
| `ConversationMemory._compress_if_needed` | 327 | 45 | 2/0/0 |
| `ConversationMemory._update_summary_meta` | 373 | 19 | 2/0/0 |
| `ConversationMemory.mark_summarized` | 393 | 16 | 2/0/0 |
| `ConversationMemory.is_summarized` | 410 | 3 | 1/0/0 |
| `ExperienceStore.__init__` | 422 | 2 | 26/112/3 |
| `ExperienceStore.add` | 425 | 8 | 50/9/1 |
| `ExperienceStore.all` | 434 | 2 | 6/18/4 |
| `SimilaritySearch.search` | 439 | 2 | 11/4/1 |
| `SimpleSimilaritySearch.search` | 444 | 11 | 11/4/1 |
| `TokenSimilaritySearch.__init__` | 458 | 2 | 26/112/3 |
| `TokenSimilaritySearch.search` | 461 | 16 | 11/4/1 |
| `_tokenize` | 479 | 6 | 2/0/0 |

#### `src/gpt5_roleplay_system/models.py` — 70 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|

#### `src/gpt5_roleplay_system/name_utils.py` — 91 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `split_display_and_username` | 9 | 19 | 28/2/0 |
| `extract_username` | 30 | 3 | 10/4/0 |
| `extract_display_name` | 35 | 3 | 7/4/0 |
| `normalize_display_name` | 40 | 11 | 7/2/0 |
| `normalize_for_match` | 53 | 12 | 12/3/0 |
| `name_matches` | 67 | 25 | 13/2/0 |

#### `src/gpt5_roleplay_system/neo4j_experience_vector.py` — 266 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `Neo4jExperienceVectorIndex.__init__` | 21 | 18 | 26/112/3 |
| `Neo4jExperienceVectorIndex.is_enabled` | 40 | 2 | 13/3/1 |
| `Neo4jExperienceVectorIndex.add_record_async` | 43 | 4 | 5/3/0 |
| `Neo4jExperienceVectorIndex.search` | 48 | 4 | 11/4/1 |
| `Neo4jExperienceVectorIndex._session` | 53 | 2 | 24/3/0 |
| `Neo4jExperienceVectorIndex._ensure_vector_index` | 56 | 21 | 1/0/0 |
| `Neo4jExperienceVectorIndex._check_genai_available` | 78 | 17 | 1/0/0 |
| `Neo4jExperienceVectorIndex._external_available` | 96 | 2 | 4/0/0 |
| `Neo4jExperienceVectorIndex._add_record_sync` | 99 | 6 | 3/0/0 |
| `Neo4jExperienceVectorIndex._add_with_genai` | 106 | 37 | 1/0/0 |
| `Neo4jExperienceVectorIndex._add_with_external_embedder` | 144 | 32 | 2/0/0 |
| `Neo4jExperienceVectorIndex._search_sync` | 177 | 9 | 1/0/0 |
| `Neo4jExperienceVectorIndex._search_with_genai` | 187 | 28 | 1/0/0 |
| `Neo4jExperienceVectorIndex._search_with_embedding` | 216 | 20 | 3/0/0 |
| `_normalize_metadata` | 238 | 7 | 2/0/0 |
| `_records_from_query` | 247 | 20 | 2/0/0 |

#### `src/gpt5_roleplay_system/neo4j_store.py` — 663 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `_dedupe_preserve_order` | 14 | 10 | 15/0/0 |
| `_normalize_fact_key` | 26 | 6 | 15/0/0 |
| `KnowledgeStore.fetch_people` | 51 | 2 | 8/4/0 |
| `KnowledgeStore.fetch_people_by_name` | 54 | 2 | 3/2/0 |
| `KnowledgeStore.fetch_people_by_partial_name` | 57 | 2 | 3/2/0 |
| `KnowledgeStore.fetch_people_name_index` | 60 | 2 | 3/2/0 |
| `KnowledgeStore.upsert_person_facts` | 63 | 2 | 4/14/0 |
| `KnowledgeStore.upsert_relationship` | 66 | 2 | 2/0/0 |
| `KnowledgeStore.update_last_seen` | 69 | 2 | 4/3/0 |
| `InMemoryKnowledgeStore.__init__` | 74 | 2 | 26/112/3 |
| `InMemoryKnowledgeStore._merge_aliases` | 78 | 13 | 3/0/0 |
| `InMemoryKnowledgeStore.fetch_people` | 92 | 2 | 8/4/0 |
| `InMemoryKnowledgeStore.fetch_people_by_name` | 95 | 10 | 3/2/0 |
| `InMemoryKnowledgeStore.fetch_people_by_partial_name` | 106 | 11 | 3/2/0 |
| `InMemoryKnowledgeStore.fetch_people_name_index` | 118 | 10 | 3/2/0 |
| `InMemoryKnowledgeStore.upsert_person_facts` | 129 | 13 | 4/14/0 |
| `InMemoryKnowledgeStore.upsert_relationship` | 143 | 4 | 2/0/0 |
| `InMemoryKnowledgeStore.update_last_seen` | 148 | 9 | 4/3/0 |
| `Neo4jKnowledgeStore.__init__` | 160 | 8 | 26/112/3 |
| `Neo4jKnowledgeStore._system_session` | 169 | 2 | 1/0/0 |
| `Neo4jKnowledgeStore._session` | 172 | 2 | 24/3/0 |
| `Neo4jKnowledgeStore._ensure_database_and_constraints` | 175 | 28 | 1/0/0 |
| `Neo4jKnowledgeStore.fetch_people` | 204 | 30 | 8/4/0 |
| `Neo4jKnowledgeStore.fetch_people_by_name` | 235 | 32 | 3/2/0 |
| `Neo4jKnowledgeStore.fetch_people_by_partial_name` | 268 | 32 | 3/2/0 |
| `Neo4jKnowledgeStore.fetch_people_name_index` | 301 | 28 | 3/2/0 |
| `Neo4jKnowledgeStore._name_lower_owner_user_id` | 331 | 13 | 2/0/0 |
| `Neo4jKnowledgeStore.upsert_person_facts` | 345 | 77 | 4/14/0 |
| `Neo4jKnowledgeStore.upsert_relationship` | 423 | 9 | 2/0/0 |
| `Neo4jKnowledgeStore.update_last_seen` | 433 | 67 | 4/3/0 |
| `Neo4jKnowledgeStore.init_runtime_marker` | 501 | 9 | 2/0/0 |
| `Neo4jKnowledgeStore.mark_runtime_running` | 511 | 10 | 2/0/0 |
| `Neo4jKnowledgeStore.mark_runtime_success` | 522 | 12 | 2/0/0 |
| `Neo4jKnowledgeStore.mark_runtime_error` | 535 | 10 | 2/0/0 |
| `Neo4jKnowledgeStore.fetch_facts_dedupe_candidates` | 546 | 8 | 1/0/0 |
| `Neo4jKnowledgeStore.apply_deduped_facts` | 555 | 17 | 1/0/0 |
| `Neo4jKnowledgeStore.mark_facts_deduped` | 573 | 9 | 1/0/0 |
| `Neo4jKnowledgeStore.fetch_experience_dedupe_candidates` | 583 | 38 | 1/0/0 |
| `Neo4jKnowledgeStore.merge_experience_group` | 622 | 34 | 1/0/0 |
| `Neo4jKnowledgeStore.driver` | 658 | 2 | 4/4/9 |
| `Neo4jKnowledgeStore.database` | 662 | 2 | 15/5/11 |

#### `src/gpt5_roleplay_system/observability.py` — 59 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `Tracer.start_run` | 12 | 2 | 3/1/0 |
| `Tracer.log_event` | 15 | 2 | 21/3/0 |
| `Tracer.finish` | 18 | 2 | 5/1/0 |
| `NoOpTracer.start_run` | 23 | 2 | 3/1/0 |
| `NoOpTracer.log_event` | 26 | 2 | 21/3/0 |
| `NoOpTracer.finish` | 29 | 2 | 5/1/0 |
| `WandbTracer.__init__` | 34 | 9 | 26/112/3 |
| `WandbTracer.start_run` | 44 | 5 | 3/1/0 |
| `WandbTracer.log_event` | 50 | 5 | 21/3/0 |
| `WandbTracer.finish` | 56 | 4 | 5/1/0 |

#### `src/gpt5_roleplay_system/payload_contract.py` — 582 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `looks_like_uuid` | 23 | 2 | 9/0/0 |
| `is_ignored_user_id` | 27 | 2 | 4/0/0 |
| `canonical_identity_key` | 31 | 8 | 11/0/0 |
| `is_placeholder_self_id` | 41 | 15 | 2/0/0 |
| `normalize_participants` | 58 | 74 | 6/0/0 |
| `normalize_and_validate_payload` | 134 | 25 | 2/2/0 |
| `_normalize_participant_entry` | 161 | 25 | 4/0/0 |
| `_merge_participant_entries` | 188 | 12 | 3/0/0 |
| `_pick_richer_text` | 202 | 10 | 1/0/0 |
| `_pick_richer_full_name` | 214 | 10 | 1/0/0 |
| `_participant_names` | 226 | 7 | 3/0/0 |
| `_participant_name` | 235 | 3 | 5/0/0 |
| `_participant_entries_match_name` | 240 | 8 | 1/0/0 |
| `_ensure_participant_coverage` | 250 | 36 | 1/0/0 |
| `_collect_active_speakers` | 288 | 50 | 1/0/0 |
| `_message_sender_name` | 340 | 7 | 5/0/0 |
| `_is_self_identity` | 349 | 8 | 1/0/0 |
| `_repair_incoming_batch_and_incoming` | 359 | 74 | 1/0/0 |
| `_ordered_incoming_batch` | 435 | 37 | 1/0/0 |
| `_repair_summary_meta` | 474 | 47 | 1/0/0 |
| `_self_identity_warnings` | 523 | 32 | 1/0/0 |
| `_safe_float` | 557 | 5 | 10/0/0 |
| `_parse_timestamp` | 564 | 19 | 2/0/0 |

#### `src/gpt5_roleplay_system/persona_policy.py` — 5 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `is_chat_and_state_restricted_persona` | 4 | 2 | 8/0/0 |

#### `src/gpt5_roleplay_system/pipeline.py` — 3200 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `ExperienceIndexProtocol.is_enabled` | 74 | 2 | 13/3/1 |
| `ExperienceIndexProtocol.add_record_async` | 77 | 2 | 5/3/0 |
| `ExperienceIndexProtocol.search` | 80 | 2 | 11/4/1 |
| `MessagePipeline.__init__` | 85 | 160 | 26/112/3 |
| `MessagePipeline.snapshot_state` | 246 | 30 | 7/3/0 |
| `MessagePipeline.restore_state` | 277 | 122 | 6/3/0 |
| `MessagePipeline._snapshot_facts_state` | 400 | 14 | 3/0/0 |
| `MessagePipeline._restore_facts_state` | 415 | 43 | 3/0/0 |
| `MessagePipeline._recover_pending_facts_from_memory` | 459 | 12 | 3/0/0 |
| `MessagePipeline._serialize_inbound_chat_state` | 473 | 9 | 3/0/0 |
| `MessagePipeline._deserialize_inbound_chat_state` | 484 | 11 | 3/0/0 |
| `MessagePipeline.update_environment` | 496 | 52 | 4/15/0 |
| `MessagePipeline.set_persona` | 549 | 4 | 3/3/0 |
| `MessagePipeline.upsert_persona_profile` | 554 | 7 | 3/0/0 |
| `MessagePipeline.set_user_id` | 562 | 4 | 4/3/0 |
| `MessagePipeline.set_llm_chat_enabled` | 567 | 3 | 5/12/0 |
| `MessagePipeline.llm_chat_enabled` | 571 | 2 | 6/4/0 |
| `MessagePipeline.user_id` | 574 | 2 | 481/182/9 |
| `MessagePipeline.process_chat` | 577 | 2 | 6/79/1 |
| `MessagePipeline.process_chat_batch` | 580 | 145 | 4/16/0 |
| `MessagePipeline._apply_overflow_summary` | 726 | 48 | 2/0/0 |
| `MessagePipeline._build_chat` | 775 | 28 | 2/0/0 |
| `MessagePipeline._resolve_participants` | 804 | 47 | 3/0/0 |
| `MessagePipeline._merge_participants` | 852 | 7 | 2/0/0 |
| `MessagePipeline._build_context` | 860 | 157 | 2/0/0 |
| `MessagePipeline._collapse_batch` | 1018 | 40 | 2/0/0 |
| `MessagePipeline._handle_self_message` | 1059 | 9 | 1/0/0 |
| `MessagePipeline._is_duplicate_ai_text` | 1069 | 12 | 1/0/0 |
| `MessagePipeline._is_self_message` | 1082 | 11 | 6/0/0 |
| `MessagePipeline._chat_full_name` | 1094 | 8 | 3/0/0 |
| `MessagePipeline._self_name_candidates` | 1103 | 25 | 1/0/0 |
| `MessagePipeline._maybe_promote_self_user_id` | 1129 | 10 | 1/0/0 |
| `MessagePipeline._self_sender_id_for_payload` | 1140 | 5 | 2/0/0 |
| `MessagePipeline._is_ignored_user_id` | 1147 | 2 | 23/0/0 |
| `MessagePipeline._is_ignored_message` | 1150 | 2 | 4/0/0 |
| `MessagePipeline._normalize_name` | 1154 | 2 | 18/0/0 |
| `MessagePipeline._normalize_name_for_match` | 1158 | 2 | 18/0/0 |
| `MessagePipeline._name_key` | 1161 | 8 | 12/0/0 |
| `MessagePipeline._canonical_name` | 1171 | 9 | 11/0/0 |
| `MessagePipeline._dedupe_preserve_order` | 1182 | 10 | 15/0/0 |
| `MessagePipeline._normalize_fact_key` | 1194 | 6 | 15/0/0 |
| `MessagePipeline._fact_tokens` | 1202 | 8 | 2/0/0 |
| `MessagePipeline._fact_similarity` | 1212 | 8 | 1/0/0 |
| `MessagePipeline._is_near_duplicate` | 1222 | 11 | 1/0/0 |
| `MessagePipeline._people_facts_for_participants` | 1234 | 33 | 4/0/0 |
| `MessagePipeline._record_display_name` | 1268 | 15 | 18/0/0 |
| `MessagePipeline._full_name_for` | 1284 | 4 | 13/0/0 |
| `MessagePipeline._clean_name_candidate` | 1290 | 5 | 18/0/0 |
| `MessagePipeline._best_available_sender_full_name` | 1296 | 44 | 1/0/0 |
| `MessagePipeline._participant_payload` | 1341 | 12 | 4/0/0 |
| `MessagePipeline._normalize_participants` | 1354 | 18 | 6/1/0 |
| `MessagePipeline._select_primary_incoming_chat` | 1374 | 12 | 1/0/0 |
| `MessagePipeline._enforce_summary_before_recent` | 1387 | 35 | 3/1/0 |
| `MessagePipeline._check_payload_contract` | 1423 | 19 | 2/0/0 |
| `MessagePipeline._update_last_seen_cache_from_agents` | 1443 | 14 | 3/0/0 |
| `MessagePipeline._update_reappearance_signals_from_agents` | 1458 | 32 | 1/0/0 |
| `MessagePipeline._prune_reappearance_signals` | 1491 | 5 | 1/0/0 |
| `MessagePipeline._active_reappearance_signal` | 1497 | 10 | 1/0/0 |
| `MessagePipeline._update_last_seen_from_agents` | 1508 | 19 | 5/0/0 |
| `MessagePipeline._schedule_update_last_seen_from_agents` | 1528 | 12 | 3/0/0 |
| `MessagePipeline._memory_items_to_chats` | 1541 | 35 | 8/0/0 |
| `MessagePipeline._filter_overflow_against_summary` | 1577 | 30 | 2/2/0 |
| `MessagePipeline._chat_payload` | 1608 | 37 | 4/0/0 |
| `MessagePipeline._participant_key` | 1646 | 5 | 12/0/0 |
| `MessagePipeline._participants_from_messages` | 1652 | 29 | 8/0/0 |
| `MessagePipeline._facts_context` | 1682 | 15 | 3/0/0 |
| `MessagePipeline._maybe_schedule_fact_sweep` | 1698 | 16 | 2/0/0 |
| `MessagePipeline._enqueue_fact_messages` | 1715 | 42 | 5/7/0 |
| `MessagePipeline._mark_facts_swept` | 1758 | 8 | 1/0/0 |
| `MessagePipeline._fact_message_key` | 1767 | 6 | 8/0/0 |
| `MessagePipeline._facts_pending_age_seconds` | 1774 | 5 | 5/0/0 |
| `MessagePipeline._fact_flush_reason` | 1780 | 13 | 3/0/0 |
| `MessagePipeline._maybe_start_facts_worker` | 1794 | 32 | 5/8/0 |
| `MessagePipeline._facts_worker` | 1827 | 69 | 3/0/0 |
| `MessagePipeline._requeue_fact_evidence` | 1897 | 36 | 2/0/0 |
| `MessagePipeline._extract_and_store_facts` | 1934 | 32 | 3/0/0 |
| `MessagePipeline._store_facts` | 1967 | 54 | 4/0/0 |
| `MessagePipeline._schedule_store_facts` | 2022 | 7 | 2/0/0 |
| `MessagePipeline._update_participant_hints` | 2030 | 18 | 2/0/0 |
| `MessagePipeline._is_self_participant` | 2049 | 6 | 22/0/0 |
| `MessagePipeline.seconds_since_activity` | 2056 | 4 | 15/9/0 |
| `MessagePipeline.activity_snapshot` | 2061 | 17 | 6/6/0 |
| `MessagePipeline.consume_autonomy_delay_hint_seconds` | 2079 | 4 | 3/7/0 |
| `MessagePipeline._effective_posture_is_sitting` | 2084 | 10 | 1/1/0 |
| `MessagePipeline._agent_state` | 2095 | 21 | 4/0/0 |
| `MessagePipeline._update_mood_and_status` | 2117 | 13 | 2/0/0 |
| `MessagePipeline._clean_optional_text` | 2132 | 5 | 5/0/0 |
| `MessagePipeline._derive_status_from_actions` | 2138 | 6 | 3/0/0 |
| `MessagePipeline._status_from_command` | 2146 | 16 | 3/0/0 |
| `MessagePipeline._chat_texts_from_actions` | 2163 | 10 | 1/0/0 |
| `MessagePipeline._filter_autonomous_actions` | 2174 | 35 | 1/1/0 |
| `MessagePipeline.generate_autonomous_actions` | 2210 | 44 | 3/6/0 |
| `MessagePipeline._sanitize_autonomy_delay_hint` | 2256 | 8 | 5/1/0 |
| `MessagePipeline._normalize_autonomy_decision` | 2266 | 10 | 7/0/0 |
| `MessagePipeline._normalize_autonomy_decision_value` | 2278 | 7 | 4/0/0 |
| `MessagePipeline._apply_autonomy_scheduler_override_from_incoming` | 2286 | 17 | 1/0/0 |
| `MessagePipeline._participants_for_autonomy` | 2304 | 22 | 2/0/0 |
| `MessagePipeline._autonomy_query_text` | 2327 | 8 | 1/0/0 |
| `MessagePipeline._schedule_episode_check` | 2336 | 6 | 5/0/0 |
| `MessagePipeline._maybe_finalize_episode` | 2343 | 55 | 1/5/0 |
| `MessagePipeline._mark_episode_covered` | 2399 | 9 | 2/0/0 |
| `MessagePipeline._summarize_episode` | 2409 | 25 | 3/0/0 |
| `MessagePipeline._episode_metadata` | 2436 | 27 | 3/0/0 |
| `MessagePipeline._latest_timestamp` | 2464 | 13 | 1/0/0 |
| `MessagePipeline._entity_prompt_payload` | 2479 | 9 | 7/0/0 |
| `MessagePipeline._environment_payload` | 2489 | 14 | 1/0/0 |
| `MessagePipeline._bundle_prompt_payload` | 2504 | 77 | 2/0/0 |
| `MessagePipeline._summary_meta_for_payload` | 2583 | 27 | 1/0/0 |
| `MessagePipeline._incoming_payload_from_batch_entry` | 2612 | 21 | 1/0/0 |
| `MessagePipeline._persona_instructions` | 2634 | 5 | 3/0/0 |
| `MessagePipeline._log_reasoning_trace` | 2640 | 4 | 2/0/0 |
| `MessagePipeline._log_quality_fallbacks` | 2645 | 7 | 6/0/0 |
| `MessagePipeline._bundle_response_payload` | 2654 | 26 | 2/0/0 |
| `MessagePipeline._state_update_payload` | 2682 | 16 | 0/0/0 |
| `MessagePipeline._partial_name_tokens` | 2700 | 16 | 1/0/0 |
| `MessagePipeline._mark_mention_index_dirty` | 2717 | 2 | 4/0/0 |
| `MessagePipeline._remember_runtime_mention_name` | 2720 | 17 | 2/0/0 |
| `MessagePipeline._normalize_mention_phrase` | 2739 | 4 | 3/0/0 |
| `MessagePipeline._mention_name_variants` | 2745 | 34 | 2/0/0 |
| `MessagePipeline._refresh_mention_index_if_needed` | 2780 | 45 | 1/0/0 |
| `MessagePipeline._resolve_text_mentions` | 2826 | 39 | 1/0/0 |
| `MessagePipeline._looks_like_uuid` | 2867 | 14 | 0/0/0 |
| `MessagePipeline._merge_related_experiences` | 2883 | 28 | 3/0/0 |
| `MessagePipeline._collapse_near_duplicate_experiences` | 2912 | 44 | 3/0/0 |
| `MessagePipeline._normalize_experience_text` | 2958 | 3 | 3/0/0 |
| `MessagePipeline._near_duplicate_similarity_score` | 2963 | 13 | 3/0/0 |
| `MessagePipeline._char_trigrams` | 2978 | 4 | 2/0/0 |
| `MessagePipeline._with_near_duplicate_metadata` | 2983 | 18 | 1/0/0 |
| `MessagePipeline._iso_date` | 3003 | 2 | 2/0/0 |
| `MessagePipeline._gate_semantic_experiences` | 3006 | 33 | 1/0/1 |
| `MessagePipeline._experiences_for_persona` | 3040 | 11 | 3/0/0 |
| `MessagePipeline._build_routine_summaries` | 3052 | 54 | 1/0/0 |
| `MessagePipeline._routine_tokens` | 3108 | 16 | 1/0/0 |
| `MessagePipeline._token_jaccard` | 3126 | 7 | 5/0/0 |
| `MessagePipeline._routine_phrase` | 3135 | 13 | 1/0/0 |
| `MessagePipeline._experience_timestamp` | 3150 | 19 | 3/0/0 |
| `MessagePipeline._format_last_seen` | 3171 | 13 | 1/0/0 |
| `MessagePipeline._normalize_entities` | 3186 | 15 | 5/0/0 |

#### `src/gpt5_roleplay_system/pipeline_state.py` — 20 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|

#### `src/gpt5_roleplay_system/protocol.py` — 79 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `_normalize_data` | 19 | 12 | 1/0/0 |
| `decode_message` | 33 | 10 | 2/4/0 |
| `_now_ts` | 45 | 2 | 2/0/0 |
| `_new_id` | 49 | 2 | 1/0/0 |
| `encode_message` | 53 | 8 | 3/2/0 |
| `build_chat_response` | 63 | 17 | 2/3/0 |

#### `src/gpt5_roleplay_system/server.py` — 575 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `_format_idle_duration` | 29 | 22 | 1/9/0 |
| `StatusChannelErrorHandler.__init__` | 56 | 4 | 26/112/3 |
| `StatusChannelErrorHandler.emit` | 61 | 20 | 10/2/0 |
| `GPT5RoleplayServer.__init__` | 84 | 22 | 26/112/3 |
| `GPT5RoleplayServer.start` | 107 | 25 | 24/9/12 |
| `GPT5RoleplayServer._handle_client` | 133 | 82 | 1/2/0 |
| `GPT5RoleplayServer._enqueue_message` | 216 | 21 | 1/0/0 |
| `GPT5RoleplayServer._install_error_handler` | 238 | 12 | 1/1/0 |
| `GPT5RoleplayServer._remove_error_handler` | 251 | 7 | 1/1/0 |
| `GPT5RoleplayServer._broadcast_status_chat` | 259 | 14 | 1/0/0 |
| `run_server` | 275 | 9 | 1/0/0 |
| `main` | 286 | 13 | 1/45/12 |
| `_build_knowledge_store` | 301 | 13 | 1/0/4 |
| `_build_llm_client` | 316 | 46 | 1/0/0 |
| `_build_tracer` | 364 | 13 | 1/0/4 |
| `_build_experience_index` | 379 | 44 | 1/3/2 |
| `_compute_autonomy_delay` | 425 | 21 | 1/3/0 |
| `_autonomy_loop` | 448 | 23 | 1/0/0 |
| `_status_loop` | 473 | 45 | 2/3/0 |
| `_parse_deduped_facts_response` | 520 | 2 | 0/0/0 |
| `_facts_deduplication_loop` | 524 | 7 | 0/0/0 |
| `_build_experience_dedupe_plans` | 533 | 11 | 0/2/0 |
| `_experience_deduplication_loop` | 546 | 5 | 0/0/0 |
| `_cancel_task` | 553 | 11 | 5/0/0 |
| `_format_error_text` | 566 | 6 | 1/0/0 |

#### `src/gpt5_roleplay_system/session.py` — 125 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `ClientSession.__post_init__` | 21 | 2 | 0/0/0 |
| `ClientSession.handle_message` | 24 | 20 | 2/3/2 |
| `ClientSession.send_actions` | 45 | 10 | 5/6/4 |
| `ClientSession.send_status` | 56 | 5 | 1/4/0 |
| `ClientSession.process_queue` | 62 | 41 | 1/2/0 |
| `_parse_bool` | 105 | 8 | 1/0/0 |
| `_controller_blocks_actions_output` | 115 | 11 | 1/0/0 |

#### `src/gpt5_roleplay_system/session_state.py` — 131 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `_slug` | 13 | 3 | 2/0/0 |
| `_memory_item_from_dict` | 18 | 14 | 1/0/0 |
| `SessionStateStore.__init__` | 35 | 8 | 26/112/3 |
| `SessionStateStore.load` | 44 | 17 | 9/5/1 |
| `SessionStateStore.save` | 62 | 12 | 5/7/0 |
| `SessionStateStore._resolve_load_path` | 75 | 14 | 1/0/0 |
| `SessionStateStore._atomic_write` | 90 | 20 | 1/0/0 |
| `SessionStateStore._load_items` | 112 | 8 | 3/0/0 |
| `SessionStateStore._dump_items` | 122 | 10 | 3/0/0 |

#### `src/gpt5_roleplay_system/time_utils.py` — 22 lines

| Function | Line | Span | Refs (src/tests/scripts) |
|---|---|---|---|
| `get_pacific_time` | 4 | 7 | 5/0/0 |
| `format_pacific_time` | 12 | 4 | 54/0/0 |
| `get_pacific_timestamp` | 17 | 6 | 7/0/0 |
