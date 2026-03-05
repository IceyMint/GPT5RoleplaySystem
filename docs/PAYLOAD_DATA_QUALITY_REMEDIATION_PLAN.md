# Context Payload Data-Quality Remediation Plan

## Summary
This plan fixes identity, participant, time-range, incoming-event, and state-reconciliation issues in LLM context payloads without changing the external NDJSON wire protocol.  
Target plan document path (for implementation mode): `docs/PAYLOAD_DATA_QUALITY_REMEDIATION_PLAN.md`.

Primary outcomes:
1. Eliminate duplicate/fragmented identities in participants.
2. Guarantee participant coverage for all active non-self speakers.
3. Enforce deterministic payload invariants for `incoming`, `incoming_batch`, `recent_time_range`, and `summary_meta`.
4. Use viewer-authoritative posture state from `environment_update.is_sitting` with freshness-aware unknown handling.
5. Keep rollout safe with soft-heal + warnings/metrics (no hard failures initially).

## User Decisions Locked
1. Participant scope: **All active speakers**.
2. Validation mode: **Soft-heal + log**.
3. Self mapping: **`sender = persona`, `sender_id = uuid`**.

## Public Interfaces / Contract Changes
1. No external protocol message-type changes in `src/gpt5_roleplay_system/protocol.py`.
2. Internal prompt payload contract in `src/gpt5_roleplay_system/llm.py` and tracer bundle payload in `src/gpt5_roleplay_system/pipeline.py` will enforce:
- `participants` unique by canonical identity.
- `participants` includes all active non-self speakers from recent + incoming windows.
- Self-authored message payloads use `sender=<persona>`, `sender_id=<canonical self uuid when available>`.
- `incoming` equals the deterministic latest event from `incoming_batch` when batch exists.
- `summary` explicitly treated as pre-recent-window context; `summary_meta` range consistency enforced.
- `environment.is_sitting` remains viewer-authoritative and backward-compatible as last-known state.
3. `ConversationMemory` compression behavior in `src/gpt5_roleplay_system/memory.py` changes to avoid unstable split semantics at equal timestamps (timestamp-boundary aware slicing).
4. New internal validator/helper module added: `src/gpt5_roleplay_system/payload_contract.py` (normalization, invariant checks, soft repairs, warning emission).

## Detailed Implementation Plan

## 1) Canonical Identity and Participant Normalization
Files:
- `src/gpt5_roleplay_system/pipeline.py`
- `src/gpt5_roleplay_system/context_builder.py`
- `src/gpt5_roleplay_system/payload_contract.py` (new)

Changes:
1. Add canonical key helpers:
- `id:<uuid>` when UUID exists.
- `name:<normalized_name>` when UUID missing.
2. Merge name-only and id-backed duplicates when name match is unambiguous:
- If one name-only entry matches exactly one id-backed entry by `name_matches`, merge into id-backed record.
- If ambiguous, keep both and emit warning.
3. Deduplicate participants at two layers:
- Upstream (participant assembly).
- Final payload layer (safety net).
4. Preserve best available naming fields on merge:
- Prefer richer full name (`Display (username)`).
- Keep username/display consistently derived via `split_display_and_username`.

Acceptance invariant:
- Never emit both `{user_id:"", name:"Hailykitty"}` and `{user_id:"uuid", name:"Hailykitty"}` in one payload.

## 2) Participant Coverage = All Active Non-Self Speakers
Files:
- `src/gpt5_roleplay_system/pipeline.py`
- `src/gpt5_roleplay_system/context_builder.py`
- `src/gpt5_roleplay_system/llm.py`

Changes:
1. Define active speaker set as union of non-self senders from:
- `context.recent_messages`
- `overflow` (if present)
- current batch (`incoming_batch` / current non-self chats)
2. Build `participants` as:
- existing resolved participants + active speakers (then canonical dedupe).
3. Keep environment agents as optional hints only; do not rely on them for active-speaker inclusion.

Acceptance invariant:
- Every non-self sender_id present in recent/incoming windows appears in participants (unless sender_id+name both empty).

## 3) Self Identity Unification (Persona-first display, UUID sender_id)
Files:
- `src/gpt5_roleplay_system/pipeline.py`
- `src/gpt5_roleplay_system/controller.py` (only if needed for auto-promotion helper)
- `docs/VIEWER_INTEGRATION.md`

Changes:
1. Add canonical self-id resolver:
- Prefer session `user_id` when it looks UUID-like.
- If current `user_id` is placeholder and a confirmed self message contains UUID-like `from_id`, promote session self ID.
2. Self payload output rule:
- `sender` always persona string.
- `sender_id` canonical self UUID when available.
- If UUID unavailable, keep existing id but emit warning metric (soft mode).
3. Document strong requirement that viewer should send `set_user_id` early.

Acceptance invariant:
- Self messages are never emitted as `sender="default_user"`; sender uses persona.
- `sender_id` is UUID when known.

## 4) Summary Contract and Summary-Range Integrity
Files:
- `src/gpt5_roleplay_system/context_builder.py`
- `src/gpt5_roleplay_system/pipeline.py`
- `docs/MEMORY_AND_EPISODES.md` (or `docs/ARCHITECTURE.md`)

Changes:
1. Keep existing `summary` field name for compatibility, but define semantics as prior-context summary (pre-recent window).
2. Enforce metadata integrity at payload build:
- `range_start_ts <= range_end_ts`
- if recent window has start timestamp, enforce `range_end_ts <= recent_start_ts` (clamp with warning in soft mode)
3. Compute `age_seconds` and `range_age_seconds` from one shared `now_ts` snapshot.

Acceptance invariant:
- No payload where `summary_meta.range_end_ts > recent_time_range.start`.

## 5) Deterministic Window Slicing and Overflow Boundary
Files:
- `src/gpt5_roleplay_system/memory.py`

Changes:
1. Update `_compress_if_needed` to timestamp-boundary-aware slicing:
- Prevent arbitrary split of a tied timestamp group across `recent` and `overflow` when avoidable.
2. Keep ordering deterministic and stable.
3. Preserve existing max window behavior as closely as possible; allow slight temporary overage only for tie preservation.

Acceptance invariant:
- Repeated runs with same input order produce identical `recent_messages` and `overflow_messages` boundary behavior.

## 6) Incoming Batch Consistency and Deterministic Latest Selection
Files:
- `src/gpt5_roleplay_system/pipeline.py`
- `src/gpt5_roleplay_system/llm.py`

Changes:
1. Make `incoming_batch` ordering deterministic:
- Sort by `(last_timestamp, arrival_order, sender_key)`.
2. Define deterministic incoming event:
- Latest by `(timestamp, arrival_order)` among current non-self chats.
3. Enforce payload consistency:
- When `incoming_batch` non-empty, `incoming` must correspond to final `incoming_batch` element’s latest event semantics.
- Repair mismatch in soft mode and emit warning.

Acceptance invariant:
- `incoming` and `incoming_batch[-1]` agree on sender + latest text semantics.

## 7) Viewer-Authoritative Sit/Stand State
Files:
- `src/gpt5_roleplay_system/pipeline.py`
- `src/gpt5_roleplay_system/context_builder.py`
- `src/gpt5_roleplay_system/pipeline_state.py`
- `src/gpt5_roleplay_system/config.py`
- `config.yaml`

Changes:
1. Treat viewer `environment_update.is_sitting` as the only source of posture truth.
2. Do not parse chat/emote/system text for posture inference.
3. Track `last_environment_update_ts` in runtime state and ignore out-of-order posture updates.
4. Add `posture_stale_seconds` config (default `6.0`) and mark posture unknown after timeout.
5. Keep `environment.is_sitting` as last-known value for backward compatibility.
6. Keep prompt contract unchanged for now (no new posture payload fields).

Acceptance invariant:
- Posture state used by prompting is derived only from viewer updates and freshness policy.

## 8) Soft-Heal Validator + Observability
Files:
- `src/gpt5_roleplay_system/payload_contract.py` (new)
- `src/gpt5_roleplay_system/pipeline.py`
- `src/gpt5_roleplay_system/llm.py`
- `docs/OBSERVABILITY.md`

Changes:
1. Add `normalize_and_validate_payload(payload) -> (payload, warnings)` used before prompt serialization/logging.
2. Warning categories:
- `identity_merge`
- `participant_coverage`
- `summary_range_clamp`
- `incoming_repair`
- `self_id_missing_uuid`
3. Emit structured tracer event (e.g., `payload_repair`) with counts and details.

Acceptance invariant:
- Invalid but repairable payloads still proceed; repairs are visible in telemetry.

## Test Plan and Scenarios

## Unit Tests (new/updated)
Files:
- `tests/test_pipeline.py`
- `tests/test_memory.py`
- `tests/test_llm.py`

Add tests:
1. Duplicate participant merge:
- name-only + id-backed same person emits one participant.
2. Active speaker coverage:
- all non-self senders in recent/incoming windows are present.
3. Self mapping:
- self payload emits `sender=persona`; `sender_id=uuid` when available.
4. Summary range integrity:
- clamp/repair path when `range_end_ts` exceeds recent start.
5. Incoming consistency:
- multiple same-second messages from different senders produce deterministic `incoming` and batch ordering.
6. Overflow tie boundary:
- equal-timestamp messages don’t produce unstable split behavior across runs.
7. Posture freshness:
- stale posture (`> posture_stale_seconds`) is treated as unknown for internal decisioning.
8. Out-of-order posture update handling:
- older `environment_update` posture timestamps do not override newer posture state.
9. Soft-heal warnings:
- payload repairs emit expected warning categories.

## Regression Fixtures
1. Convert the three provided payload patterns into deterministic fixture-driven tests.
2. Assert corrected payload shape against invariant checks.

## Acceptance Criteria
1. All existing tests pass.
2. New tests above pass.
3. No duplicate participants in emitted payloads for sampled traces.
4. No `incoming`/`incoming_batch` mismatch in sampled traces.
5. `summary_meta` and `recent_time_range` invariants hold in sampled traces.
6. `payload_repair` telemetry confirms repairs are rare and trending down after rollout.

## Rollout Plan
1. Phase 1:
- Ship soft-heal + warnings, no hard failures.
- Monitor `payload_repair` volume by category.
2. Phase 2:
- If repair rate is low and stable for 7 days, optionally upgrade selected checks to strict mode behind config flag.
3. Phase 3:
- Document stable contract in `docs/PROTOCOL.md` and prompt-format docs.

## Assumptions and Defaults
1. `summary` remains prior-context summary (not current-window summary).
2. No breaking changes to NDJSON message schema.
3. Viewer eventually provides stable UUID via `set_user_id`; until then, fallback behavior is logged.
4. All repairs are non-destructive in this phase (soft-heal mode).
5. Viewer posture updates are expected approximately every 2 seconds.
6. If viewer timestamp is missing, server receive time is used for freshness tracking.
