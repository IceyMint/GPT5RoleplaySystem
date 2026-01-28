# GPT5RoleplaySystem Rewrite TODOs (Prioritized)

Top-risk items first. Check off as completed.

## P0 — Safety & correctness
- [x] **Prevent AI self-reply loops.** Filter messages authored by the AI persona (from_name/agent) and de-duplicate against recent AI outputs. Seed AI-typed chat into memory without generating a response.
- [x] **Add robust self-chat detection inputs.** Ensure the pipeline has access to the AI persona name and logged-in agent name; normalize both and use exact/contains matching with a safe fallback.

## P1 — Addressing & context quality
- [x] **Spatial addressee detection.** Include avatar position, nearby agents/objects, and environment context in `is_addressed_to_me` so the model can decide who is being spoken to in crowded scenes.
- [x] **Participant extraction improvements.** Use `target_uuid`/`uuid` normalization everywhere and filter the AI persona out of participant lists.

## P2 — Memory & knowledge retention
- [x] **Experience similarity retrieval.** Restore semantic experience search (top‑k relevant prior experiences) and include it in the context payload.
- [x] **Knowledge extraction to Neo4j.** Batch store facts/relationships from conversations into Neo4j; ensure it runs async and doesn’t block chat response latency.
- [x] **Memory compression / summarization.** Add LLM-backed summary updates for overflowed context beyond the rolling window.

## P3 — Autonomy & status
- [x] **Autonomous behavior loop.** Schedule autonomous actions with backoff and activity-aware delays.
- [x] **Status updates.** Emit status/mood updates to the viewer for UI display.

## P4 — Observability & testing
- [x] **LLM prompt/response logging.** Capture structured inputs/outputs for debugging (no secrets).
- [ ] **Perf metrics.** Add per-message latency metrics and periodic summaries.
- [ ] **Scenario tests.** Add tests for self‑chat filtering, multi‑message batching, and participant extraction.
