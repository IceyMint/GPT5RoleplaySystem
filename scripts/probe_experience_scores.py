#!/usr/bin/env python3
"""Measure real semantic-experience scores and what the gate does to them.

Embedding models differ widely in how they normalise cosine similarity, so the
right `experience_score_min` cannot be guessed -- it has to be measured against
the actual corpus. This probe reports the live score distribution and simulates
alternative gate settings before you change config.

Read-only: it only runs vector searches.

Usage:
    PYTHONPATH=src python scripts/probe_experience_scores.py [persona_id] [query ...]

If no persona is given it probes the persona with the most stored experiences.
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpt5_roleplay_system.config_loader import ConfigLoader  # noqa: E402
from gpt5_roleplay_system.neo4j_store import Neo4jKnowledgeStore  # noqa: E402
from gpt5_roleplay_system.server import _build_experience_index  # noqa: E402

DEFAULT_QUERIES = [
    "time for bed",
    "goodnight, sleepy",
    "are you hungry?",
    "what did we do yesterday?",
    "let's go outside and play",
    "how are you feeling today?",
]


def simulate_gate(scores: list[float], score_min: float, delta: float, top_k: int) -> int:
    """Mirror MessagePipeline._gate_semantic_experiences."""
    if not scores:
        return 0
    ordered = sorted(scores, reverse=True)
    top = ordered[0]
    if top < score_min:
        return 0
    kept = 0
    for score in ordered:
        if score < score_min or (top - score) > delta:
            continue
        kept += 1
        if kept >= top_k:
            break
    return kept


async def main() -> int:
    config = ConfigLoader().load("config.yaml")
    store = Neo4jKnowledgeStore(
        uri=config.neo4j.uri,
        user=config.neo4j.user,
        password=config.neo4j.password,
        database=config.neo4j.database,
    )
    index = _build_experience_index(config, store)
    if index is None or not index.is_enabled():
        print("Semantic experience index is DISABLED -- nothing to measure.")
        print("Check the GenAI/embedding provider (billing, API key, plugin).")
        return 1

    args = sys.argv[1:]
    persona = args[0] if args else ""
    queries = args[1:] or DEFAULT_QUERIES
    if not persona:
        with store.driver.session(database=store.database) as session:
            row = session.run(
                "MATCH (p:Persona)-[:HAD_EXPERIENCE]->(e:Experience) "
                "RETURN p.id AS id, count(e) AS c ORDER BY c DESC LIMIT 1"
            ).single()
        persona = str(row["id"]) if row else ""
        if not persona:
            print("No Persona nodes with experiences found.")
            return 1

    score_min = config.knowledge.experience_score_min
    delta = config.knowledge.experience_score_delta
    top_k = config.knowledge.experience_similar_limit
    print(f"persona={persona!r}  score_min={score_min}  delta={delta}  top_k={top_k}\n")

    header = f"{'query':34s} {'top':>6} {'current':>8} {'.65/.15':>8} {'.60/.20':>8}"
    print(header)
    print("-" * len(header))

    all_scores: list[float] = []
    for query in queries:
        results = await index.search(query, persona, top_k=12)
        scores: list[float] = []
        for record in results:
            try:
                scores.append(float(record.metadata.get("score")))
            except (TypeError, ValueError):
                continue
        all_scores.extend(scores)
        top = f"{max(scores):.3f}" if scores else "-"
        print(
            f"{query[:34]:34s} {top:>6} "
            f"{simulate_gate(scores, score_min, delta, top_k):>8} "
            f"{simulate_gate(scores, 0.65, 0.15, top_k):>8} "
            f"{simulate_gate(scores, 0.60, 0.20, top_k):>8}"
        )

    print(f"\n=== score distribution ({len(all_scores)} results) ===")
    if all_scores:
        ordered = sorted(all_scores, reverse=True)
        print(
            f"  max={ordered[0]:.3f}  median={ordered[len(ordered) // 2]:.3f}  min={ordered[-1]:.3f}"
        )
        buckets = Counter(round(s, 1) for s in ordered)
        for bucket in sorted(buckets, reverse=True):
            print(f"  {bucket:.1f}  {'#' * buckets[bucket]} ({buckets[bucket]})")
        blocked = sum(1 for s in all_scores if s < score_min)
        print(f"\n  blocked by current score_min={score_min}: {blocked}/{len(all_scores)}")
        if blocked == len(all_scores):
            print("  WARNING: the gate is rejecting every result -- memory recall is off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
