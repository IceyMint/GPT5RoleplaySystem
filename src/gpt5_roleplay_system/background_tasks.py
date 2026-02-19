from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .config import ServerConfig
from .deduplication import ExperienceDedupePlanner, parse_deduped_facts_response
from .llm import OpenRouterLLMClient
from .neo4j_experience_vector import Neo4jVectorConfig
from .neo4j_store import Neo4jKnowledgeStore

logger = logging.getLogger("gpt5_roleplay_server")


class FactsDeduplicationService:
    def __init__(
        self,
        config: ServerConfig,
        knowledge_store: Neo4jKnowledgeStore,
        llm_client: OpenRouterLLMClient,
    ) -> None:
        self._config = config
        self._knowledge_store = knowledge_store
        self._llm_client = llm_client
        self._runtime_key = "facts_deduplication"
        self._dedupe_model = str(getattr(llm_client, "_facts_model", getattr(llm_client, "_model", "")) or "")

    async def run_forever(self) -> None:
        interval_seconds = float(self._config.facts_deduplication.interval_hours) * 3600.0
        if interval_seconds <= 0.0:
            return
        self._knowledge_store.init_runtime_marker(self._runtime_key, time.time())
        while True:
            await asyncio.sleep(interval_seconds)
            await self.run_once()

    async def run_once(self) -> dict[str, Any]:
        run_started_ts = time.time()
        stats = {
            "candidate_count": 0,
            "processed_count": 0,
            "changed_count": 0,
            "unchanged_count": 0,
            "failed_count": 0,
            "started_ts": run_started_ts,
        }
        self._knowledge_store.mark_runtime_running(self._runtime_key, run_started_ts)
        try:
            people = self._knowledge_store.fetch_facts_dedupe_candidates()
            stats["candidate_count"] = len(people)
            for person in people:
                user_id = str(person.get("user_id", "") or "")
                name = str(person.get("name", "") or "")
                facts = [item for item in (person.get("facts", []) or []) if isinstance(item, str) and item.strip()]

                prompt = (
                    "You are a data cleaning expert. Refine these facts about a person, removing redundancy and duplicates. "
                    "Combine similar facts. Output a JSON object with key 'facts' containing a list of strings.\n\n"
                    f"Person Name: {name}\nCurrent Facts: {facts}"
                )

                response_text = ""
                try:
                    response_text = await asyncio.to_thread(
                        self._llm_client._request_text_with_model,
                        self._dedupe_model,
                        "You are a precision data cleaner.",
                        prompt,
                        self._llm_client._max_tokens,
                        0.0,
                        include_reasoning=False,
                        include_provider=True,
                    )
                    try:
                        refined_facts = parse_deduped_facts_response(response_text)
                    except Exception:
                        retry_prompt = (
                            f"{prompt}\n\n"
                            "Return ONLY valid minified JSON with exactly one key named \"facts\"."
                        )
                        response_text = await asyncio.to_thread(
                            self._llm_client._request_text_with_model,
                            self._dedupe_model,
                            "You are a precision data cleaner.",
                            retry_prompt,
                            self._llm_client._max_tokens,
                            0.0,
                            include_reasoning=False,
                            include_provider=True,
                        )
                        refined_facts = parse_deduped_facts_response(response_text)

                    deduped_ts = time.time()
                    if refined_facts and refined_facts != facts:
                        self._knowledge_store.apply_deduped_facts(user_id, refined_facts, deduped_ts=deduped_ts)
                        stats["changed_count"] += 1
                    else:
                        self._knowledge_store.mark_facts_deduped(user_id, deduped_ts=deduped_ts)
                        stats["unchanged_count"] += 1
                    stats["processed_count"] += 1
                except Exception as exc:
                    stats["failed_count"] += 1
                    preview = (response_text or "").replace("\n", " ").strip()
                    if len(preview) > 220:
                        preview = preview[:220] + "..."
                    logger.error(
                        "Failed to deduplicate facts for %s (%s): %s | response_preview=%r",
                        name,
                        user_id,
                        exc,
                        preview,
                    )
                    continue

            payload = {
                "last_completed_ts": time.time(),
                "last_candidate_count": int(stats["candidate_count"]),
                "last_processed_count": int(stats["processed_count"]),
                "last_changed_count": int(stats["changed_count"]),
                "last_unchanged_count": int(stats["unchanged_count"]),
                "last_failed_count": int(stats["failed_count"]),
                "last_error": "",
            }
            self._knowledge_store.mark_runtime_success(self._runtime_key, payload)
            return stats
        except Exception as exc:
            payload = {
                "last_failed_ts": time.time(),
                "last_candidate_count": int(stats["candidate_count"]),
                "last_processed_count": int(stats["processed_count"]),
                "last_changed_count": int(stats["changed_count"]),
                "last_unchanged_count": int(stats["unchanged_count"]),
                "last_failed_count": int(stats["failed_count"]),
                "last_error": str(exc),
            }
            self._knowledge_store.mark_runtime_error(self._runtime_key, payload)
            logger.error("Error in facts deduplication loop: %s", exc)
            return stats


class ExperienceDeduplicationService:
    def __init__(
        self,
        config: ServerConfig,
        knowledge_store: Neo4jKnowledgeStore,
    ) -> None:
        self._config = config
        self._knowledge_store = knowledge_store
        self._runtime_key = "experience_deduplication"

    async def run_forever(self) -> None:
        interval_seconds = float(self._config.experience_deduplication.interval_hours) * 3600.0
        if interval_seconds <= 0.0:
            return
        self._knowledge_store.init_runtime_marker(self._runtime_key, time.time())
        while True:
            await asyncio.sleep(interval_seconds)
            await self.run_once()

    async def run_once(self) -> dict[str, Any]:
        dedupe_config = self._config.experience_deduplication
        similarity_threshold = min(1.0, max(0.0, float(dedupe_config.similarity_threshold)))
        max_gap_seconds = max(0.0, float(dedupe_config.max_time_gap_hours) * 3600.0)
        neighbor_k = max(2, int(dedupe_config.neighbor_k))
        dry_run = bool(dedupe_config.dry_run)
        index_name = Neo4jVectorConfig().index_name

        run_started_ts = time.time()
        stats = {
            "candidate_pair_count": 0,
            "qualifying_pair_count": 0,
            "planned_group_count": 0,
            "merged_node_count": 0,
            "failed_count": 0,
            "started_ts": run_started_ts,
        }
        self._knowledge_store.mark_runtime_running(self._runtime_key, run_started_ts)
        try:
            candidate_rows = self._knowledge_store.fetch_experience_dedupe_candidates(
                index_name=index_name,
                neighbor_k=neighbor_k,
                score_floor=similarity_threshold,
            )
            stats["candidate_pair_count"] = len(candidate_rows)
            plans, qualifying_pair_count = ExperienceDedupePlanner.build_plans(
                candidate_rows,
                similarity_threshold=similarity_threshold,
                max_gap_seconds=max_gap_seconds,
            )
            stats["qualifying_pair_count"] = qualifying_pair_count
            stats["planned_group_count"] = len(plans)

            if dry_run:
                stats["merged_node_count"] = sum(len(plan.get("dup_ids", [])) for plan in plans)
            else:
                for plan in plans:
                    keep_id = str(plan.get("keep_id", "") or "")
                    dup_ids = [str(item) for item in plan.get("dup_ids", []) if str(item)]
                    if not keep_id or not dup_ids:
                        continue
                    try:
                        deleted_count = self._knowledge_store.merge_experience_group(
                            keep_id=keep_id,
                            dup_ids=dup_ids,
                            merged_timestamp=float(plan.get("merged_timestamp", 0.0) or 0.0),
                            merged_timestamp_start=str(plan.get("merged_timestamp_start", "") or ""),
                            merged_timestamp_end=str(plan.get("merged_timestamp_end", "") or ""),
                            deduped_ts=time.time(),
                        )
                        stats["merged_node_count"] += max(0, int(deleted_count))
                    except Exception as plan_exc:
                        stats["failed_count"] += 1
                        logger.warning(
                            "Experience dedupe merge failed for keep=%s dup_ids=%s: %s",
                            keep_id,
                            dup_ids,
                            plan_exc,
                        )

            payload = {
                "last_completed_ts": time.time(),
                "last_candidate_pair_count": int(stats["candidate_pair_count"]),
                "last_qualifying_pair_count": int(stats["qualifying_pair_count"]),
                "last_plan_count": int(stats["planned_group_count"]),
                "last_merged_node_count": int(stats["merged_node_count"]),
                "last_failed_count": int(stats["failed_count"]),
                "last_dry_run": bool(dry_run),
                "last_error": "",
                "status": "dry_run" if dry_run else "success",
            }
            self._knowledge_store.mark_runtime_success(self._runtime_key, payload)
            return stats
        except Exception as exc:
            payload = {
                "last_failed_ts": time.time(),
                "last_candidate_pair_count": int(stats["candidate_pair_count"]),
                "last_qualifying_pair_count": int(stats["qualifying_pair_count"]),
                "last_plan_count": int(stats["planned_group_count"]),
                "last_merged_node_count": int(stats["merged_node_count"]),
                "last_failed_count": int(stats["failed_count"]),
                "last_dry_run": bool(dry_run),
                "last_error": str(exc),
            }
            self._knowledge_store.mark_runtime_error(self._runtime_key, payload)
            logger.error("Error in experience deduplication loop: %s", exc)
            return stats
