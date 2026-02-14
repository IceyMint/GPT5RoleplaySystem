from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import re
import time
from typing import Dict, Optional

from .config import AutonomyConfig, ServerConfig, load_config
from .controller import SessionController
from .experience_vector import ExperienceVectorIndex, NullEmbeddingClient, OpenAIEmbeddingClient
from .llm import OpenRouterLLMClient, RuleBasedLLMClient, log_prompt_cache_summary
from .models import Action, CommandType
from .neo4j_experience_vector import Neo4jExperienceVectorIndex, Neo4jVectorConfig
from .neo4j_store import InMemoryKnowledgeStore, Neo4jKnowledgeStore, KnowledgeStore
from .observability import NoOpTracer, WandbTracer
from .protocol import decode_message
from .session import ClientSession
from .time_utils import get_pacific_timestamp


logger = logging.getLogger("gpt5_roleplay_server")


class StatusChannelErrorHandler(logging.Handler):
    """Broadcast ERROR logs to the configured status chat channel."""

    def __init__(self, server: "GPT5RoleplayServer", channel: int, level: int = logging.ERROR) -> None:
        super().__init__(level=level)
        self._server = server
        self._channel = int(channel)

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - thin wrapper
        loop = self._server._loop
        if loop is None or loop.is_closed():
            return
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        text = _format_error_text(record, message)

        def _schedule() -> None:
            asyncio.create_task(self._server._broadcast_status_chat(text, channel=self._channel, source="error"))

        loop.call_soon_threadsafe(_schedule)


class GPT5RoleplayServer:
    def __init__(
        self,
        host: str,
        port: int,
        config: ServerConfig,
        knowledge_store: KnowledgeStore,
        llm_client,
        tracer,
    ) -> None:
        self._host = host
        self._port = port
        self._config = config
        self._knowledge_store = knowledge_store
        self._llm_client = llm_client
        self._tracer = tracer
        self._sessions: Dict[str, ClientSession] = {}
        self._counter = 0
        self._server: asyncio.AbstractServer | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._error_handler: StatusChannelErrorHandler | None = None
        self._dedup_task: asyncio.Task | None = None

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._install_error_handler()
        if self._config.facts_deduplication.enabled:
            self._dedup_task = asyncio.create_task(
                _facts_deduplication_loop(self._config, self._knowledge_store, self._llm_client)
            )
        self._server = await asyncio.start_server(self._handle_client, self._host, self._port)
        addr = ", ".join(str(sock.getsockname()) for sock in self._server.sockets or [])
        logger.info("Server listening on %s", addr)
        try:
            async with self._server:
                await self._server.serve_forever()
        finally:
            await _cancel_task(self._dedup_task, session_id="server", label="facts_deduplication")
            self._dedup_task = None
            self._remove_error_handler()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._counter += 1
        session_id = f"session_{self._counter}"
        experience_vector_index = _build_experience_index(self._config, self._knowledge_store)
        controller = SessionController(
            persona=self._config.persona,
            user_id=self._config.user_id,
            knowledge_store=self._knowledge_store,
            llm=self._llm_client,
            tracer=self._tracer,
            experience_vector_index=experience_vector_index,
            max_recent_messages=self._config.memory.max_recent_messages,
            max_rolling_buffer=self._config.memory.max_rolling_buffer,
            summary_strategy=self._config.memory.summary_strategy,
            experience_top_k=self._config.knowledge.experience_similar_limit,
            experience_score_min=self._config.knowledge.experience_score_min,
            experience_score_delta=self._config.knowledge.experience_score_delta,
            max_environment_participants=self._config.max_environment_participants,
            facts_config=self._config.facts,
            episode_config=self._config.episode,
            persona_profiles=self._config.persona_profiles,
        )
        # Start each new connection with chat output disabled until explicitly enabled.
        controller.set_llm_chat_enabled(False)
        session = ClientSession(
            session_id=session_id,
            controller=controller,
            writer=writer,
            batch_window_seconds=self._config.chat_batch_window_ms / 1000.0,
            batch_max_size=self._config.chat_batch_max,
        )
        self._sessions[session_id] = session
        peer = writer.get_extra_info("peername")
        logger.info("Client connected: %s (%s)", session_id, peer)

        queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue(maxsize=self._config.queue_max_size)
        worker = asyncio.create_task(session.process_queue(queue))
        autonomy_task = None
        status_task = None
        if self._config.autonomy.enabled:
            autonomy_task = asyncio.create_task(_autonomy_loop(session, queue, self._config.autonomy))
            status_task = asyncio.create_task(_status_loop(session, self._config.autonomy))

        try:
            while True:
                if worker.done():
                    with contextlib.suppress(asyncio.CancelledError):
                        worker_error = worker.exception()
                        if worker_error and not isinstance(worker_error, ConnectionError):
                            logger.warning("Session worker failed for %s: %s", session_id, worker_error)
                    break
                if reader.at_eof():
                    break
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if not line:
                    break
                try:
                    message = decode_message(line.decode("utf-8"))
                except Exception as exc:
                    logger.warning("Decode error from %s: %s", session_id, exc)
                    continue
                await self._enqueue_message(queue, session_id, message.msg_type, message.data)
        finally:
            await _cancel_task(worker, session_id=session_id, label="worker")
            await _cancel_task(autonomy_task, session_id=session_id, label="autonomy")
            await _cancel_task(status_task, session_id=session_id, label="status")
            with contextlib.suppress(Exception):
                await session.controller.flush_state()
            writer.close()
            await writer.wait_closed()
            self._sessions.pop(session_id, None)
            logger.info("Client disconnected: %s", session_id)

    async def _enqueue_message(
        self,
        queue: asyncio.Queue[tuple[str, dict]],
        session_id: str,
        msg_type: str,
        data: dict,
    ) -> None:
        item = (msg_type, data)
        if not queue.full():
            await queue.put(item)
            return
        if self._config.queue_drop_policy == "drop_newest":
            logger.warning("Dropped incoming message for %s: %s", session_id, msg_type)
            return
        try:
            dropped = queue.get_nowait()
            queue.task_done()
            logger.warning("Dropped queued message for %s: %s", session_id, dropped[0])
        except asyncio.QueueEmpty:
            pass
        await queue.put(item)

    def _install_error_handler(self) -> None:
        channel = self._config.autonomy.status_channel
        if channel is None:
            return
        root = logging.getLogger()
        if self._error_handler is not None:
            root.removeHandler(self._error_handler)
        handler = StatusChannelErrorHandler(self, channel=channel)
        handler.setLevel(logging.ERROR)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
        self._error_handler = handler

    def _remove_error_handler(self) -> None:
        handler = self._error_handler
        if handler is None:
            return
        root = logging.getLogger()
        root.removeHandler(handler)
        self._error_handler = None

    async def _broadcast_status_chat(self, text: str, *, channel: int, source: str) -> None:
        if not text or not self._sessions:
            return
        action = Action(
            command_type=CommandType.CHAT,
            content=text,
            parameters={"channel": int(channel), "source": source},
        )
        for session in list(self._sessions.values()):
            try:
                await session.send_actions([action])
            except Exception:
                # Avoid logging here to prevent handler recursion.
                continue


async def run_server(host: str, port: int, config_path: Optional[str] = None) -> None:
    config = load_config(config_path)
    config.host = host
    config.port = port
    knowledge_store = _build_knowledge_store(config)
    llm_client = _build_llm_client(config)
    tracer = _build_tracer(config)
    server = GPT5RoleplayServer(host, port, config, knowledge_store, llm_client, tracer)
    await server.start()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="GPT5 Roleplay multi-client server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    try:
        asyncio.run(run_server(args.host, args.port, args.config))
    except KeyboardInterrupt:
        logger.info("Shutdown requested via Ctrl+C.")
    finally:
        log_prompt_cache_summary()


def _build_knowledge_store(config: ServerConfig) -> KnowledgeStore:
    if config.neo4j.uri and config.neo4j.user and config.neo4j.password:
        try:
            return Neo4jKnowledgeStore(
                config.neo4j.uri,
                config.neo4j.user,
                config.neo4j.password,
                database=config.neo4j.database,
            )
        except RuntimeError as exc:
            logger.warning("Neo4j unavailable (%s); falling back to in-memory knowledge store", exc)
            return InMemoryKnowledgeStore()
    return InMemoryKnowledgeStore()


def _build_llm_client(config: ServerConfig):
    if config.llm.api_key:
        try:
            logger.info(
                "LLM: OpenRouter enabled (model=%s, address_model=%s, base_url=%s)",
                config.llm.model,
                config.llm.address_model or config.llm.model,
                config.llm.base_url,
            )
            return OpenRouterLLMClient(
                api_key=config.llm.api_key,
                base_url=config.llm.base_url,
                model=config.llm.model,
                address_model=config.llm.address_model,
                max_tokens=config.llm.max_tokens,
                temperature=config.llm.temperature,
                timeout_seconds=config.llm.timeout_seconds,
                facts_in_bundle=config.facts.in_bundle,
                fallback=RuleBasedLLMClient(),
                reasoning=config.llm.reasoning,
            )
        except RuntimeError as exc:
            logger.warning("LLM: OpenRouter client unavailable (%s); using rule-based fallback", exc)
            return RuleBasedLLMClient()
    logger.warning(
        "LLM: No OpenRouter API key found; using rule-based fallback. "
        "Pass --config /abs/path/config.yaml or set GPT5_ROLEPLAY_CONFIG / OPENROUTER_API_KEY."
    )
    return RuleBasedLLMClient()


def _build_tracer(config: ServerConfig):
    if not config.wandb.enabled:
        logger.info("Observability: W&B disabled; using no-op tracer")
        return NoOpTracer()
    if config.wandb.api_key:
        os.environ.setdefault("WANDB_API_KEY", config.wandb.api_key)
    try:
        tracer = WandbTracer(project=config.wandb.project)
        tracer.start_run(config.wandb.run_name or "gpt5-roleplay")
        return tracer
    except RuntimeError as exc:
        logger.warning("Observability: W&B unavailable (%s); using no-op tracer", exc)
        return NoOpTracer()


def _build_experience_index(config: ServerConfig, knowledge_store: KnowledgeStore):
    embedding_model = config.llm.embedding_model
    embedding_api_key = config.llm.embedding_api_key or config.llm.api_key
    if not embedding_model or not embedding_api_key:
        return None
    try:
        embedder = OpenAIEmbeddingClient(
            api_key=embedding_api_key,
            base_url=config.llm.embedding_base_url,
            model=embedding_model,
            timeout_seconds=config.llm.timeout_seconds,
        )
    except RuntimeError:
        embedder = NullEmbeddingClient()
    if isinstance(knowledge_store, Neo4jKnowledgeStore):
        vector_config = Neo4jVectorConfig(dimensions=config.llm.embedding_dimensions)
        index = Neo4jExperienceVectorIndex(
            driver=knowledge_store.driver,
            database=knowledge_store.database,
            config=vector_config,
            token=embedding_api_key,
            model=embedding_model,
            external_embedder=embedder,
        )
        return index if index.is_enabled() else None
    index = ExperienceVectorIndex(embedder=embedder)
    return index if index.is_enabled() else None


def _compute_autonomy_delay(snapshot: Dict[str, float], autonomy: AutonomyConfig) -> float:
    seconds_since_activity = float(snapshot.get("seconds_since_activity", 0.0) or 0.0)
    delay = float(autonomy.base_delay_seconds)
    if seconds_since_activity < autonomy.recent_activity_window_seconds:
        delay *= autonomy.recent_activity_multiplier
    delay = max(delay, autonomy.min_delay_seconds)
    delay = min(delay, autonomy.max_delay_seconds)
    return delay


async def _autonomy_loop(
    session: ClientSession,
    queue: asyncio.Queue[tuple[str, dict]],
    autonomy: AutonomyConfig,
) -> None:
    window = autonomy.recent_activity_window_seconds
    while True:
        snapshot = session.controller.activity_snapshot(window)
        delay = _compute_autonomy_delay(snapshot, autonomy)
        await asyncio.sleep(delay)
        if queue.full():
            logger.warning("Dropped autonomous tick for %s: queue full", session.session_id)
            continue
        queue.put_nowait(("autonomous_tick", {"recent_activity_window_seconds": window}))


async def _status_loop(session: ClientSession, autonomy: AutonomyConfig) -> None:
    interval = autonomy.status_interval_seconds
    if interval <= 0:
        return
    window = autonomy.recent_activity_window_seconds
    while True:
        snapshot = session.controller.activity_snapshot(window)
        mood = str(snapshot.get("mood") or "neutral")
        status_value = str(snapshot.get("status") or "idle")
        payload = {
            "session_id": session.session_id,
            "persona": session.controller.persona(),
            "user_id": session.controller.user_id(),
            "seconds_since_activity": snapshot.get("seconds_since_activity", 0.0),
            "recent_messages": snapshot.get("recent_messages", 0),
            "recent_activity_window_seconds": snapshot.get("recent_activity_window_seconds", window),
            "last_inbound_ts": get_pacific_timestamp(snapshot.get("last_inbound_ts", 0.0)),
            "last_response_ts": get_pacific_timestamp(snapshot.get("last_response_ts", 0.0)),
            "mood": mood,
            "mood_ts": get_pacific_timestamp(snapshot.get("mood_ts", 0.0)),
            "mood_source": snapshot.get("mood_source", ""),
            "status": status_value,
            "status_ts": get_pacific_timestamp(snapshot.get("status_ts", 0.0)),
            "status_source": snapshot.get("status_source", ""),
            "autonomy_enabled": autonomy.enabled,
        }
        try:
            await session.send_status(payload)
            if autonomy.status_channel is not None:
                seconds = float(payload["seconds_since_activity"])
                status_text = (
                    f"[Status]\n"
                    f"Mood: {payload['mood']}\n"
                    f"Status: {payload['status']}\n"
                    f"Idle: {seconds:.0f}s"
                )
                action = Action(
                    command_type=CommandType.CHAT,
                    content=status_text,
                    parameters={"channel": autonomy.status_channel, "source": "status"},
                )
                await session.send_actions([action])
        except ConnectionError:
            return
        await asyncio.sleep(interval)


async def _facts_deduplication_loop(config: ServerConfig, knowledge_store: KnowledgeStore, llm_client) -> None:
    from .llm import OpenRouterLLMClient
    if not isinstance(llm_client, OpenRouterLLMClient):
        return

    interval_seconds = config.facts_deduplication.interval_hours * 3600
    if interval_seconds <= 0:
        return

    from .neo4j_store import Neo4jKnowledgeStore
    if not isinstance(knowledge_store, Neo4jKnowledgeStore):
        return

    driver = knowledge_store.driver
    database = knowledge_store.database
    runtime_key = "facts_deduplication"
    try:
        with driver.session(database=database) as session:
            session.run(
                "MERGE (m:SystemRuntime {name: $name}) "
                "SET m.status = coalesce(m.status, 'idle'), "
                "    m.initialized_ts = coalesce(m.initialized_ts, $initialized_ts)",
                name=runtime_key,
                initialized_ts=time.time(),
            )
    except Exception as exc:
        logger.warning("Failed to initialize dedupe runtime metadata: %s", exc)

    while True:
        await asyncio.sleep(interval_seconds)
        logger.info("Starting periodic facts deduplication...")
        run_started_ts = time.time()
        candidate_count = 0
        processed_count = 0
        changed_count = 0
        unchanged_count = 0
        failed_count = 0
        try:
            with driver.session(database=database) as session:
                session.run(
                    "MERGE (m:SystemRuntime {name: $name}) "
                    "SET m.status = 'running', "
                    "    m.last_started_ts = $last_started_ts, "
                    "    m.last_error = ''",
                    name=runtime_key,
                    last_started_ts=run_started_ts,
                )
            with driver.session(database=database) as session:
                result = session.run(
                    "MATCH (p:Person) "
                    "WHERE size(coalesce(p.facts, [])) > 1 AND coalesce(p.needs_dedupe, false) = true "
                    "RETURN p.user_id as user_id, p.name as name, p.facts as facts"
                )
                people = list(result)
                candidate_count = len(people)

            logger.info("Facts dedupe candidates: %d", candidate_count)
            for person in people:
                user_id = person["user_id"]
                name = person["name"]
                facts = [item for item in (person["facts"] or []) if isinstance(item, str) and item.strip()]

                prompt = (
                    "You are a data cleaning expert. Refine these facts about a person, removing redundancy and duplicates. "
                    "Combine similar facts. Output a JSON object with key 'facts' containing a list of strings.\n\n"
                    f"Person Name: {name}\nCurrent Facts: {json.dumps(facts)}"
                )

                try:
                    response_text = await asyncio.to_thread(llm_client._request_text, "You are a precision data cleaner.", prompt)

                    clean_text = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", response_text, flags=re.DOTALL)
                    start = clean_text.find("{")
                    end = clean_text.rfind("}")
                    if start != -1 and end != -1:
                        clean_text = clean_text[start : end + 1]

                    data = json.loads(clean_text)
                    refined_raw = data.get("facts", [])
                    if not isinstance(refined_raw, list):
                        raise ValueError("facts must be a list")
                    seen: set[str] = set()
                    refined_facts: list[str] = []
                    for item in refined_raw:
                        if not isinstance(item, str):
                            continue
                        text = item.strip()
                        if not text or text in seen:
                            continue
                        seen.add(text)
                        refined_facts.append(text)
                    deduped_ts = time.time()

                    if refined_facts and refined_facts != facts:
                        logger.info(
                            "Deduplicated facts for %s (%s): %d -> %d",
                            name,
                            user_id,
                            len(facts),
                            len(refined_facts),
                        )
                        changed_count += 1
                        with driver.session(database=database) as session:
                            session.run(
                                "MATCH (p:Person {user_id: $user_id}) "
                                "SET p.facts = $facts, "
                                "    p.needs_dedupe = false, "
                                "    p.facts_deduped_ts = $facts_deduped_ts",
                                user_id=user_id,
                                facts=refined_facts,
                                facts_deduped_ts=deduped_ts,
                            )
                    else:
                        unchanged_count += 1
                        with driver.session(database=database) as session:
                            session.run(
                                "MATCH (p:Person {user_id: $user_id}) "
                                "SET p.needs_dedupe = false, "
                                "    p.facts_deduped_ts = $facts_deduped_ts",
                                user_id=user_id,
                                facts_deduped_ts=deduped_ts,
                            )
                    processed_count += 1
                except Exception as e:
                    failed_count += 1
                    logger.error("Failed to deduplicate facts for %s (%s): %s", name, user_id, e)
                    continue

            run_completed_ts = time.time()
            with driver.session(database=database) as session:
                session.run(
                    "MERGE (m:SystemRuntime {name: $name}) "
                    "SET m.status = 'success', "
                    "    m.last_completed_ts = $last_completed_ts, "
                    "    m.last_candidate_count = $last_candidate_count, "
                    "    m.last_processed_count = $last_processed_count, "
                    "    m.last_changed_count = $last_changed_count, "
                    "    m.last_unchanged_count = $last_unchanged_count, "
                    "    m.last_failed_count = $last_failed_count, "
                    "    m.last_error = ''",
                    name=runtime_key,
                    last_completed_ts=run_completed_ts,
                    last_candidate_count=int(candidate_count),
                    last_processed_count=int(processed_count),
                    last_changed_count=int(changed_count),
                    last_unchanged_count=int(unchanged_count),
                    last_failed_count=int(failed_count),
                )
            logger.info("Periodic facts deduplication complete.")
        except Exception as e:
            run_failed_ts = time.time()
            try:
                with driver.session(database=database) as session:
                    session.run(
                        "MERGE (m:SystemRuntime {name: $name}) "
                        "SET m.status = 'error', "
                        "    m.last_failed_ts = $last_failed_ts, "
                        "    m.last_candidate_count = $last_candidate_count, "
                        "    m.last_processed_count = $last_processed_count, "
                        "    m.last_changed_count = $last_changed_count, "
                        "    m.last_unchanged_count = $last_unchanged_count, "
                        "    m.last_failed_count = $last_failed_count, "
                        "    m.last_error = $last_error",
                        name=runtime_key,
                        last_failed_ts=run_failed_ts,
                        last_candidate_count=int(candidate_count),
                        last_processed_count=int(processed_count),
                        last_changed_count=int(changed_count),
                        last_unchanged_count=int(unchanged_count),
                        last_failed_count=int(failed_count),
                        last_error=str(e),
                    )
            except Exception as meta_exc:
                logger.warning("Failed to persist dedupe error metadata: %s", meta_exc)
            logger.error("Error in facts deduplication loop: %s", e)


async def _cancel_task(task: asyncio.Task | None, *, session_id: str, label: str) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    try:
        await task
    except (asyncio.CancelledError, ConnectionError):
        return
    except Exception as exc:
        logger.warning("Task %s failed for %s during shutdown: %s", label, session_id, exc)


def _format_error_text(record: logging.LogRecord, message: str, max_len: int = 280) -> str:
    name = record.name or "error"
    base = f"[error] {name}: {message}".strip()
    if len(base) <= max_len:
        return base
    return base[: max_len - 3].rstrip() + "..."


if __name__ == "__main__":
    main()
