from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .experience_vector import EmbeddingClient
from .memory import ExperienceRecord


@dataclass
class Neo4jVectorConfig:
    index_name: str = "experience_embedding_idx"
    dimensions: int = 3072
    similarity: str = "cosine"
    provider: str = "OpenAI"


class Neo4jExperienceVectorIndex:
    def __init__(
        self,
        driver,
        database: str,
        config: Neo4jVectorConfig,
        token: str,
        model: str,
        external_embedder: Optional[EmbeddingClient] = None,
    ) -> None:
        self._driver = driver
        self._database = database
        self._config = config
        self._token = token
        self._model = model
        self._external_embedder = external_embedder
        self._genai_available = False
        self._ensure_vector_index()
        self._genai_available = self._check_genai_available()

    def is_enabled(self) -> bool:
        return bool(self._model and self._token)

    async def add_record_async(self, record: ExperienceRecord, persona_id: str) -> None:
        if not self.is_enabled():
            return
        await asyncio.to_thread(self._add_record_sync, record, persona_id)

    async def search(self, query: str, persona_id: str, top_k: int = 3) -> List[ExperienceRecord]:
        if not self.is_enabled() or not query.strip():
            return []
        return await asyncio.to_thread(self._search_sync, query, persona_id, top_k)

    def _session(self):
        return self._driver.session(database=self._database)

    def _ensure_vector_index(self) -> None:
        statement = (
            f"CREATE VECTOR INDEX {self._config.index_name} IF NOT EXISTS "
            "FOR (e:Experience) ON (e.embedding) "
            "OPTIONS {indexConfig: {"
            f"`vector.dimensions`: {int(self._config.dimensions)}, "
            f"`vector.similarity_function`: '{self._config.similarity}'"
            "}}"
        )
        with self._session() as session:
            try:
                session.run(statement)
            except Exception:
                # Ignore if the Neo4j version doesn't support vector indexes.
                pass

    def _check_genai_available(self) -> bool:
        probe = (
            "RETURN genai.vector.encode('hello', $provider, {token: $token, model: $model}) AS v"
        )
        with self._session() as session:
            try:
                session.run(
                    probe,
                    provider=self._config.provider,
                    token=self._token,
                    model=self._model,
                ).consume()
                return True
            except Exception:
                return False

    def _add_record_sync(self, record: ExperienceRecord, persona_id: str) -> None:
        if self._genai_available:
            self._add_with_genai(record, persona_id)
            return
        if self._external_embedder and self._external_embedder.is_available():
            self._add_with_external_embedder(record, persona_id)

    def _add_with_genai(self, record: ExperienceRecord, persona_id: str) -> None:
        metadata = _normalize_metadata(record.metadata)
        statement = """
        WITH genai.vector.encode($text, $provider, {token: $token, model: $model}) AS embedding
        MERGE (p:Persona {id: $persona_id})
        MERGE (e:Experience {id: $id})
        SET e.text = $text,
            e.sender_id = $sender_id,
            e.sender_name = $sender_name,
            e.timestamp = $timestamp,
            e.timestamp_start = $timestamp_start,
            e.timestamp_end = $timestamp_end,
            e.embedding = embedding,
            e.persona_id = $persona_id
        MERGE (p)-[:HAD_EXPERIENCE]->(e)
        """
        with self._session() as session:
            try:
                session.run(
                    statement,
                    text=record.text,
                    provider=self._config.provider,
                    token=self._token,
                    model=self._model,
                    id=metadata.get("experience_id", ""),
                    sender_id=metadata.get("sender_id", ""),
                    sender_name=metadata.get("sender_name", ""),
                    timestamp=float(metadata.get("timestamp", 0.0) or 0.0),
                    timestamp_start=str(metadata.get("timestamp_start", "") or ""),
                    timestamp_end=str(metadata.get("timestamp_end", "") or ""),
                    persona_id=persona_id,
                ).consume()
            except Exception:
                # If genai fails at runtime, fall back to external embedder if available.
                self._genai_available = False
                if self._external_embedder and self._external_embedder.is_available():
                    self._add_with_external_embedder(record, persona_id)

    def _add_with_external_embedder(self, record: ExperienceRecord, persona_id: str) -> None:
        vectors = self._external_embedder.embed([record.text]) if self._external_embedder else [[]]
        embedding = vectors[0] if vectors else []
        if not embedding:
            return
        metadata = _normalize_metadata(record.metadata)
        statement = """
        MERGE (p:Persona {id: $persona_id})
        MERGE (e:Experience {id: $id})
        SET e.text = $text,
            e.sender_id = $sender_id,
            e.sender_name = $sender_name,
            e.timestamp = $timestamp,
            e.timestamp_start = $timestamp_start,
            e.timestamp_end = $timestamp_end,
            e.embedding = $embedding,
            e.persona_id = $persona_id
        MERGE (p)-[:HAD_EXPERIENCE]->(e)
        """
        with self._session() as session:
            session.run(
                statement,
                id=metadata.get("experience_id", ""),
                text=record.text,
                sender_id=metadata.get("sender_id", ""),
                sender_name=metadata.get("sender_name", ""),
                timestamp=float(metadata.get("timestamp", 0.0) or 0.0),
                timestamp_start=str(metadata.get("timestamp_start", "") or ""),
                timestamp_end=str(metadata.get("timestamp_end", "") or ""),
                embedding=embedding,
                persona_id=persona_id,
            ).consume()

    def _search_sync(self, query: str, persona_id: str, top_k: int) -> List[ExperienceRecord]:
        if self._genai_available:
            return self._search_with_genai(query, persona_id, top_k)
        if self._external_embedder and self._external_embedder.is_available():
            vectors = self._external_embedder.embed([query])
            if not vectors or not vectors[0]:
                return []
            return self._search_with_embedding(vectors[0], persona_id, top_k)
        return []

    def _search_with_genai(self, query: str, persona_id: str, top_k: int) -> List[ExperienceRecord]:
        # Filter by relationship to the Persona node.
        statement = """
        MATCH (p:Persona {id: $persona_id})-[:HAD_EXPERIENCE]->(e:Experience)
        WITH p, collect(e) as persona_experiences
        WITH genai.vector.encode($text, $provider, {token: $token, model: $model}) AS embedding, persona_experiences
        CALL db.index.vector.queryNodes($index_name, $top_k, embedding) YIELD node, score
        WHERE node IN persona_experiences
        RETURN node, score
        ORDER BY score DESC
        LIMIT $top_k
        """
        with self._session() as session:
            try:
                records = session.run(
                    statement,
                    text=query,
                    provider=self._config.provider,
                    token=self._token,
                    model=self._model,
                    index_name=self._config.index_name,
                    top_k=int(top_k),
                    persona_id=persona_id,
                )
                return _records_from_query(records)
            except Exception:
                self._genai_available = False
                return []

    def _search_with_embedding(self, embedding: List[float], persona_id: str, top_k: int) -> List[ExperienceRecord]:
        # Filter by relationship to the Persona node.
        statement = """
        MATCH (p:Persona {id: $persona_id})-[:HAD_EXPERIENCE]->(e:Experience)
        WITH collect(e) as persona_experiences
        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding) YIELD node, score
        WHERE node IN persona_experiences
        RETURN node, score
        ORDER BY score DESC
        LIMIT $top_k
        """
        with self._session() as session:
            records = session.run(
                statement,
                index_name=self._config.index_name,
                top_k=int(top_k),
                embedding=embedding,
                persona_id=persona_id,
            )
            return _records_from_query(records)


def _normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(metadata or {})
    for key in ("experience_id", "sender_id", "sender_name"):
        value = data.get(key, "")
        if not isinstance(value, str):
            data[key] = str(value)
    return data


def _records_from_query(records) -> List[ExperienceRecord]:
    results: List[ExperienceRecord] = []
    for record in records:
        node = record.get("node")
        score = record.get("score", 0.0)
        if node is None:
            continue
        text = node.get("text", "")
        metadata = {
            "experience_id": node.get("id", ""),
            "sender_id": node.get("sender_id", ""),
            "sender_name": node.get("sender_name", ""),
            "timestamp": node.get("timestamp", 0.0),
            "timestamp_start": node.get("timestamp_start", ""),
            "timestamp_end": node.get("timestamp_end", ""),
            "score": float(score or 0.0),
            "source": "neo4j_vector",
        }
        results.append(ExperienceRecord(text=text, metadata=metadata))
    return results

