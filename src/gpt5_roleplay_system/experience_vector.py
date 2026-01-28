from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

from .memory import ExperienceRecord


class EmbeddingClient:
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError

    def is_available(self) -> bool:
        return True


class NullEmbeddingClient(EmbeddingClient):
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [[] for _ in texts]

    def is_available(self) -> bool:
        return False


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        cache_size: int = 256,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        self._client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"), timeout=timeout_seconds)
        self._model = model
        self._available = bool(api_key and model)
        self._cache_size = max(0, int(cache_size))
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._cache_lock = Lock()

    def is_available(self) -> bool:
        return self._available

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not self._available or not texts:
            return [[] for _ in texts]
        results: List[Optional[List[float]]] = [None] * len(texts)
        missing: OrderedDict[str, List[int]] = OrderedDict()

        for idx, raw_text in enumerate(texts):
            text = str(raw_text)
            cached = self._get_cached(text)
            if cached:
                results[idx] = cached
                continue
            missing.setdefault(text, []).append(idx)

        if missing:
            missing_texts = list(missing.keys())
            response = self._client.embeddings.create(model=self._model, input=missing_texts)
            vectors: List[List[float]] = []
            for item in getattr(response, "data", []) or []:
                vectors.append(list(getattr(item, "embedding", []) or []))
            while len(vectors) < len(missing_texts):
                vectors.append([])
            for text, vector in zip(missing_texts, vectors):
                if vector:
                    self._set_cached(text, vector)
                for idx in missing.get(text, []):
                    results[idx] = vector

        return [vector or [] for vector in results]

    def _get_cached(self, text: str) -> List[float]:
        if self._cache_size <= 0:
            return []
        with self._cache_lock:
            vector = self._cache.pop(text, None)
            if vector is None:
                return []
            # Move to end to preserve LRU ordering.
            self._cache[text] = vector
            return vector

    def _set_cached(self, text: str, vector: List[float]) -> None:
        if self._cache_size <= 0 or not vector:
            return
        with self._cache_lock:
            self._cache[text] = vector
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)


@dataclass
class VectorItem:
    record: ExperienceRecord
    embedding: List[float]


class ExperienceVectorIndex:
    def __init__(self, embedder: EmbeddingClient, max_items: int = 2000) -> None:
        self._embedder = embedder
        self._max_items = max_items
        self._items: List[VectorItem] = []
        self._lock = Lock()

    def is_enabled(self) -> bool:
        return self._embedder.is_available()

    async def add_record_async(self, record: ExperienceRecord) -> None:
        if not self.is_enabled():
            return
        await asyncio.to_thread(self._add_record_sync, record)

    async def search(self, query: str, top_k: int = 3) -> List[ExperienceRecord]:
        if not self.is_enabled() or not query.strip():
            return []
        query_embedding = await asyncio.to_thread(self._embed_query, query)
        if not query_embedding:
            return []
        return await asyncio.to_thread(self._search_with_embedding, query_embedding, top_k)

    def _add_record_sync(self, record: ExperienceRecord) -> None:
        embedding = self._embed_query(record.text)
        if not embedding:
            return
        with self._lock:
            self._items.append(VectorItem(record=record, embedding=embedding))
            self._trim_locked()

    def _embed_query(self, text: str) -> List[float]:
        vectors = self._embedder.embed([text])
        if not vectors:
            return []
        return vectors[0]

    def _search_with_embedding(self, query_embedding: List[float], top_k: int) -> List[ExperienceRecord]:
        with self._lock:
            candidates = list(self._items)
        scored: List[Tuple[float, ExperienceRecord]] = []
        for item in candidates:
            score = _cosine_similarity(query_embedding, item.embedding)
            if score <= 0.0:
                continue
            scored.append((score, item.record))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    def _trim_locked(self) -> None:
        if len(self._items) <= self._max_items:
            return
        overflow = len(self._items) - self._max_items
        if overflow > 0:
            del self._items[:overflow]


def _cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    a_list = list(a)
    b_list = list(b)
    if not a_list or not b_list or len(a_list) != len(b_list):
        return 0.0
    dot = sum(x * y for x, y in zip(a_list, b_list))
    norm_a = math.sqrt(sum(x * x for x in a_list))
    norm_b = math.sqrt(sum(y * y for y in b_list))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
