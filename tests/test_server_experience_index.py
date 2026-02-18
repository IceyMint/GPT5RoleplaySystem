from __future__ import annotations

import gpt5_roleplay_system.server as server_module
from gpt5_roleplay_system.config import ServerConfig


class _FakeEmbedderClient:
    def __init__(self, api_key: str, base_url: str, model: str, timeout_seconds: float) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        return True


class _FakeNeo4jStore:
    def __init__(self) -> None:
        self.driver = object()
        self.database = "neo4j"


class _FakeNeo4jIndex:
    calls = []

    def __init__(self, driver, database, config, token, model, external_embedder=None) -> None:
        self.driver = driver
        self.database = database
        self.config = config
        self.token = token
        self.model = model
        self.external_embedder = external_embedder
        _FakeNeo4jIndex.calls.append(self)

    def is_enabled(self) -> bool:
        return True


def test_build_experience_index_prefers_dedicated_neo4j_genai_key(monkeypatch):
    monkeypatch.setattr(server_module, "OpenAIEmbeddingClient", _FakeEmbedderClient)
    monkeypatch.setattr(server_module, "Neo4jKnowledgeStore", _FakeNeo4jStore)
    monkeypatch.setattr(server_module, "Neo4jExperienceVectorIndex", _FakeNeo4jIndex)
    _FakeNeo4jIndex.calls.clear()

    config = ServerConfig()
    config.llm.embedding_model = "openai/text-embedding-3-large"
    config.llm.embedding_api_key = "sk-or-v1-embedding"
    config.llm.neo4j_genai_api_key = "sk-openai-genai"

    index = server_module._build_experience_index(config, _FakeNeo4jStore())
    assert index is not None
    assert _FakeNeo4jIndex.calls
    assert _FakeNeo4jIndex.calls[-1].token == "sk-openai-genai"


def test_build_experience_index_does_not_reuse_openrouter_key_for_neo4j_genai(monkeypatch):
    monkeypatch.setattr(server_module, "OpenAIEmbeddingClient", _FakeEmbedderClient)
    monkeypatch.setattr(server_module, "Neo4jKnowledgeStore", _FakeNeo4jStore)
    monkeypatch.setattr(server_module, "Neo4jExperienceVectorIndex", _FakeNeo4jIndex)
    _FakeNeo4jIndex.calls.clear()

    config = ServerConfig()
    config.llm.embedding_model = "openai/text-embedding-3-large"
    config.llm.embedding_api_key = "sk-or-v1-embedding"
    config.llm.neo4j_genai_api_key = ""

    index = server_module._build_experience_index(config, _FakeNeo4jStore())
    assert index is not None
    assert _FakeNeo4jIndex.calls
    assert _FakeNeo4jIndex.calls[-1].token == ""


def test_build_experience_index_can_disable_external_fallback(monkeypatch):
    monkeypatch.setattr(server_module, "OpenAIEmbeddingClient", _FakeEmbedderClient)
    monkeypatch.setattr(server_module, "Neo4jKnowledgeStore", _FakeNeo4jStore)
    monkeypatch.setattr(server_module, "Neo4jExperienceVectorIndex", _FakeNeo4jIndex)
    _FakeNeo4jIndex.calls.clear()

    config = ServerConfig()
    config.llm.embedding_model = "openai/text-embedding-3-large"
    config.llm.embedding_api_key = "sk-or-v1-embedding"
    config.llm.neo4j_genai_api_key = "sk-openai-genai"
    config.llm.neo4j_genai_only = True

    index = server_module._build_experience_index(config, _FakeNeo4jStore())
    assert index is not None
    assert _FakeNeo4jIndex.calls
    assert _FakeNeo4jIndex.calls[-1].external_embedder.is_available() is False


def test_build_experience_dedupe_plans_merges_high_similarity_close_time_pairs():
    rows = [
        {
            "left_id": "e1",
            "left_timestamp": 100.0,
            "left_timestamp_start": "",
            "left_timestamp_end": "",
            "right_id": "e2",
            "right_timestamp": 200.0,
            "right_timestamp_start": "",
            "right_timestamp_end": "",
            "score": 0.998,
        },
        {
            "left_id": "e2",
            "left_timestamp": 200.0,
            "left_timestamp_start": "",
            "left_timestamp_end": "",
            "right_id": "e3",
            "right_timestamp": 260.0,
            "right_timestamp_start": "",
            "right_timestamp_end": "",
            "score": 0.997,
        },
    ]

    plans, qualifying_pairs = server_module._build_experience_dedupe_plans(
        rows,
        similarity_threshold=0.995,
        max_gap_seconds=3600.0,
    )

    assert qualifying_pairs == 2
    assert len(plans) == 1
    plan = plans[0]
    assert plan["keep_id"] == "e3"
    assert plan["dup_ids"] == ["e1", "e2"]
    assert plan["merged_timestamp"] == 260.0
    assert plan["merged_timestamp_start"] == "1969-12-31 16:01:40"
    assert plan["merged_timestamp_end"] == "1969-12-31 16:04:20"


def test_build_experience_dedupe_plans_skips_far_apart_pairs():
    rows = [
        {
            "left_id": "e1",
            "left_timestamp": 100.0,
            "left_timestamp_start": "",
            "left_timestamp_end": "",
            "right_id": "e2",
            "right_timestamp": 90000.0,
            "right_timestamp_start": "",
            "right_timestamp_end": "",
            "score": 0.999,
        },
    ]

    plans, qualifying_pairs = server_module._build_experience_dedupe_plans(
        rows,
        similarity_threshold=0.995,
        max_gap_seconds=3600.0,
    )

    assert qualifying_pairs == 0
    assert plans == []
