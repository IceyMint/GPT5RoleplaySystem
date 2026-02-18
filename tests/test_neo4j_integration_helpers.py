from __future__ import annotations

from gpt5_roleplay_system.experience_vector import EmbeddingClient
from gpt5_roleplay_system.neo4j_experience_vector import Neo4jExperienceVectorIndex, Neo4jVectorConfig
from gpt5_roleplay_system.neo4j_store import _quote_cypher_identifier


class _FakeEmbedder(EmbeddingClient):
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def is_available(self) -> bool:
        return True


class _FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, *args, **kwargs):  # pragma: no cover - no-op test double
        class _Result:
            def consume(self):
                return None

        return _Result()


class _FakeDriver:
    def session(self, database=None):
        return _FakeSession()


def test_quote_cypher_identifier_supports_hyphenated_database_names():
    assert _quote_cypher_identifier("gpt5-roleplay") == "`gpt5-roleplay`"


def test_quote_cypher_identifier_escapes_backticks():
    assert _quote_cypher_identifier("db`name") == "`db``name`"


def test_neo4j_vector_index_enabled_with_external_embedder_without_genai_token():
    index = Neo4jExperienceVectorIndex(
        driver=_FakeDriver(),
        database="neo4j",
        config=Neo4jVectorConfig(),
        token="",
        model="openai/text-embedding-3-large",
        external_embedder=_FakeEmbedder(),
    )
    assert index.is_enabled() is True

