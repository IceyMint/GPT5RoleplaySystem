from __future__ import annotations

from gpt5_roleplay_system.experience_vector import EmbeddingClient
from gpt5_roleplay_system.neo4j_experience_vector import Neo4jExperienceVectorIndex, Neo4jVectorConfig
from gpt5_roleplay_system.neo4j_store import Neo4jKnowledgeStore
from gpt5_roleplay_system.cypher_utils import quote_cypher_identifier


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


class _FakeResultWithPeek:
    def __init__(self, *, peek_value=None, single_value=None):
        self._peek_value = peek_value
        self._single_value = single_value

    def consume(self):
        return None

    def peek(self):
        return self._peek_value

    def single(self):
        return self._single_value


class _RecordingSession:
    def __init__(self, responder):
        self.calls = []
        self._responder = responder

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, statement, **kwargs):
        self.calls.append((statement, kwargs))
        return self._responder(statement, kwargs)


def test_quote_cypher_identifier_supports_hyphenated_database_names():
    assert quote_cypher_identifier("gpt5-roleplay") == "`gpt5-roleplay`"


def test_quote_cypher_identifier_escapes_backticks():
    assert quote_cypher_identifier("db`name") == "`db``name`"


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


def test_update_last_seen_claims_existing_unowned_name_before_merging_by_id():
    def responder(statement, kwargs):
        if "MATCH (p:Person {name_lower: $name_lower})" in statement and "RETURN p" in statement:
            return _FakeResultWithPeek(peek_value={"claimed": True})
        raise AssertionError(f"unexpected query: {statement}")

    session = _RecordingSession(responder)
    store = object.__new__(Neo4jKnowledgeStore)
    store._session = lambda: session

    store.update_last_seen("user-2", "Evie", 123.0)

    assert len(session.calls) == 1
    _statement, params = session.calls[0]
    assert params["user_id"] == "user-2"
    assert params["name_lower"] == "evie"
    assert params["last_seen_ts"] == 123.0


def test_update_last_seen_skips_conflicting_name_lower_assignment():
    def responder(statement, kwargs):
        if "MATCH (p:Person {name_lower: $name_lower})" in statement and "RETURN p" in statement:
            return _FakeResultWithPeek(peek_value=None)
        if "RETURN coalesce(p.user_id, '') AS user_id" in statement:
            return _FakeResultWithPeek(single_value={"user_id": "user-existing"})
        if "MERGE (p:Person {user_id: $user_id})" in statement:
            return _FakeResultWithPeek()
        raise AssertionError(f"unexpected query: {statement}")

    session = _RecordingSession(responder)
    store = object.__new__(Neo4jKnowledgeStore)
    store._session = lambda: session

    store.update_last_seen("user-2", "Evie", 123.0)

    assert len(session.calls) == 3
    merge_statement, merge_params = session.calls[-1]
    assert "MERGE (p:Person {user_id: $user_id})" in merge_statement
    assert merge_params["user_id"] == "user-2"
    assert merge_params["name_lower"] == "evie"
    assert merge_params["set_name_lower"] is False


def test_upsert_person_facts_skips_conflicting_name_lower_assignment():
    def responder(statement, kwargs):
        if "MATCH (p:Person {name_lower: $name_lower})" in statement and "RETURN p" in statement:
            return _FakeResultWithPeek(peek_value=None)
        if "RETURN coalesce(p.user_id, '') AS user_id" in statement:
            return _FakeResultWithPeek(single_value={"user_id": "user-existing"})
        if "MERGE (p:Person {user_id: $user_id})" in statement:
            return _FakeResultWithPeek()
        raise AssertionError(f"unexpected query: {statement}")

    session = _RecordingSession(responder)
    store = object.__new__(Neo4jKnowledgeStore)
    store._session = lambda: session

    store.upsert_person_facts("user-2", "Evie", ["likes tea"])

    assert len(session.calls) == 3
    merge_statement, merge_params = session.calls[-1]
    assert "MERGE (p:Person {user_id: $user_id})" in merge_statement
    assert merge_params["user_id"] == "user-2"
    assert merge_params["name_lower"] == "evie"
    assert merge_params["set_name_lower"] is False
    assert merge_params["facts"] == ["likes tea"]
