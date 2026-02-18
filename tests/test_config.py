from __future__ import annotations

from gpt5_roleplay_system.config import load_config


def test_load_config_separates_embedding_key_from_neo4j_genai_key(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
api_keys:
  embedding_api_key: "embed-key"
  openai_api_key: "openai-key"
llm:
  embedding_model: "openai/text-embedding-3-large"
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.delenv("GPT5_ROLEPLAY_LLM_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NEO4J_GENAI_API_KEY", raising=False)

    config = load_config(str(config_path))
    assert config.llm.embedding_api_key == "embed-key"
    assert config.llm.neo4j_genai_api_key == "openai-key"
    assert config.llm.neo4j_genai_provider == "OpenAI"


def test_load_config_allows_neo4j_genai_env_override(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
api_keys:
  openai_api_key: "openai-from-config"
llm:
  embedding_model: "openai/text-embedding-3-large"
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("NEO4J_GENAI_API_KEY", "openai-from-env")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = load_config(str(config_path))
    assert config.llm.neo4j_genai_api_key == "openai-from-env"


def test_load_config_parses_neo4j_genai_only_flag(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
llm:
  neo4j_genai_only: true
  embedding_model: "openai/text-embedding-3-large"
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.delenv("NEO4J_GENAI_ONLY", raising=False)
    config = load_config(str(config_path))
    assert config.llm.neo4j_genai_only is True
