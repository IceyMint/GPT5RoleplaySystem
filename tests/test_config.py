from __future__ import annotations

import yaml

from gpt5_roleplay_system.config import load_config
from gpt5_roleplay_system.config_loader import ConfigLoader


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


def test_load_config_parses_routine_summary_settings(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
knowledge_storage:
  near_duplicate_collapse_enabled: true
  near_duplicate_similarity: 0.95
  routine_summary_enabled: true
  routine_summary_limit: 3
  routine_summary_min_count: 4
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.delenv("NEO4J_GENAI_ONLY", raising=False)
    config = load_config(str(config_path))
    assert config.knowledge.near_duplicate_collapse_enabled is True
    assert config.knowledge.near_duplicate_similarity == 0.95
    assert config.knowledge.routine_summary_enabled is True
    assert config.knowledge.routine_summary_limit == 3
    assert config.knowledge.routine_summary_min_count == 4


def test_load_config_parses_experience_deduplication_settings(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experience_deduplication:
  enabled: true
  dry_run: false
  interval_hours: 2
  similarity_threshold: 0.997
  max_time_gap_hours: 12
  neighbor_k: 16
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.delenv("NEO4J_GENAI_ONLY", raising=False)
    config = load_config(str(config_path))
    assert config.experience_deduplication.enabled is True
    assert config.experience_deduplication.dry_run is False
    assert config.experience_deduplication.interval_hours == 2.0
    assert config.experience_deduplication.similarity_threshold == 0.997
    assert config.experience_deduplication.max_time_gap_hours == 12.0
    assert config.experience_deduplication.neighbor_k == 16


def test_load_config_parses_facts_provider_override(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
llm:
  thinking_model: minimax/minimax-m2.5
  facts_model: z-ai/glm-5
  facts_provider:
    order:
      - siliconflow
    allow_fallbacks: false
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.delenv("GPT5_ROLEPLAY_LLM_PROVIDER_ORDER", raising=False)
    monkeypatch.delenv("GPT5_ROLEPLAY_LLM_PROVIDER_ALLOW_FALLBACKS", raising=False)
    monkeypatch.delenv("GPT5_ROLEPLAY_LLM_FACTS_PROVIDER_ORDER", raising=False)
    monkeypatch.delenv("GPT5_ROLEPLAY_LLM_FACTS_PROVIDER_ALLOW_FALLBACKS", raising=False)

    config = load_config(str(config_path))
    assert config.llm.model == "minimax/minimax-m2.5"
    assert config.llm.facts_model == "z-ai/glm-5"
    assert config.llm.provider_order == []
    assert config.llm.provider_allow_fallbacks is None
    assert config.llm.facts_provider_order == ["siliconflow"]
    assert config.llm.facts_provider_allow_fallbacks is False


def test_load_config_parses_posture_stale_seconds(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
posture_stale_seconds: 8.5
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.delenv("GPT5_ROLEPLAY_POSTURE_STALE_SECONDS", raising=False)
    config = load_config(str(config_path))
    assert config.posture_stale_seconds == 8.5


def test_ensure_persona_profile_clones_default_persona_template(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
ai_name: DefaultPersona
persona_profiles:
  DefaultPersona: |
    Be warm.
    Stay brief.
""".strip(),
        encoding="utf-8",
    )

    instructions = ConfigLoader().ensure_persona_profile(
        config_path,
        persona_name="Isabella",
        template_name="DefaultPersona",
    )

    assert instructions == "Be warm.\nStay brief."
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert payload["persona_profiles"]["Isabella"] == "Be warm.\nStay brief."
