from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class LLMConfig:
    base_url: str = "https://openrouter.ai/api/v1"
    embedding_base_url: str = ""
    api_key: str = ""
    model: str = "deepseek/deepseek-v3.2"
    bundle_model: str = ""
    summary_model: str = ""
    facts_model: str = ""
    address_model: str = ""
    embedding_model: str = ""
    embedding_api_key: str = ""
    neo4j_genai_api_key: str = ""
    neo4j_genai_provider: str = "OpenAI"
    neo4j_genai_only: bool = False
    embedding_dimensions: int = 3072
    max_tokens: int = 1024
    temperature: float = 0.6
    timeout_seconds: float = 30.0
    reasoning: str = ""
    provider_order: List[str] = field(default_factory=list)
    provider_allow_fallbacks: Optional[bool] = None
    facts_provider_order: Optional[List[str]] = None
    facts_provider_allow_fallbacks: Optional[bool] = None


@dataclass
class Neo4jConfig:
    uri: str = ""
    user: str = ""
    password: str = ""
    database: str = "neo4j"


@dataclass
class MemoryConfig:
    max_recent_messages: int = 20
    max_rolling_buffer: int = 30
    summary_strategy: str = "simple"  # simple | llm


@dataclass
class KnowledgeConfig:
    experience_similar_limit: int = 3
    experience_score_min: float = 0.78
    experience_score_delta: float = 0.03
    near_duplicate_collapse_enabled: bool = True
    near_duplicate_similarity: float = 0.9
    routine_summary_enabled: bool = False
    routine_summary_limit: int = 2
    routine_summary_min_count: int = 2


@dataclass
class FactsConfig:
    enabled: bool = True
    mode: str = "periodic"  # periodic | per_message
    interval_seconds: float = 30.0
    evidence_max_messages: int = 24
    in_bundle: bool = False
    min_pending_messages: int = 6
    max_pending_age_seconds: float = 120.0
    flush_on_overflow: bool = False


@dataclass
class FactsDeduplicationConfig:
    enabled: bool = True
    interval_hours: float = 4.0


@dataclass
class ExperienceDeduplicationConfig:
    enabled: bool = False
    dry_run: bool = True
    interval_hours: float = 6.0
    similarity_threshold: float = 0.995
    max_time_gap_hours: float = 6.0
    neighbor_k: int = 12


@dataclass
class EpisodeConfig:
    enabled: bool = True
    min_messages: int = 8
    max_messages: int = 30
    inactivity_seconds: float = 300.0
    forced_interval_seconds: float = 7200.0
    overlap_messages: int = 3
    persist_state: bool = True
    state_dir: str = ".episode_state"


@dataclass
class AutonomyConfig:
    enabled: bool = False
    base_delay_seconds: float = 90.0
    min_delay_seconds: float = 30.0
    max_delay_seconds: float = 600.0
    recent_activity_window_seconds: float = 45.0
    recent_activity_multiplier: float = 3.0
    status_interval_seconds: float = 15.0
    status_channel: Optional[int] = None


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "gpt5-roleplay"
    run_name: Optional[str] = None
    api_key: str = ""


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 9999
    persona: str = "DefaultPersona"
    user_id: str = ""
    config_path: str = ""
    queue_max_size: int = 200
    queue_drop_policy: str = "drop_oldest"  # drop_oldest | drop_newest
    chat_batch_window_ms: int = 250
    chat_batch_max: int = 6
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    facts: FactsConfig = field(default_factory=FactsConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    autonomy: AutonomyConfig = field(default_factory=AutonomyConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    facts_deduplication: FactsDeduplicationConfig = field(default_factory=FactsDeduplicationConfig)
    experience_deduplication: ExperienceDeduplicationConfig = field(default_factory=ExperienceDeduplicationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    max_environment_participants: int = 10
    posture_stale_seconds: float = 6.0
    persona_profiles: Dict[str, str] = field(default_factory=dict)


def _resolve_default_config_path() -> Optional[Path]:
    env_path = os.getenv("GPT5_ROLEPLAY_CONFIG", "")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
    local = Path.cwd() / "config.yaml"
    if local.exists():
        return local
    parent = Path.cwd().parent / "qwenRoleplayAISystem" / "config.yaml"
    if parent.exists():
        return parent
    return None


def _load_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "")
    if not raw:
        return bool(default)
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _parse_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return None


def _pick_model(llm_data: Dict[str, Any]) -> str:
    if not llm_data:
        return "deepseek/deepseek-v3.2"
    if llm_data.get("thinking_model"):
        return llm_data["thinking_model"]
    models = llm_data.get("models", {}) if isinstance(llm_data.get("models"), dict) else {}
    for key in ("context_understanding", "action_planning", "persona_consistency"):
        if key in models:
            return models[key]
    if isinstance(models, dict) and models:
        return next(iter(models.values()))
    return llm_data.get("model", "deepseek/deepseek-v3.2")


def load_config(path: Optional[str] = None) -> ServerConfig:
    from .config_loader import ConfigLoader

    return ConfigLoader().load(path)
