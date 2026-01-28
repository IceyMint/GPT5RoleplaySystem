from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class LLMConfig:
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    model: str = "deepseek/deepseek-v3.2"
    address_model: str = ""
    embedding_model: str = ""
    embedding_api_key: str = ""
    embedding_dimensions: int = 3072
    max_tokens: int = 500
    temperature: float = 0.6
    timeout_seconds: float = 30.0


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


@dataclass
class FactsConfig:
    enabled: bool = True
    mode: str = "periodic"  # periodic | per_message
    interval_seconds: float = 30.0
    evidence_max_messages: int = 24
    in_bundle: bool = False


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
    wandb: WandbConfig = field(default_factory=WandbConfig)
    max_environment_participants: int = 10
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
    config_path = Path(path) if path else _resolve_default_config_path()
    raw = _load_yaml(config_path)

    api_keys = raw.get("api_keys", {}) if isinstance(raw.get("api_keys"), dict) else {}
    llm_raw = raw.get("llm", {}) if isinstance(raw.get("llm"), dict) else {}
    memory_raw = raw.get("memory", {}) if isinstance(raw.get("memory"), dict) else {}
    knowledge_raw = raw.get("knowledge_storage", {}) if isinstance(raw.get("knowledge_storage"), dict) else {}
    facts_raw = raw.get("facts", {}) if isinstance(raw.get("facts"), dict) else {}
    episode_raw = raw.get("episode_summary", {}) if isinstance(raw.get("episode_summary"), dict) else {}
    autonomy_raw = raw.get("autonomy", {}) if isinstance(raw.get("autonomy"), dict) else {}
    db_raw = raw.get("database", {}) if isinstance(raw.get("database"), dict) else {}
    wandb_raw = raw.get("wandb", {}) if isinstance(raw.get("wandb"), dict) else {}
    persona_profiles_raw = raw.get("persona_profiles", {}) if isinstance(raw.get("persona_profiles"), dict) else {}

    llm_config = LLMConfig(
        base_url=os.getenv("GPT5_ROLEPLAY_LLM_BASE_URL", llm_raw.get("base_url", LLMConfig().base_url)),
        api_key=os.getenv("OPENROUTER_API_KEY", api_keys.get("openrouter_api_key", "")),
        model=os.getenv("GPT5_ROLEPLAY_LLM_MODEL", _pick_model(llm_raw)),
        address_model=os.getenv("GPT5_ROLEPLAY_LLM_ADDRESS_MODEL", llm_raw.get("address_model", "")),
        embedding_model=os.getenv("GPT5_ROLEPLAY_LLM_EMBEDDING_MODEL", llm_raw.get("embedding_model", "")),
        embedding_api_key=os.getenv(
            "OPENAI_API_KEY",
            api_keys.get("openai_api_key", llm_raw.get("embedding_api_key", "")),
        ),
        embedding_dimensions=int(
            os.getenv(
                "GPT5_ROLEPLAY_LLM_EMBEDDING_DIMENSIONS",
                llm_raw.get("embedding_dimensions", LLMConfig().embedding_dimensions),
            )
        ),
        max_tokens=int(os.getenv("GPT5_ROLEPLAY_LLM_MAX_TOKENS", llm_raw.get("max_tokens", 500))),
        temperature=float(os.getenv("GPT5_ROLEPLAY_LLM_TEMPERATURE", llm_raw.get("temperature", 0.6))),
        timeout_seconds=float(os.getenv("GPT5_ROLEPLAY_LLM_TIMEOUT", llm_raw.get("timeout", 30.0))),
    )

    summary_strategy = os.getenv(
        "GPT5_ROLEPLAY_SUMMARY",
        "llm" if llm_config.api_key else "simple",
    )
    if summary_strategy == "llm" and not llm_config.api_key:
        summary_strategy = "simple"

    memory_config = MemoryConfig(
        max_recent_messages=int(
            os.getenv(
                "GPT5_ROLEPLAY_MAX_RECENT",
                memory_raw.get("max_recent_messages", knowledge_raw.get("context_window_size", 20)),
            )
        ),
        max_rolling_buffer=int(os.getenv("GPT5_ROLEPLAY_ROLLING", 30)),
        summary_strategy=summary_strategy,
    )

    knowledge_config = KnowledgeConfig(
        experience_similar_limit=int(knowledge_raw.get("experience_similar_limit", 3)),
        experience_score_min=float(knowledge_raw.get("experience_score_min", KnowledgeConfig().experience_score_min)),
        experience_score_delta=float(
            knowledge_raw.get("experience_score_delta", KnowledgeConfig().experience_score_delta)
        ),
    )

    facts_config = FactsConfig(
        enabled=bool(facts_raw.get("enabled", FactsConfig().enabled)),
        mode=str(facts_raw.get("mode", FactsConfig().mode) or FactsConfig().mode),
        interval_seconds=float(facts_raw.get("interval_seconds", FactsConfig().interval_seconds)),
        evidence_max_messages=int(facts_raw.get("evidence_max_messages", FactsConfig().evidence_max_messages)),
        in_bundle=bool(facts_raw.get("in_bundle", FactsConfig().in_bundle)),
    )

    episode_config = EpisodeConfig(
        enabled=bool(episode_raw.get("enabled", EpisodeConfig().enabled)),
        min_messages=int(episode_raw.get("min_messages", EpisodeConfig().min_messages)),
        max_messages=int(episode_raw.get("max_messages", EpisodeConfig().max_messages)),
        inactivity_seconds=float(episode_raw.get("inactivity_seconds", EpisodeConfig().inactivity_seconds)),
        forced_interval_seconds=float(
            episode_raw.get("forced_interval_seconds", EpisodeConfig().forced_interval_seconds)
        ),
        overlap_messages=int(episode_raw.get("overlap_messages", EpisodeConfig().overlap_messages)),
        persist_state=bool(episode_raw.get("persist_state", EpisodeConfig().persist_state)),
        state_dir=str(episode_raw.get("state_dir", EpisodeConfig().state_dir)),
    )

    status_channel_raw = autonomy_raw.get("status_channel", AutonomyConfig().status_channel)
    status_channel = None
    if status_channel_raw not in (None, ""):
        status_channel = int(status_channel_raw)

    autonomy_config = AutonomyConfig(
        enabled=bool(autonomy_raw.get("enabled", AutonomyConfig().enabled)),
        base_delay_seconds=float(autonomy_raw.get("base_delay_seconds", AutonomyConfig().base_delay_seconds)),
        min_delay_seconds=float(autonomy_raw.get("min_delay_seconds", AutonomyConfig().min_delay_seconds)),
        max_delay_seconds=float(autonomy_raw.get("max_delay_seconds", AutonomyConfig().max_delay_seconds)),
        recent_activity_window_seconds=float(
            autonomy_raw.get(
                "recent_activity_window_seconds",
                AutonomyConfig().recent_activity_window_seconds,
            )
        ),
        recent_activity_multiplier=float(
            autonomy_raw.get("recent_activity_multiplier", AutonomyConfig().recent_activity_multiplier)
        ),
        status_interval_seconds=float(
            autonomy_raw.get("status_interval_seconds", AutonomyConfig().status_interval_seconds)
        ),
        status_channel=status_channel,
    )

    neo4j_config = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", db_raw.get("uri", "")),
        user=os.getenv("NEO4J_USER", db_raw.get("user", "")),
        password=os.getenv("NEO4J_PASSWORD", db_raw.get("password", "")),
        database=os.getenv("NEO4J_DATABASE", db_raw.get("name", Neo4jConfig().database)),
    )

    wandb_config = WandbConfig(
        enabled=bool(wandb_raw.get("enabled", False)),
        project=wandb_raw.get("project", WandbConfig().project),
        run_name=wandb_raw.get("run_name"),
        api_key=os.getenv("WANDB_API_KEY", api_keys.get("wandb_api_key", "")),
    )

    persona = raw.get("ai_name") or os.getenv("GPT5_ROLEPLAY_PERSONA", ServerConfig().persona)

    persona_profiles: Dict[str, str] = {}
    for key, value in persona_profiles_raw.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, str):
            continue
        persona_profiles[key.casefold()] = value.strip()

    return ServerConfig(
        host=os.getenv("GPT5_ROLEPLAY_HOST", ServerConfig().host),
        port=int(os.getenv("GPT5_ROLEPLAY_PORT", ServerConfig().port)),
        persona=persona,
        user_id=raw.get("user_id", os.getenv("GPT5_ROLEPLAY_USER_ID", "")),
        queue_max_size=int(os.getenv("GPT5_ROLEPLAY_QUEUE_MAX", raw.get("queue_max_size", 200))),
        queue_drop_policy=os.getenv("GPT5_ROLEPLAY_QUEUE_DROP", raw.get("queue_drop_policy", "drop_oldest")),
        chat_batch_window_ms=int(os.getenv("GPT5_ROLEPLAY_BATCH_WINDOW_MS", raw.get("chat_batch_window_ms", 250))),
        chat_batch_max=int(os.getenv("GPT5_ROLEPLAY_BATCH_MAX", raw.get("chat_batch_max", 6))),
        llm=llm_config,
        memory=memory_config,
        knowledge=knowledge_config,
        facts=facts_config,
        episode=episode_config,
        autonomy=autonomy_config,
        neo4j=neo4j_config,
        wandb=wandb_config,
        max_environment_participants=int(os.getenv("GPT5_ROLEPLAY_MAX_PARTICIPANTS", 10)),
        persona_profiles=persona_profiles,
    )
