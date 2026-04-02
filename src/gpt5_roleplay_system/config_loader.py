from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import (
    AutonomyConfig,
    EpisodeConfig,
    ExperienceDeduplicationConfig,
    FactsConfig,
    FactsDeduplicationConfig,
    KnowledgeConfig,
    LLMConfig,
    MemoryConfig,
    Neo4jConfig,
    ServerConfig,
    WandbConfig,
)


logger = logging.getLogger("gpt5_roleplay_config")
_CONFIG_WRITE_LOCK = threading.Lock()


class _LiteralDumper(yaml.SafeDumper):
    pass


def _represent_multiline_str(dumper: yaml.SafeDumper, value: str) -> yaml.nodes.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_LiteralDumper.add_representer(str, _represent_multiline_str)


class ConfigLoader:
    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        self._env = dict(env or os.environ)

    def load(self, path: str | None = None) -> ServerConfig:
        config_path = Path(path) if path else self._resolve_default_config_path()
        raw = self._load_yaml(config_path)

        api_keys = raw.get("api_keys", {}) if isinstance(raw.get("api_keys"), dict) else {}
        llm_raw = raw.get("llm", {}) if isinstance(raw.get("llm"), dict) else {}

        llm_config = self.load_llm_config(raw, api_keys)
        memory_config = self.load_memory_config(raw, llm_config)
        knowledge_config = self.load_knowledge_config(raw)
        facts_config = self.load_facts_config(raw)
        episode_config = self.load_episode_config(raw)
        autonomy_config = self.load_autonomy_config(raw)
        neo4j_config = self.load_neo4j_config(raw)

        facts_deduplication_raw = raw.get("facts_deduplication", {}) if isinstance(raw.get("facts_deduplication"), dict) else {}
        experience_deduplication_raw = (
            raw.get("experience_deduplication", {}) if isinstance(raw.get("experience_deduplication"), dict) else {}
        )
        wandb_raw = raw.get("wandb", {}) if isinstance(raw.get("wandb"), dict) else {}
        persona_profiles_raw = raw.get("persona_profiles", {}) if isinstance(raw.get("persona_profiles"), dict) else {}

        facts_deduplication_config = FactsDeduplicationConfig(
            enabled=bool(facts_deduplication_raw.get("enabled", FactsDeduplicationConfig().enabled)),
            interval_hours=float(facts_deduplication_raw.get("interval_hours", FactsDeduplicationConfig().interval_hours)),
        )

        experience_deduplication_config = ExperienceDeduplicationConfig(
            enabled=bool(experience_deduplication_raw.get("enabled", ExperienceDeduplicationConfig().enabled)),
            dry_run=bool(experience_deduplication_raw.get("dry_run", ExperienceDeduplicationConfig().dry_run)),
            interval_hours=max(
                0.0,
                float(experience_deduplication_raw.get("interval_hours", ExperienceDeduplicationConfig().interval_hours)),
            ),
            similarity_threshold=min(
                1.0,
                max(
                    0.0,
                    float(
                        experience_deduplication_raw.get(
                            "similarity_threshold",
                            ExperienceDeduplicationConfig().similarity_threshold,
                        )
                    ),
                ),
            ),
            max_time_gap_hours=max(
                0.0,
                float(
                    experience_deduplication_raw.get(
                        "max_time_gap_hours",
                        ExperienceDeduplicationConfig().max_time_gap_hours,
                    )
                ),
            ),
            neighbor_k=max(
                2,
                int(experience_deduplication_raw.get("neighbor_k", ExperienceDeduplicationConfig().neighbor_k)),
            ),
        )

        wandb_config = WandbConfig(
            enabled=self._env_bool("GPT5_ROLEPLAY_WANDB_ENABLED", bool(wandb_raw.get("enabled", False))),
            project=wandb_raw.get("project", WandbConfig().project),
            run_name=wandb_raw.get("run_name"),
            api_key=self._getenv("WANDB_API_KEY", api_keys.get("wandb_api_key", "")),
        )

        persona = raw.get("ai_name") or self._getenv("GPT5_ROLEPLAY_PERSONA", ServerConfig().persona)

        persona_profiles: dict[str, str] = {}
        for key, value in persona_profiles_raw.items():
            if not isinstance(key, str):
                continue
            if not isinstance(value, str):
                continue
            persona_profiles[key.casefold()] = value.strip()

        return ServerConfig(
            host=self._getenv("GPT5_ROLEPLAY_HOST", ServerConfig().host),
            port=int(self._getenv("GPT5_ROLEPLAY_PORT", ServerConfig().port)),
            persona=persona,
            user_id=raw.get("user_id", self._getenv("GPT5_ROLEPLAY_USER_ID", "")),
            config_path=str(config_path) if config_path else "",
            queue_max_size=int(self._getenv("GPT5_ROLEPLAY_QUEUE_MAX", raw.get("queue_max_size", 200))),
            queue_drop_policy=self._getenv("GPT5_ROLEPLAY_QUEUE_DROP", raw.get("queue_drop_policy", "drop_oldest")),
            chat_batch_window_ms=int(self._getenv("GPT5_ROLEPLAY_BATCH_WINDOW_MS", raw.get("chat_batch_window_ms", 250))),
            chat_batch_max=int(self._getenv("GPT5_ROLEPLAY_BATCH_MAX", raw.get("chat_batch_max", 6))),
            llm=llm_config,
            memory=memory_config,
            knowledge=knowledge_config,
            facts=facts_config,
            episode=episode_config,
            autonomy=autonomy_config,
            neo4j=neo4j_config,
            facts_deduplication=facts_deduplication_config,
            experience_deduplication=experience_deduplication_config,
            wandb=wandb_config,
            max_environment_participants=int(self._getenv("GPT5_ROLEPLAY_MAX_PARTICIPANTS", 10)),
            posture_stale_seconds=float(
                self._getenv(
                    "GPT5_ROLEPLAY_POSTURE_STALE_SECONDS",
                    raw.get("posture_stale_seconds", ServerConfig().posture_stale_seconds),
                )
            ),
            persona_profiles=persona_profiles,
        )

    def ensure_persona_profile(
        self,
        path: str | Path | None,
        *,
        persona_name: str,
        template_name: str = "DefaultPersona",
    ) -> str:
        persona_name = str(persona_name or "").strip()
        template_name = str(template_name or "").strip()
        if not persona_name or not template_name:
            return ""
        config_path = Path(path).expanduser() if path else None
        if config_path is None or not config_path.exists():
            return ""
        with _CONFIG_WRITE_LOCK:
            raw = self._load_yaml(config_path)
            if not isinstance(raw, dict):
                return ""
            profiles = raw.get("persona_profiles")
            if not isinstance(profiles, dict):
                profiles = {}
                raw["persona_profiles"] = profiles
            persona_key = self._find_case_insensitive_key(profiles, persona_name)
            if persona_key is not None:
                existing = profiles.get(persona_key)
                return str(existing).strip() if isinstance(existing, str) else ""
            template_key = self._find_case_insensitive_key(profiles, template_name)
            if template_key is None:
                return ""
            template_value = profiles.get(template_key)
            if not isinstance(template_value, str):
                return ""
            instructions = template_value.strip()
            if not instructions:
                return ""
            profiles[persona_name] = instructions
            self._write_yaml(config_path, raw)
            return instructions

    def load_llm_config(self, raw: dict[str, Any], api_keys: dict[str, Any]) -> LLMConfig:
        llm_raw = raw.get("llm", {}) if isinstance(raw.get("llm"), dict) else {}
        provider_raw = llm_raw.get("provider", {}) if isinstance(llm_raw.get("provider"), dict) else {}
        facts_provider_raw = llm_raw.get("facts_provider", {}) if isinstance(llm_raw.get("facts_provider"), dict) else {}

        provider_order: list[str] = []
        provider_order_env = self._getenv("GPT5_ROLEPLAY_LLM_PROVIDER_ORDER", "").strip()
        if provider_order_env:
            provider_order = [item.strip() for item in provider_order_env.split(",") if item.strip()]
        else:
            order_raw = provider_raw.get("order", [])
            if isinstance(order_raw, str):
                provider_order = [item.strip() for item in order_raw.split(",") if item.strip()]
            elif isinstance(order_raw, list):
                provider_order = [str(item).strip() for item in order_raw if str(item).strip()]

        provider_allow_fallbacks = self._parse_optional_bool(
            self._getenv(
                "GPT5_ROLEPLAY_LLM_PROVIDER_ALLOW_FALLBACKS",
                provider_raw.get("allow_fallbacks"),
            )
        )

        facts_provider_order: list[str] | None = None
        facts_provider_order_env = self._getenv("GPT5_ROLEPLAY_LLM_FACTS_PROVIDER_ORDER", "").strip()
        if facts_provider_order_env:
            facts_provider_order = [item.strip() for item in facts_provider_order_env.split(",") if item.strip()]
        else:
            facts_order_raw = facts_provider_raw.get("order", None)
            if isinstance(facts_order_raw, str):
                facts_provider_order = [item.strip() for item in facts_order_raw.split(",") if item.strip()]
            elif isinstance(facts_order_raw, list):
                facts_provider_order = [str(item).strip() for item in facts_order_raw if str(item).strip()]

        facts_provider_allow_fallbacks: bool | None = None
        facts_provider_allow_fallbacks_env = self._getenv("GPT5_ROLEPLAY_LLM_FACTS_PROVIDER_ALLOW_FALLBACKS", "").strip()
        if facts_provider_allow_fallbacks_env:
            facts_provider_allow_fallbacks = self._parse_optional_bool(facts_provider_allow_fallbacks_env)
        elif "allow_fallbacks" in facts_provider_raw:
            facts_provider_allow_fallbacks = self._parse_optional_bool(facts_provider_raw.get("allow_fallbacks"))

        return LLMConfig(
            base_url=self._getenv("GPT5_ROLEPLAY_LLM_BASE_URL", llm_raw.get("base_url", LLMConfig().base_url)),
            api_key=self._getenv("OPENROUTER_API_KEY", api_keys.get("openrouter_api_key", "")),
            model=self._getenv("GPT5_ROLEPLAY_LLM_MODEL", self._pick_model(llm_raw)),
            bundle_model=self._getenv("GPT5_ROLEPLAY_LLM_BUNDLE_MODEL", llm_raw.get("bundle_model", "")),
            summary_model=self._getenv("GPT5_ROLEPLAY_LLM_SUMMARY_MODEL", llm_raw.get("summary_model", "")),
            facts_model=self._getenv("GPT5_ROLEPLAY_LLM_FACTS_MODEL", llm_raw.get("facts_model", "")),
            address_model=self._getenv("GPT5_ROLEPLAY_LLM_ADDRESS_MODEL", llm_raw.get("address_model", "")),
            embedding_base_url=self._getenv("GPT5_ROLEPLAY_LLM_EMBEDDING_BASE_URL", llm_raw.get("embedding_base_url", "")),
            embedding_model=self._getenv("GPT5_ROLEPLAY_LLM_EMBEDDING_MODEL", llm_raw.get("embedding_model", "")),
            embedding_api_key=self._getenv(
                "GPT5_ROLEPLAY_LLM_EMBEDDING_API_KEY",
                api_keys.get(
                    "embedding_api_key",
                    llm_raw.get(
                        "embedding_api_key",
                        api_keys.get("openai_api_key", self._getenv("OPENAI_API_KEY", "")),
                    ),
                ),
            ),
            neo4j_genai_api_key=self._getenv(
                "NEO4J_GENAI_API_KEY",
                llm_raw.get("neo4j_genai_api_key", api_keys.get("openai_api_key", self._getenv("OPENAI_API_KEY", ""))),
            ),
            neo4j_genai_provider=self._getenv(
                "NEO4J_GENAI_PROVIDER",
                llm_raw.get("neo4j_genai_provider", LLMConfig().neo4j_genai_provider),
            ),
            neo4j_genai_only=bool(
                self._parse_optional_bool(
                    self._getenv(
                        "NEO4J_GENAI_ONLY",
                        llm_raw.get("neo4j_genai_only"),
                    )
                )
            ),
            embedding_dimensions=int(
                self._getenv(
                    "GPT5_ROLEPLAY_LLM_EMBEDDING_DIMENSIONS",
                    llm_raw.get("embedding_dimensions", LLMConfig().embedding_dimensions),
                )
            ),
            max_tokens=int(self._getenv("GPT5_ROLEPLAY_LLM_MAX_TOKENS", llm_raw.get("max_tokens", 1024))),
            temperature=float(self._getenv("GPT5_ROLEPLAY_LLM_TEMPERATURE", llm_raw.get("temperature", 0.6))),
            timeout_seconds=float(self._getenv("GPT5_ROLEPLAY_LLM_TIMEOUT", llm_raw.get("timeout", 30.0))),
            reasoning=self._getenv("GPT5_ROLEPLAY_LLM_REASONING", llm_raw.get("reasoning", "")),
            provider_order=provider_order,
            provider_allow_fallbacks=provider_allow_fallbacks,
            facts_provider_order=facts_provider_order,
            facts_provider_allow_fallbacks=facts_provider_allow_fallbacks,
        )

    def load_memory_config(self, raw: dict[str, Any], llm_config: LLMConfig) -> MemoryConfig:
        memory_raw = raw.get("memory", {}) if isinstance(raw.get("memory"), dict) else {}
        knowledge_raw = raw.get("knowledge_storage", {}) if isinstance(raw.get("knowledge_storage"), dict) else {}

        summary_strategy = self._getenv(
            "GPT5_ROLEPLAY_SUMMARY",
            "llm" if llm_config.api_key else "simple",
        )
        if summary_strategy == "llm" and not llm_config.api_key:
            logger.warning(
                "Memory summary strategy downgraded from llm to simple because no LLM API key is configured."
            )
            summary_strategy = "simple"

        return MemoryConfig(
            max_recent_messages=int(
                self._getenv(
                    "GPT5_ROLEPLAY_MAX_RECENT",
                    memory_raw.get("max_recent_messages", knowledge_raw.get("context_window_size", 20)),
                )
            ),
            max_rolling_buffer=int(self._getenv("GPT5_ROLEPLAY_ROLLING", 30)),
            summary_strategy=summary_strategy,
        )

    def load_knowledge_config(self, raw: dict[str, Any]) -> KnowledgeConfig:
        knowledge_raw = raw.get("knowledge_storage", {}) if isinstance(raw.get("knowledge_storage"), dict) else {}
        return KnowledgeConfig(
            experience_similar_limit=int(knowledge_raw.get("experience_similar_limit", 3)),
            experience_score_min=float(knowledge_raw.get("experience_score_min", KnowledgeConfig().experience_score_min)),
            experience_score_delta=float(
                knowledge_raw.get("experience_score_delta", KnowledgeConfig().experience_score_delta)
            ),
            near_duplicate_collapse_enabled=bool(
                knowledge_raw.get(
                    "near_duplicate_collapse_enabled",
                    KnowledgeConfig().near_duplicate_collapse_enabled,
                )
            ),
            near_duplicate_similarity=float(
                knowledge_raw.get(
                    "near_duplicate_similarity",
                    KnowledgeConfig().near_duplicate_similarity,
                )
            ),
            routine_summary_enabled=bool(
                knowledge_raw.get(
                    "routine_summary_enabled",
                    KnowledgeConfig().routine_summary_enabled,
                )
            ),
            routine_summary_limit=int(knowledge_raw.get("routine_summary_limit", KnowledgeConfig().routine_summary_limit)),
            routine_summary_min_count=int(
                knowledge_raw.get("routine_summary_min_count", KnowledgeConfig().routine_summary_min_count)
            ),
        )

    def load_facts_config(self, raw: dict[str, Any]) -> FactsConfig:
        facts_raw = raw.get("facts", {}) if isinstance(raw.get("facts"), dict) else {}
        return FactsConfig(
            enabled=bool(facts_raw.get("enabled", FactsConfig().enabled)),
            mode=str(facts_raw.get("mode", FactsConfig().mode) or FactsConfig().mode),
            interval_seconds=float(facts_raw.get("interval_seconds", FactsConfig().interval_seconds)),
            evidence_max_messages=int(facts_raw.get("evidence_max_messages", FactsConfig().evidence_max_messages)),
            in_bundle=bool(facts_raw.get("in_bundle", FactsConfig().in_bundle)),
            min_pending_messages=int(facts_raw.get("min_pending_messages", FactsConfig().min_pending_messages)),
            max_pending_age_seconds=float(
                facts_raw.get("max_pending_age_seconds", FactsConfig().max_pending_age_seconds)
            ),
            flush_on_overflow=bool(facts_raw.get("flush_on_overflow", FactsConfig().flush_on_overflow)),
        )

    def load_episode_config(self, raw: dict[str, Any]) -> EpisodeConfig:
        episode_raw = raw.get("episode_summary", {}) if isinstance(raw.get("episode_summary"), dict) else {}
        return EpisodeConfig(
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

    def load_autonomy_config(self, raw: dict[str, Any]) -> AutonomyConfig:
        autonomy_raw = raw.get("autonomy", {}) if isinstance(raw.get("autonomy"), dict) else {}
        status_channel_raw = autonomy_raw.get("status_channel", AutonomyConfig().status_channel)
        status_channel = None
        if status_channel_raw not in (None, ""):
            status_channel = int(status_channel_raw)

        return AutonomyConfig(
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

    def load_neo4j_config(self, raw: dict[str, Any]) -> Neo4jConfig:
        db_raw = raw.get("database", {}) if isinstance(raw.get("database"), dict) else {}
        return Neo4jConfig(
            uri=self._getenv("NEO4J_URI", db_raw.get("uri", "")),
            user=self._getenv("NEO4J_USER", db_raw.get("user", "")),
            password=self._getenv("NEO4J_PASSWORD", db_raw.get("password", "")),
            database=self._getenv("NEO4J_DATABASE", db_raw.get("name", Neo4jConfig().database)),
        )

    def _resolve_default_config_path(self) -> Path | None:
        env_path = self._getenv("GPT5_ROLEPLAY_CONFIG", "")
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

    @staticmethod
    def _load_yaml(path: Path | None) -> dict[str, Any]:
        if path is None or not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _write_yaml(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.dump(
                data,
                handle,
                Dumper=_LiteralDumper,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )

    @staticmethod
    def _find_case_insensitive_key(mapping: dict[str, Any], target: str) -> str | None:
        key = str(target or "").casefold()
        if not key:
            return None
        for existing in mapping.keys():
            if str(existing).casefold() == key:
                return str(existing)
        return None

    def _env_bool(self, name: str, default: bool) -> bool:
        raw = self._getenv(name, "")
        if not raw:
            return bool(default)
        value = str(raw).strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    @staticmethod
    def _parse_optional_bool(value: Any) -> bool | None:
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

    @staticmethod
    def _pick_model(llm_data: dict[str, Any]) -> str:
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

    def _getenv(self, key: str, default: Any = "") -> Any:
        return self._env.get(key, default)
