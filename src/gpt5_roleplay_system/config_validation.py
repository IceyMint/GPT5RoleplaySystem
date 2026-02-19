from __future__ import annotations

from typing import Any

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - optional dependency
    BaseModel = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]


if BaseModel is not None:

    class RawProviderConfigModel(BaseModel):
        order: list[str] | str | None = None
        allow_fallbacks: bool | str | int | None = None

    class RawLLMConfigModel(BaseModel):
        model: str | None = None
        bundle_model: str | None = None
        summary_model: str | None = None
        facts_model: str | None = None
        address_model: str | None = None
        provider: RawProviderConfigModel | None = None
        facts_provider: RawProviderConfigModel | None = None

    class RawServerConfigModel(BaseModel):
        ai_name: str | None = None
        user_id: str | None = None
        llm: RawLLMConfigModel | None = None
        memory: dict[str, Any] = Field(default_factory=dict)
        knowledge_storage: dict[str, Any] = Field(default_factory=dict)
        facts: dict[str, Any] = Field(default_factory=dict)
        autonomy: dict[str, Any] = Field(default_factory=dict)
        database: dict[str, Any] = Field(default_factory=dict)
        episode_summary: dict[str, Any] = Field(default_factory=dict)
        facts_deduplication: dict[str, Any] = Field(default_factory=dict)
        experience_deduplication: dict[str, Any] = Field(default_factory=dict)
        wandb: dict[str, Any] = Field(default_factory=dict)
        api_keys: dict[str, Any] = Field(default_factory=dict)
        persona_profiles: dict[str, str] = Field(default_factory=dict)

else:  # pragma: no cover - optional dependency

    class RawServerConfigModel:  # type: ignore[no-redef]
        @classmethod
        def model_validate(cls, _value: Any):
            raise RuntimeError("pydantic is not installed")
