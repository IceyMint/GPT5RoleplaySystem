from gpt5_roleplay_system.llm import (
    OpenRouterLLMClient,
    PromptCacheStats,
    StructuredAction,
    StructuredBundle,
    _clean_json,
    _bundle_from_structured,
    _extract_prompt_cache_usage,
)
from gpt5_roleplay_system.models import ConversationContext, EnvironmentSnapshot


def test_mixed_action_type_emits_both_chat_and_primary():
    if StructuredAction is None or StructuredBundle is None:
        return
    action = StructuredAction(
        type="LOOK_AT",
        content="Hello, Evie.",
        target_uuid="4405928b-269c-d1a4-464c-1d0e6d16f346",
        x=1.0,
        y=2.0,
        z=3.0,
        parameters={"type": "CHAT", "recipient": "Evie"},
    )
    bundle = StructuredBundle(text="", actions=[action])
    result = _bundle_from_structured(bundle)
    types = [item.command_type.value for item in result.actions]
    assert types[0] == "CHAT"
    assert "LOOK_AT" in types
    chat = result.actions[0]
    assert chat.content == "Hello, Evie."
    assert chat.parameters.get("recipient") == "Evie"


def test_autonomous_text_only_does_not_emit_chat_action():
    if StructuredBundle is None:
        return
    bundle = StructuredBundle(text="I'll wait a bit longer.", actions=[])
    result = _bundle_from_structured(bundle, mode="autonomous")
    assert result.text
    assert result.actions == []


def test_chat_mode_text_only_emits_chat_action():
    if StructuredBundle is None:
        return
    bundle = StructuredBundle(text="Hello there.", actions=[])
    result = _bundle_from_structured(bundle, mode="chat")
    assert result.actions
    assert result.actions[0].command_type.value == "CHAT"
    assert result.actions[0].content == "Hello there."


def test_structured_action_accepts_command_alias():
    if StructuredAction is None or StructuredBundle is None:
        return
    action = StructuredAction.model_validate({"command": "CHAT", "content": "Hi"})
    bundle = StructuredBundle(text="", actions=[action])
    result = _bundle_from_structured(bundle, mode="chat")
    assert result.actions
    assert result.actions[0].command_type.value == "CHAT"
    assert result.actions[0].content == "Hi"


def test_structured_action_accepts_action_alias():
    if StructuredAction is None or StructuredBundle is None:
        return
    action = StructuredAction.model_validate({"action": "CHAT", "content": "Hi"})
    bundle = StructuredBundle(text="", actions=[action])
    result = _bundle_from_structured(bundle, mode="chat")
    assert result.actions
    assert result.actions[0].command_type.value == "CHAT"
    assert result.actions[0].content == "Hi"


def test_system_prompt_includes_persona_instructions():
    env = EnvironmentSnapshot()
    context = ConversationContext(
        persona="isabella.elara",
        user_id="ai-uuid",
        environment=env,
        participants=[],
        people_facts={},
        recent_messages=[],
        summary="",
        related_experiences=[],
        summary_meta={},
        agent_state={},
        persona_instructions="You are Isabella, a friendly cat.",
    )
    client = OpenRouterLLMClient(api_key="test-key", base_url="http://localhost:1234", model="test-model")
    prompt = client._system_prompt_for_context(context)
    assert "Persona name: isabella.elara." in prompt
    assert "You are Isabella, a friendly cat." in prompt


class _Usage:
    def __init__(self) -> None:
        self.prompt_tokens = 120
        self.completion_tokens = 30
        self.total_tokens = 150
        self.prompt_tokens_details = {"cached_tokens": 40, "cache_write_tokens": 10}
        self.cache_discount = 0.0012


class _Completion:
    def __init__(self) -> None:
        self.usage = _Usage()


def test_extract_prompt_cache_usage_parses_usage_fields():
    sample = _extract_prompt_cache_usage(_Completion())
    assert sample is not None
    assert sample.prompt_tokens == 120
    assert sample.completion_tokens == 30
    assert sample.total_tokens == 150
    assert sample.cached_read_tokens == 40
    assert sample.cache_write_tokens == 10
    assert sample.uncached_prompt_tokens == 80
    assert abs(sample.cache_discount - 0.0012) < 1e-9


def test_clean_json_repairs_common_malformed_wrappers():
    raw = "prefix ```json\n{\"a\":1,}\n``` suffix"
    assert _clean_json(raw) == "{\"a\":1}"


def test_prompt_cache_stats_aggregates_counts():
    stats = PromptCacheStats()
    sample = _extract_prompt_cache_usage(_Completion())
    stats.record("structured.parse", sample)
    snapshot = stats.snapshot()
    assert snapshot["requests_total"] == 1
    assert snapshot["requests_with_usage"] == 1
    assert snapshot["cache_hit_requests"] == 1
    assert snapshot["prompt_tokens"] == 120
    assert snapshot["cached_read_tokens"] == 40
    assert snapshot["cache_write_tokens"] == 10
    assert snapshot["request_type_counts"]["structured.parse"] == 1
