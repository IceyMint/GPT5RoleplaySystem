import asyncio
import json

from gpt5_roleplay_system.llm import (
    OpenRouterLLMClient,
    PromptCacheStats,
    StructuredAction,
    StructuredBundle,
    StructuredStateUpdate,
    _clean_json,
    _bundle_from_structured,
    _extract_prompt_cache_usage,
    _state_update_from_structured,
)
from gpt5_roleplay_system.models import ConversationContext, EnvironmentSnapshot, InboundChat, Participant


def test_mixed_action_type_emits_both_chat_and_primary():
    if StructuredAction is None or StructuredBundle is None:
        return
    action = StructuredAction(
        type="TOUCH",
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
    assert "TOUCH" in types
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


def test_autonomous_bundle_parses_explicit_wait_and_delay():
    if StructuredBundle is None:
        return
    bundle = StructuredBundle(
        text="",
        actions=[],
        autonomy_decision="wait",
        next_delay_seconds=900.0,
    )
    result = _bundle_from_structured(bundle, mode="autonomous")
    assert result.autonomy_decision == "wait"
    assert result.next_delay_seconds == 900.0
    assert result.actions == []


def test_chat_mode_text_only_emits_chat_action():
    if StructuredBundle is None:
        return
    bundle = StructuredBundle(text="Hello there.", actions=[])
    result = _bundle_from_structured(bundle, mode="chat")
    assert result.actions
    assert result.actions[0].command_type.value == "CHAT"
    assert result.actions[0].content == "Hello there."


def test_chat_mode_preserves_scheduler_override_fields():
    if StructuredBundle is None:
        return
    bundle = StructuredBundle(
        text="We should rest now.",
        actions=[],
        autonomy_decision="sleep",
        next_delay_seconds=3600.0,
    )
    result = _bundle_from_structured(bundle, mode="chat")
    assert result.autonomy_decision == "sleep"
    assert result.next_delay_seconds == 3600.0


def test_state_update_parses_scheduler_override_fields():
    if StructuredStateUpdate is None:
        return
    update = StructuredStateUpdate(autonomy_decision="wait", next_delay_seconds=1200.0)
    parsed = _state_update_from_structured(update)
    assert parsed.autonomy_decision == "wait"
    assert parsed.next_delay_seconds == 1200.0


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


def test_structured_action_parses_face_target():
    if StructuredAction is None or StructuredBundle is None:
        return
    action = StructuredAction(type="FACE_TARGET", target_uuid="target-uuid", x=1.5, y=2.5, z=3.5)
    bundle = StructuredBundle(text="", actions=[action])
    result = _bundle_from_structured(bundle, mode="chat")
    assert result.actions
    parsed = result.actions[0]
    assert parsed.command_type.value == "FACE_TARGET"
    assert parsed.target_uuid == "target-uuid"
    assert parsed.parameters.get("x") == "1.5"
    assert parsed.parameters.get("y") == "2.5"
    assert parsed.parameters.get("z") == "3.5"


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


def test_format_address_check_handles_participant_without_username_field():
    client = OpenRouterLLMClient(api_key="test-key", base_url="http://localhost:1234", model="test-model")
    chat = InboundChat(text="hello", sender_id="user-1", sender_name="User", timestamp=1.0, raw={})
    payload = client._format_address_check(
        chat,
        "persona",
        EnvironmentSnapshot(),
        [Participant(user_id="user-2", name="Evie")],
        None,
    )
    decoded = json.loads(payload)
    assert decoded["participants"] == ["Evie"]


def test_bundle_model_override_only_applies_to_bundle_requests():
    if StructuredBundle is None or StructuredStateUpdate is None:
        return

    class CaptureClient(OpenRouterLLMClient):
        def __init__(self) -> None:
            self._model = "default-model"
            self._bundle_model = "bundle-only-model"
            self._max_tokens = 123
            self._temperature = 0.1
            self._reasoning = ""
            self.calls = []

        def _system_prompt_for_context(self, context):
            return "sys"

        def _state_system_prompt_for_context(self, context):
            return "state-sys"

        def _autonomous_system_prompt_for_context(self, context):
            return "autonomy-sys"

        def _format_context(self, chat, context, overflow, incoming_batch):
            return "ctx"

        def _format_autonomous_context(self, context, activity):
            return "autonomy-ctx"

        def _request_structured(self, model_class, kwargs):
            self.calls.append(kwargs["model"])
            return None

    env = EnvironmentSnapshot()
    context = ConversationContext(
        persona="persona",
        user_id="ai-uuid",
        environment=env,
        participants=[],
        people_facts={},
        recent_messages=[],
        summary="",
        related_experiences=[],
        summary_meta={},
        agent_state={},
    )
    chat = InboundChat(text="hello", sender_id="user-1", sender_name="User", timestamp=1.0, raw={})
    client = CaptureClient()

    client._request_bundle(chat, context, None, None)
    client._request_state_update(chat, context, None, None)
    client._request_autonomous_bundle(context, {})

    assert client.calls == ["bundle-only-model", "default-model", "default-model"]


def test_summarize_uses_summary_model():
    class CaptureClient(OpenRouterLLMClient):
        def __init__(self) -> None:
            self._api_key = "test-key"
            self._summary_model = "google/gemini-3-flash-preview"
            self._max_tokens = 111
            self._temperature = 0.5
            self.captured = {}

        def _request_text_with_model(self, model, system_prompt, user_prompt, max_tokens, temperature):
            self.captured = {
                "model": model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            return "summary result"

    client = CaptureClient()
    messages = [InboundChat(text="hello", sender_id="user-1", sender_name="User", timestamp=1.0, raw={})]
    result = asyncio.run(client.summarize("old summary", messages))

    assert result == "summary result"
    assert client.captured["model"] == "google/gemini-3-flash-preview"
    assert client.captured["max_tokens"] == 111
    assert client.captured["temperature"] == 0.5


def test_request_bundle_includes_provider_routing_in_extra_body():
    if StructuredBundle is None:
        return

    class CaptureClient(OpenRouterLLMClient):
        def __init__(self) -> None:
            self._bundle_model = "moonshotai/kimi-k2.5"
            self._max_tokens = 256
            self._temperature = 0.2
            self._reasoning = "low"
            self._provider_order = ["siliconflow"]
            self._provider_allow_fallbacks = False
            self.captured = {}

        def _system_prompt_for_context(self, context):
            return "sys"

        def _format_context(self, chat, context, overflow, incoming_batch):
            return "ctx"

        def _request_structured(self, model_class, kwargs):
            self.captured = kwargs
            return None

    env = EnvironmentSnapshot()
    context = ConversationContext(
        persona="persona",
        user_id="ai-uuid",
        environment=env,
        participants=[],
        people_facts={},
        recent_messages=[],
        summary="",
        related_experiences=[],
        summary_meta={},
        agent_state={},
    )
    chat = InboundChat(text="hello", sender_id="user-1", sender_name="User", timestamp=1.0, raw={})

    client = CaptureClient()
    client._request_bundle(chat, context, None, None)

    assert client.captured["model"] == "moonshotai/kimi-k2.5"
    assert client.captured["extra_body"]["provider"] == {
        "order": ["siliconflow"],
        "allow_fallbacks": False,
    }
    assert client.captured["extra_body"]["include_reasoning"] is True
    assert client.captured["extra_body"]["reasoning_effort"] == "low"


def test_request_facts_uses_facts_model_with_provider_routing():
    class CaptureClient(OpenRouterLLMClient):
        def __init__(self) -> None:
            self._facts_model = "z-ai/glm-5"
            self._max_tokens = 256
            self._reasoning = "low"
            self._provider_order = ["siliconflow"]
            self._provider_allow_fallbacks = False
            self.captured = {}

        def _facts_system_prompt(self):
            return "facts-sys"

        def _format_facts_context_from_messages(
            self,
            evidence_messages,
            participants,
            persona,
            user_id,
            summary,
            summary_meta,
            related_experiences,
            people_facts,
        ):
            return "facts-ctx"

        def _request_structured(self, model_class, kwargs):
            self.captured = kwargs
            return None

    client = CaptureClient()
    msg = InboundChat(text="hello", sender_id="user-1", sender_name="User", timestamp=1.0, raw={})
    client._request_facts_from_messages(
        evidence_messages=[msg],
        participants=[],
        persona="persona",
        user_id="ai-uuid",
    )

    assert client.captured["model"] == "z-ai/glm-5"
    assert client.captured["extra_body"]["provider"] == {
        "order": ["siliconflow"],
        "allow_fallbacks": False,
    }
    assert client.captured["extra_body"]["include_reasoning"] is True


def test_is_addressed_to_me_uses_address_model_with_reasoning():
    class CaptureClient(OpenRouterLLMClient):
        def __init__(self) -> None:
            self._api_key = "test-key"
            self._address_model = "x-ai/grok-4.1-fast"
            self.captured = {}

        def _format_address_check(self, chat, persona, environment, participants, context):
            return "address-check-payload"

        def _request_text_with_model(self, model, system_prompt, user_prompt, max_tokens, temperature):
            self.captured = {
                "model": model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            return "true"

    client = CaptureClient()
    chat = InboundChat(text="hello", sender_id="user-1", sender_name="User", timestamp=1.0, raw={})
    addressed = asyncio.run(client.is_addressed_to_me(chat, "persona"))

    assert addressed is True
    assert client.captured["model"] == "x-ai/grok-4.1-fast"
    assert client.captured["max_tokens"] == 5
    assert client.captured["temperature"] == 0.0


def test_request_text_with_model_includes_reasoning_extra_body():
    class _Message:
        def __init__(self, content: str) -> None:
            self.content = content

    class _Choice:
        def __init__(self, content: str) -> None:
            self.message = _Message(content)

    class _Completion:
        def __init__(self) -> None:
            self.choices = [_Choice("true")]

    class _Completions:
        def __init__(self) -> None:
            self.last_kwargs = {}

        def create(self, **kwargs):
            self.last_kwargs = kwargs
            return _Completion()

    class _Chat:
        def __init__(self, completions) -> None:
            self.completions = completions

    class _Client:
        def __init__(self, completions) -> None:
            self.chat = _Chat(completions)

    completions = _Completions()
    client = object.__new__(OpenRouterLLMClient)
    client._client = _Client(completions)
    client._reasoning = "low"
    client._record_cache_usage = lambda *_args, **_kwargs: None

    response = client._request_text_with_model(
        model="x-ai/grok-4.1-fast",
        system_prompt="sys",
        user_prompt="user",
        max_tokens=5,
        temperature=0.0,
    )

    assert response == "true"
    assert completions.last_kwargs["model"] == "x-ai/grok-4.1-fast"
    assert completions.last_kwargs["extra_body"] == {
        "include_reasoning": True,
        "reasoning_effort": "low",
    }


def test_request_text_with_model_does_not_include_provider_routing():
    class _Message:
        def __init__(self, content: str) -> None:
            self.content = content

    class _Choice:
        def __init__(self, content: str) -> None:
            self.message = _Message(content)

    class _Completion:
        def __init__(self) -> None:
            self.choices = [_Choice("true")]

    class _Completions:
        def __init__(self) -> None:
            self.last_kwargs = {}

        def create(self, **kwargs):
            self.last_kwargs = kwargs
            return _Completion()

    class _Chat:
        def __init__(self, completions) -> None:
            self.completions = completions

    class _Client:
        def __init__(self, completions) -> None:
            self.chat = _Chat(completions)

    completions = _Completions()
    client = object.__new__(OpenRouterLLMClient)
    client._client = _Client(completions)
    client._reasoning = ""
    client._provider_order = ["siliconflow"]
    client._provider_allow_fallbacks = False
    client._record_cache_usage = lambda *_args, **_kwargs: None

    response = client._request_text_with_model(
        model="x-ai/grok-4.1-fast",
        system_prompt="sys",
        user_prompt="user",
        max_tokens=5,
        temperature=0.0,
    )

    assert response == "true"
    assert "extra_body" not in completions.last_kwargs


def test_request_text_with_model_can_include_provider_without_reasoning():
    class _Message:
        def __init__(self, content: str) -> None:
            self.content = content

    class _Choice:
        def __init__(self, content: str) -> None:
            self.message = _Message(content)

    class _Completion:
        def __init__(self) -> None:
            self.choices = [_Choice("true")]

    class _Completions:
        def __init__(self) -> None:
            self.last_kwargs = {}

        def create(self, **kwargs):
            self.last_kwargs = kwargs
            return _Completion()

    class _Chat:
        def __init__(self, completions) -> None:
            self.completions = completions

    class _Client:
        def __init__(self, completions) -> None:
            self.chat = _Chat(completions)

    completions = _Completions()
    client = object.__new__(OpenRouterLLMClient)
    client._client = _Client(completions)
    client._reasoning = "low"
    client._provider_order = ["siliconflow"]
    client._provider_allow_fallbacks = False
    client._record_cache_usage = lambda *_args, **_kwargs: None

    response = client._request_text_with_model(
        model="z-ai/glm-5",
        system_prompt="sys",
        user_prompt="user",
        max_tokens=5,
        temperature=0.0,
        include_reasoning=False,
        include_provider=True,
    )

    assert response == "true"
    assert completions.last_kwargs["extra_body"] == {
        "provider": {
            "order": ["siliconflow"],
            "allow_fallbacks": False,
        }
    }


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
