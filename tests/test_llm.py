from gpt5_roleplay_system.llm import OpenRouterLLMClient, StructuredAction, StructuredBundle, _bundle_from_structured
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
