import asyncio

from gpt5_roleplay_system.llm import (
    OpenRouterLLMClient,
    StructuredBundle,
    StructuredFact,
    StructuredFactsOnly,
)
from gpt5_roleplay_system.models import ConversationContext, EnvironmentSnapshot, InboundChat, Participant


class StubFactsClient(OpenRouterLLMClient):
    def __init__(self) -> None:
        super().__init__(api_key="test-key", base_url="http://localhost:1234", model="test-model")
        self.bundle_calls = 0
        self.facts_calls = 0

    def _request_bundle(self, chat, context, overflow, incoming_batch):
        self.bundle_calls += 1
        return StructuredBundle(
            text="ok",
            actions=[],
            participant_hints=[],
            facts=[
                StructuredFact(
                    user_id="user-2",
                    name="Evie",
                    facts=["summary-like fact that should be replaced"],
                )
            ],
            summary_update=None,
        )

    def _request_facts_from_chat(self, chat, context):
        self.facts_calls += 1
        return StructuredFactsOnly(
            facts=[
                StructuredFact(
                    user_id="user-2",
                    name="Evie",
                    facts=["grounded fact from evidence"],
                )
            ]
        )


def _context() -> ConversationContext:
    env = EnvironmentSnapshot(location="Test Zone")
    participants = [Participant(user_id="user-2", name="Evie")]
    recent = [
        InboundChat(
            text="I prefer tea over coffee.",
            sender_id="user-2",
            sender_name="Evie",
            timestamp=1.0,
            raw={},
        )
    ]
    return ConversationContext(
        persona="TestPersona",
        user_id="ai-1",
        environment=env,
        participants=participants,
        people_facts={},
        recent_messages=recent,
        summary="Some summary text that should not drive facts.",
        related_experiences=[{"text": "Experience summary", "metadata": {"source": "episode_summary"}}],
        summary_meta={},
    )


def test_generate_bundle_prefers_facts_only_prompt():
    client = StubFactsClient()
    chat = InboundChat(
        text="hello",
        sender_id="user-1",
        sender_name="User",
        timestamp=2.0,
        raw={},
    )
    bundle = asyncio.run(client.generate_bundle(chat, _context()))
    assert client.bundle_calls == 1
    assert client.facts_calls == 1
    assert bundle.facts
    assert bundle.facts[0].facts == ["grounded fact from evidence"]
