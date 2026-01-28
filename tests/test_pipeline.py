import asyncio

from gpt5_roleplay_system.config import EpisodeConfig, FactsConfig
from gpt5_roleplay_system.llm import (
    EchoLLMClient,
    ExtractedFact,
    LLMClient,
    LLMResponse,
    LLMResponseBundle,
    ParticipantHint,
)
from gpt5_roleplay_system.memory import ConversationMemory, ExperienceStore, RollingBuffer, SimpleMemoryCompressor
from gpt5_roleplay_system.neo4j_store import InMemoryKnowledgeStore
from gpt5_roleplay_system.observability import NoOpTracer, Tracer
from gpt5_roleplay_system.pipeline import MessagePipeline
from gpt5_roleplay_system.models import Action, CommandType, EnvironmentSnapshot, Participant


class CaptureTracer(Tracer):
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def start_run(self, run_name: str, config=None) -> None:  # pragma: no cover - unused
        return None

    def log_event(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))

    def finish(self) -> None:  # pragma: no cover - unused
        return None


class MoodStatusLLM(LLMClient):
    async def is_addressed_to_me(self, chat, persona, environment=None, participants=None, context=None) -> bool:
        return True

    async def generate_response(self, chat, context) -> LLMResponse:
        text = "ok"
        action = Action(command_type=CommandType.CHAT, content=text, parameters={"content": text})
        return LLMResponse(text=text, actions=[action])

    async def extract_facts(self, chat, context) -> list[ExtractedFact]:
        return []

    async def summarize(self, summary, messages) -> str:
        return summary

    async def generate_bundle(self, chat, context, overflow=None, incoming_batch=None) -> LLMResponseBundle:
        action = Action(command_type=CommandType.CHAT, content="ok", parameters={"content": "ok"})
        return LLMResponseBundle(
            text="ok",
            actions=[action],
            facts=[],
            participant_hints=[],
            summary=None,
            mood="calm",
            status="chatting",
        )

    async def generate_autonomous_bundle(self, context, activity) -> LLMResponseBundle:
        action = Action(
            command_type=CommandType.EMOTE,
            content="looks around quietly.",
            parameters={"content": "looks around quietly."},
        )
        return LLMResponseBundle(
            text="",
            actions=[action],
            facts=[],
            participant_hints=[],
            summary=None,
            mood="focused",
            status="scanning the area",
        )


class AutonomyTextOnlyLLM(LLMClient):
    async def is_addressed_to_me(self, chat, persona, environment=None, participants=None, context=None) -> bool:
        return True

    async def generate_response(self, chat, context) -> LLMResponse:  # pragma: no cover - unused
        raise NotImplementedError

    async def extract_facts(self, chat, context) -> list[ExtractedFact]:  # pragma: no cover - unused
        return []

    async def summarize(self, summary, messages) -> str:  # pragma: no cover - unused
        return summary

    async def generate_autonomous_bundle(self, context, activity) -> LLMResponseBundle:
        return LLMResponseBundle(
            text="I'll wait a bit longer to see if Isabella has more to say.",
            actions=[],
            facts=[],
            participant_hints=[],
            summary=None,
            mood="patient",
            status="waiting",
        )


class OverflowTypeLLM(LLMClient):
    async def is_addressed_to_me(self, chat, persona, environment=None, participants=None, context=None) -> bool:
        return True

    async def generate_response(self, chat, context) -> LLMResponse:  # pragma: no cover - unused
        raise NotImplementedError

    async def extract_facts(self, chat, context) -> list[ExtractedFact]:  # pragma: no cover - unused
        return []

    async def summarize(self, summary, messages) -> str:
        return summary

    async def generate_bundle(self, chat, context, overflow=None, incoming_batch=None) -> LLMResponseBundle:
        if overflow:
            # Regression check: overflow must be chat-like, not MemoryItem.
            assert all(hasattr(item, "raw") for item in overflow)
        return LLMResponseBundle(
            text="",
            actions=[],
            facts=[],
            participant_hints=[],
            summary=None,
        )


def test_pipeline_returns_action():
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
    )

    actions = asyncio.run(
        pipeline.process_chat(
            {
                "text": "Hello",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 1,
            }
        )
    )
    assert actions
    assert actions[0].content.startswith("Echo:")


def test_prompt_bundle_includes_timestamps():
    tracer = CaptureTracer()
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=tracer,
        facts_config=FactsConfig(mode="periodic", interval_seconds=1, evidence_max_messages=12, in_bundle=False),
    )

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "Hello with time",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 123.0,
            }
        )
    )

    prompt_payloads = [payload for name, payload in tracer.events if name == "llm_prompt_bundle"]
    assert prompt_payloads
    payload = prompt_payloads[-1]
    assert "now_timestamp" in payload
    assert "recent_time_range" in payload
    assert payload["incoming"]["timestamp"] == 123.0
    assert payload["incoming_sender_known"] is False
    assert payload["incoming_sender_id"] == "user-1"
    assert payload["recent_messages"]
    assert payload["recent_messages"][-1]["timestamp"] == 123.0
    assert payload["incoming_batch"]
    assert payload["incoming_batch"][-1]["timestamps"] == [123.0]
    assert "agent_state" in payload
    assert payload["agent_state"]["mood"]
    assert payload["agent_state"]["status"]


def test_prompt_bundle_includes_persona_instructions():
    tracer = CaptureTracer()
    pipeline = MessagePipeline(
        persona="isabella.elara",
        user_id="ai-1",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=tracer,
        persona_profiles={"isabella.elara": "Be playful, curious, and brief."},
    )

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "Hello with persona",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 123.0,
            }
        )
    )

    prompt_payloads = [payload for name, payload in tracer.events if name == "llm_prompt_bundle"]
    assert prompt_payloads
    payload = prompt_payloads[-1]
    assert payload.get("persona_instructions") == "Be playful, curious, and brief."


def test_prompt_bundle_never_labels_ai_as_ai():
    tracer = CaptureTracer()
    pipeline = MessagePipeline(
        persona="isabella.elara",
        user_id="ai-uuid",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=10, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=10),
        experience_store=ExperienceStore(),
        tracer=tracer,
    )

    # Seed an AI-authored message into memory/rolling buffer.
    pipeline._rolling_buffer.add_ai_message("Testing persona label", pipeline._persona)  # type: ignore[attr-defined]
    pipeline._memory.add_ai_message("Testing persona label", pipeline._persona)  # type: ignore[attr-defined]

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "Hello",
                "from_name": "Evie",
                "from_id": "user-1",
                "timestamp": 1.0,
            }
        )
    )

    prompt_payloads = [payload for name, payload in tracer.events if name == "llm_prompt_bundle"]
    assert prompt_payloads
    payload = prompt_payloads[-1]
    recent_senders = [msg.get("sender") for msg in payload.get("recent_messages", [])]
    assert "ai" not in {str(sender) for sender in recent_senders}
    assert "isabella.elara" in {str(sender) for sender in recent_senders}


def test_autonomy_updates_mood_and_status_snapshot():
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=MoodStatusLLM(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
    )
    pipeline._last_inbound_ts = 0.0
    pipeline._last_response_ts = 0.0

    asyncio.run(pipeline.generate_autonomous_actions(recent_activity_window_seconds=1.0))
    snapshot = pipeline.activity_snapshot(1.0)
    assert snapshot["mood"] == "focused"
    assert snapshot["status"] == "scanning the area"
    assert snapshot["status_source"] == "autonomy"


def test_autonomy_text_only_does_not_add_ai_messages():
    pipeline = MessagePipeline(
        persona="Isabella",
        user_id="ai-1",
        llm=AutonomyTextOnlyLLM(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
    )
    pipeline._last_inbound_ts = 0.0
    pipeline._last_response_ts = 0.0

    actions = asyncio.run(pipeline.generate_autonomous_actions(recent_activity_window_seconds=1.0))
    assert actions == []
    assert pipeline._memory.recent() == []
    assert pipeline._rolling_buffer.items() == []


def test_autonomy_filter_uses_persona_name_not_uuid():
    pipeline = MessagePipeline(
        persona="isabella.elara",
        user_id="c4cc1a90-b6de-4ec4-8649-5a06b9af1bfa",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
    )

    actions = [
        Action(
            command_type=CommandType.EMOTE,
            content="isabella.elara is waiting for herself to respond.",
            parameters={"content": "isabella.elara is waiting for herself to respond."},
        )
    ]

    filtered = pipeline._filter_autonomous_actions(actions, participants=[])  # type: ignore[attr-defined]
    assert filtered == []


def test_people_facts_include_last_seen():
    tracer = CaptureTracer()
    store = InMemoryKnowledgeStore()
    store.upsert_person_facts("user-2", "Evie", [])
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=EchoLLMClient(),
        knowledge_store=store,
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=tracer,
        facts_config=FactsConfig(mode="periodic", interval_seconds=1, evidence_max_messages=12, in_bundle=False),
    )

    pipeline.update_environment(
        {
            "agents": [{"name": "Evie", "uuid": "user-2"}],
            "objects": [],
            "location": "Test Zone",
            "timestamp": 200.0,
        }
    )

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "Hello again",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 210.0,
                "participants": [{"user_id": "user-2", "name": "Evie"}],
            }
        )
    )

    prompt_payloads = [payload for name, payload in tracer.events if name == "llm_prompt_bundle"]
    assert prompt_payloads
    payload = prompt_payloads[-1]
    evie = payload["people_facts"]["user-2"]
    assert evie["last_seen_ts"] == 200.0
    assert evie["last_seen_seconds_ago"] is not None
    assert evie["last_seen_seconds_ago"] >= 0.0


def test_pipeline_ignores_self_message():
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
    )

    actions = asyncio.run(
        pipeline.process_chat(
            {
                "text": "I am the AI speaking",
                "from_name": "TestPersona",
                "from_id": "ai-1",
                "timestamp": 2,
                "logged_in_agent": "TestPersona",
            }
        )
    )
    assert actions == []


def test_pipeline_ignores_self_message_by_persona_name_when_uuid_mismatches():
    pipeline = MessagePipeline(
        persona="isabella.elara",
        user_id="ai-config-id",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
    )

    stylized_self = "ღIѕαвєℓℓαღ (isabella.elara)"
    actions = asyncio.run(
        pipeline.process_chat(
            {
                "text": "Evie Hi",
                "from_name": stylized_self,
                "from_id": "viewer-self-uuid",
                "logged_in_agent": stylized_self,
                "persona": stylized_self,
                "timestamp": 2,
            }
        )
    )

    assert actions == []
    recent = pipeline._memory.recent()  # type: ignore[attr-defined]
    assert recent, "self message should still be seeded into memory"
    assert recent[-1].sender_name == "isabella.elara"
    assert recent[-1].sender_id == "ai"


class ParticipantCaptureLLM(EchoLLMClient):
    def __init__(self) -> None:
        super().__init__()
        self.seen_participants: list[Participant] = []
        self.seen_environment: EnvironmentSnapshot | None = None

    async def is_addressed_to_me(
        self,
        chat,
        persona,
        environment=None,
        participants=None,
        context=None,
    ) -> bool:
        if participants:
            self.seen_participants = list(participants)
        self.seen_environment = environment
        return True


def test_pipeline_filters_ai_from_participants():
    llm = ParticipantCaptureLLM()
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
    )

    pipeline.update_environment(
        {
            "agents": [
                {"name": "TestPersona", "uuid": "ai-1"},
                {"name": "Other User", "uuid": "user-2"},
            ],
            "objects": [],
            "location": "Test Zone",
        }
    )

    actions = asyncio.run(
        pipeline.process_chat(
            {
                "text": "Hey Other User",
                "from_name": "Other User",
                "from_id": "user-2",
                "timestamp": 3,
            }
        )
    )

    assert actions
    assert llm.seen_participants
    assert all(p.name != "TestPersona" for p in llm.seen_participants)


def test_pipeline_filters_stylized_self_from_participants():
    llm = ParticipantCaptureLLM()
    pipeline = MessagePipeline(
        persona="Isabella",
        user_id="self-uuid",
        llm=llm,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
    )

    stylized_self = "ღIѕαвєℓℓαღ (isabella.elara)"
    pipeline.update_environment(
        {
            "agents": [
                {"name": stylized_self, "uuid": "self-uuid"},
                {"name": "Evie", "uuid": "user-2"},
            ],
            "objects": [],
            "location": "Test Zone",
        }
    )

    actions = asyncio.run(
        pipeline.process_chat(
            {
                "text": "Hey Isabella",
                "from_name": "Evie",
                "from_id": "user-2",
                "timestamp": 3,
            }
        )
    )

    assert actions
    assert llm.seen_participants
    assert all(p.user_id != "self-uuid" for p in llm.seen_participants)


class HintLLM(LLMClient):
    def __init__(self) -> None:
        self.seen_participants: list[Participant] = []
        self._sent_hint = False

    async def is_addressed_to_me(
        self,
        chat,
        persona,
        environment=None,
        participants=None,
        context=None,
    ) -> bool:
        if participants:
            self.seen_participants = list(participants)
        return True

    async def generate_response(self, chat, context):
        raise NotImplementedError

    async def extract_facts(self, chat, context):
        return []

    async def summarize(self, summary, messages):
        return summary

    async def generate_bundle(self, chat, context, overflow=None, incoming_batch=None):
        hints = []
        if not self._sent_hint:
            hints = [ParticipantHint(user_id="", name="Alice")]
            self._sent_hint = True
        return LLMResponseBundle(
            text="",
            actions=[],
            facts=[],
            participant_hints=hints,
            summary=None,
        )


def test_pipeline_applies_participant_hints():
    llm = HintLLM()
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
    )

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "I talked to Alice earlier.",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 4,
            }
        )
    )

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "What do you think?",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 5,
            }
        )
    )

    assert any(p.name == "Alice" for p in llm.seen_participants)


class ContextCaptureLLM(LLMClient):
    def __init__(self) -> None:
        self.contexts = []

    async def is_addressed_to_me(
        self,
        chat,
        persona,
        environment=None,
        participants=None,
        context=None,
    ) -> bool:
        return True

    async def generate_response(self, chat, context):
        raise NotImplementedError

    async def extract_facts(self, chat, context):
        return []

    async def summarize(self, summary, messages):
        return summary

    async def generate_bundle(self, chat, context, overflow=None, incoming_batch=None):
        self.contexts.append(context)
        return LLMResponseBundle(
            text="",
            actions=[],
            facts=[],
            participant_hints=[],
            summary=None,
        )


class FactBundleLLM(ContextCaptureLLM):
    def __init__(self, facts) -> None:
        super().__init__()
        self._facts = facts

    async def generate_bundle(self, chat, context, overflow=None, incoming_batch=None):
        self.contexts.append(context)
        return LLMResponseBundle(
            text="",
            actions=[],
            facts=self._facts,
            participant_hints=[],
            summary=None,
        )


class CountingStore(InMemoryKnowledgeStore):
    def __init__(self) -> None:
        super().__init__()
        self.upsert_calls: list[tuple[str, str, list[str]]] = []

    def upsert_person_facts(self, user_id: str, name: str, facts: list[str]) -> None:
        self.upsert_calls.append((user_id, name, list(facts)))
        super().upsert_person_facts(user_id, name, facts)


class PeriodicFactsLLM(ContextCaptureLLM):
    def __init__(self, facts: list[ExtractedFact]) -> None:
        super().__init__()
        self._facts = facts
        self.evidence_calls: list[list[str]] = []

    def extract_facts_from_evidence_sync(self, context, evidence_messages, participants):
        self.evidence_calls.append([m.text for m in evidence_messages])
        return list(self._facts)


class DuplicateFactsStore(InMemoryKnowledgeStore):
    """Test double that simulates a backing store returning duplicate facts."""

    def fetch_people(self, user_ids):
        profiles = super().fetch_people(user_ids)
        for profile in profiles.values():
            profile.facts = list(profile.facts) + list(profile.facts)
        return profiles


def test_pipeline_dedupes_people_facts_from_store():
    llm = ContextCaptureLLM()
    store = DuplicateFactsStore()
    store.upsert_person_facts("user-2", "Evie", ["fact-a", "fact-b"])
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm,
        knowledge_store=store,
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
    )

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "Hello there",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 7,
                "participants": [{"user_id": "user-2", "name": "Evie"}],
            }
        )
    )

    assert llm.contexts
    people_facts = llm.contexts[-1].people_facts
    assert people_facts["user-2"]["facts"] == ["fact-a", "fact-b"]


def test_store_facts_skips_existing_duplicates():
    store = CountingStore()
    store.upsert_person_facts("user-2", "Evie", ["fact-a"])
    store.upsert_calls.clear()
    llm = FactBundleLLM(
        [ExtractedFact(user_id="user-2", name="Evie", facts=["fact-a", "fact-a"])]
    )
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm,
        knowledge_store=store,
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
    )

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "Hello Evie",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 8,
                "participants": [{"user_id": "user-2", "name": "Evie"}],
            }
        )
    )

    assert store.upsert_calls == []


def test_periodic_fact_sweep_stores_facts_async():
    store = CountingStore()
    llm = PeriodicFactsLLM(
        [ExtractedFact(user_id="user-2", name="Evie", facts=["likes tea"])]
    )
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm,
        knowledge_store=store,
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        facts_config=FactsConfig(mode="periodic", interval_seconds=1, evidence_max_messages=12, in_bundle=False),
    )

    async def run_once():
        await pipeline.process_chat(
            {
                "text": "Hello there",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 9,
                "participants": [{"user_id": "user-2", "name": "Evie"}],
            }
        )
        await asyncio.sleep(0.05)

    asyncio.run(run_once())

    profile = store.fetch_people(["user-2"]).get("user-2")
    assert profile is not None
    assert "likes tea" in profile.facts
    assert store.upsert_calls


def test_periodic_fact_sweep_includes_overflow_messages():
    store = CountingStore()
    llm = PeriodicFactsLLM(
        [ExtractedFact(user_id="user-2", name="Evie", facts=["remembers overflow"])]
    )
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm,
        knowledge_store=store,
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=2, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        facts_config=FactsConfig(mode="periodic", interval_seconds=1, evidence_max_messages=12, in_bundle=False),
    )

    batch = [
        {"text": "one", "from_name": "User", "from_id": "user-1", "timestamp": 10},
        {"text": "two", "from_name": "User", "from_id": "user-1", "timestamp": 11},
        {"text": "three", "from_name": "User", "from_id": "user-1", "timestamp": 12},
    ]

    async def run_batch():
        await pipeline.process_chat_batch(batch)
        await asyncio.sleep(0.05)

    asyncio.run(run_batch())

    assert any("one" in call for call in llm.evidence_calls)


def test_overflow_passed_to_llm_is_chat_like():
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=OverflowTypeLLM(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=2, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
    )

    batch = [
        {"text": "one", "from_name": "User", "from_id": "user-1", "timestamp": 10},
        {"text": "two", "from_name": "User", "from_id": "user-1", "timestamp": 11},
        {"text": "three", "from_name": "User", "from_id": "user-1", "timestamp": 12},
    ]

    asyncio.run(pipeline.process_chat_batch(batch))


def test_periodic_fact_sweep_logs_tracer_event():
    tracer = CaptureTracer()
    llm = PeriodicFactsLLM(
        [ExtractedFact(user_id="user-2", name="Evie", facts=["likes tea"])]
    )
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=tracer,
        facts_config=FactsConfig(mode="periodic", interval_seconds=1, evidence_max_messages=12, in_bundle=False),
    )

    async def run_once():
        await pipeline.process_chat(
            {
                "text": "Hello there",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 13,
                "participants": [{"user_id": "user-2", "name": "Evie"}],
            }
        )
        await asyncio.sleep(0.05)

    asyncio.run(run_once())

    facts_events = [payload for name, payload in tracer.events if name == "facts_sweep"]
    assert facts_events
    assert any(event.get("fact_strings_stored", 0) >= 1 for event in facts_events)


def test_pipeline_partial_name_fallback_marks_context():
    llm = ContextCaptureLLM()
    store = InMemoryKnowledgeStore()
    store.upsert_person_facts("user-99", "Alex Johnson", ["likes chess"])
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm,
        knowledge_store=store,
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
    )

    pipeline.update_environment(
        {
            "agents": [{"name": "Alex", "uuid": ""}],
            "objects": [],
            "location": "Test Zone",
        }
    )

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "Hey Alex, are you around?",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 6,
            }
        )
    )

    assert llm.contexts
    people_facts = llm.contexts[-1].people_facts
    assert people_facts
    assert any(
        profile.get("match_type") == "partial_name" for profile in people_facts.values()
    )


def test_token_similarity_search_returns_relevant():
    from gpt5_roleplay_system.memory import ExperienceStore, TokenSimilaritySearch

    store = ExperienceStore()
    store.add("We met Alex at the cafe", {"id": 1})
    store.add("The sky is blue today", {"id": 2})
    search = TokenSimilaritySearch()
    results = search.search("Alex cafe", store.all(), top_k=2)
    assert results
    assert results[0].metadata["id"] == 1


def test_experience_vector_index_semantic_search():
    from gpt5_roleplay_system.experience_vector import EmbeddingClient, ExperienceVectorIndex
    from gpt5_roleplay_system.memory import ExperienceRecord

    class FakeEmbedder(EmbeddingClient):
        def embed(self, texts):
            vectors = []
            for text in texts:
                lower = text.lower()
                vectors.append(
                    [
                        1.0 if "alex" in lower else 0.0,
                        1.0 if "cafe" in lower else 0.0,
                    ]
                )
            return vectors

    index = ExperienceVectorIndex(embedder=FakeEmbedder(), max_items=10)
    record_a = ExperienceRecord(text="We met Alex at the cafe", metadata={"id": "a"})
    record_b = ExperienceRecord(text="It rained all day", metadata={"id": "b"})

    asyncio.run(index.add_record_async(record_a, "TestPersona"))
    asyncio.run(index.add_record_async(record_b, "TestPersona"))

    results = asyncio.run(index.search("Alex cafe", persona_id="TestPersona", top_k=2))
    assert results
    assert results[0].metadata["id"] == "a"


def test_semantic_experience_score_gating():
    from gpt5_roleplay_system.memory import ExperienceRecord

    class FakeVectorIndex:
        def is_enabled(self) -> bool:
            return True

        async def add_record_async(self, record: ExperienceRecord, persona_id: str) -> None:
            return None

        async def search(self, query: str, persona_id: str, top_k: int = 3):
            return [
                ExperienceRecord(text="top", metadata={"experience_id": "e1", "score": 0.80}),
                ExperienceRecord(text="near", metadata={"experience_id": "e2", "score": 0.79}),
                ExperienceRecord(text="low", metadata={"experience_id": "e3", "score": 0.70}),
            ][:top_k]

    llm = ContextCaptureLLM()
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        experience_vector_index=FakeVectorIndex(),
        experience_score_min=0.78,
        experience_score_delta=0.03,
    )

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "Tell me about dinner",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 7,
            }
        )
    )

    assert llm.contexts
    related = llm.contexts[-1].related_experiences
    texts = [item.get("text") for item in related]
    assert "top" in texts
    assert "near" in texts
    assert "low" not in texts


def test_pipeline_logs_llm_prompt_and_response():
    tracer = CaptureTracer()
    pipeline = MessagePipeline(
        persona="TracePersona",
        user_id="ai-1",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=tracer,
    )

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "Hello there",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 42,
            }
        )
    )

    event_names = [name for name, _ in tracer.events]
    assert "llm_prompt_bundle" in event_names
    assert "llm_response_bundle" in event_names


def test_experience_created_on_episode_boundary():
    pipeline = MessagePipeline(
        persona="EpisodePersona",
        user_id="ai-1",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=20, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=20),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        episode_config=EpisodeConfig(
            enabled=True,
            min_messages=2,
            max_messages=3,
            inactivity_seconds=0,
            forced_interval_seconds=0,
            overlap_messages=1,
        ),
    )

    batch = [
        {"text": "one", "from_name": "User", "from_id": "user-1", "timestamp": 1},
        {"text": "two", "from_name": "User", "from_id": "user-1", "timestamp": 2},
        {"text": "three", "from_name": "User", "from_id": "user-1", "timestamp": 3},
    ]

    async def run_batch():
        await pipeline.process_chat_batch(batch)
        # Allow the scheduled episode task to complete.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(run_batch())

    experiences = pipeline._experience_store.all()  # type: ignore[attr-defined]
    assert len(experiences) == 1
    summary_text = experiences[0].text
    assert "one" in summary_text
    assert "two" in summary_text
