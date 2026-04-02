import asyncio
import time
from unittest.mock import patch

from gpt5_roleplay_system.config import EpisodeConfig, FactsConfig
from gpt5_roleplay_system.llm import (
    EchoLLMClient,
    ExtractedFact,
    LLMClient,
    LLMResponse,
    LLMResponseBundle,
    LLMStateUpdate,
    ParticipantHint,
)
from gpt5_roleplay_system.memory import ConversationMemory, ExperienceStore, RollingBuffer, SimpleMemoryCompressor
from gpt5_roleplay_system.neo4j_store import InMemoryKnowledgeStore
from gpt5_roleplay_system.observability import NoOpTracer, Tracer
from gpt5_roleplay_system.payload_contract import normalize_and_validate_payload
from gpt5_roleplay_system.pipeline import MessagePipeline
from gpt5_roleplay_system.models import Action, CommandType, EnvironmentSnapshot, InboundChat, Participant

IGNORED_USER_ID = "00000000-0000-0000-0000-000000000000"


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

    async def summarize_overflow(self, summary, messages) -> str:
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

    async def summarize_overflow(self, summary, messages) -> str:  # pragma: no cover - unused
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
            autonomy_decision="wait",
            next_delay_seconds=900.0,
        )


class OverflowTypeLLM(LLMClient):
    async def is_addressed_to_me(self, chat, persona, environment=None, participants=None, context=None) -> bool:
        return True

    async def generate_response(self, chat, context) -> LLMResponse:  # pragma: no cover - unused
        raise NotImplementedError

    async def extract_facts(self, chat, context) -> list[ExtractedFact]:  # pragma: no cover - unused
        return []

    async def summarize_overflow(self, summary, messages) -> str:
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


class QualityFallbackEventLLM(EchoLLMClient):
    def __init__(self) -> None:
        super().__init__()
        self._quality_events: list[dict] = []

    async def generate_bundle(self, chat, context, overflow=None, incoming_batch=None) -> LLMResponseBundle:
        self._quality_events.append(
            {
                "mode": "bundle",
                "reason": "test_fallback",
                "fallback_client": "RuleBasedLLMClient",
                "model": "test-model",
                "timestamp": 1.0,
            }
        )
        return await super().generate_bundle(chat, context, overflow, incoming_batch)

    def consume_quality_fallback_events(self) -> list[dict]:
        events = list(self._quality_events)
        self._quality_events = []
        return events


class IncomingSchedulerHintLLM(LLMClient):
    async def is_addressed_to_me(self, chat, persona, environment=None, participants=None, context=None) -> bool:
        return True

    async def generate_response(self, chat, context) -> LLMResponse:  # pragma: no cover - unused
        raise NotImplementedError

    async def extract_facts(self, chat, context) -> list[ExtractedFact]:  # pragma: no cover - unused
        return []

    async def summarize_overflow(self, summary, messages) -> str:  # pragma: no cover - unused
        return summary

    async def generate_bundle(self, chat, context, overflow=None, incoming_batch=None) -> LLMResponseBundle:
        return LLMResponseBundle(
            text="",
            actions=[],
            facts=[],
            participant_hints=[],
            summary=None,
            autonomy_decision="sleep",
            next_delay_seconds=3600.0,
        )

    async def generate_state_update(
        self,
        chat,
        context,
        overflow=None,
        incoming_batch=None,
    ) -> LLMStateUpdate:
        return LLMStateUpdate(
            facts=[],
            participant_hints=[],
            summary_update=None,
            mood="quiet",
            status="resting",
            autonomy_decision="wait",
            next_delay_seconds=1200.0,
        )


class NeverAddressedLLM(LLMClient):
    async def is_addressed_to_me(self, chat, persona, environment=None, participants=None, context=None) -> bool:
        return False

    async def generate_response(self, chat, context) -> LLMResponse:  # pragma: no cover - unused
        raise NotImplementedError

    async def extract_facts(self, chat, context) -> list[ExtractedFact]:  # pragma: no cover - unused
        return []

    async def summarize_overflow(self, summary, messages) -> str:
        return ""

    async def summarize_episode(self, messages) -> str:
        return "Never addressed episode containing: " + " ".join(m.text for m in messages)


class AddressedEmptySummaryLLM(LLMClient):
    async def is_addressed_to_me(self, chat, persona, environment=None, participants=None, context=None) -> bool:
        return True

    async def generate_response(self, chat, context) -> LLMResponse:  # pragma: no cover - unused
        raise NotImplementedError

    async def extract_facts(self, chat, context) -> list[ExtractedFact]:  # pragma: no cover - unused
        return []

    async def summarize_overflow(self, summary, messages) -> str:
        return ""

    async def generate_bundle(self, chat, context, overflow=None, incoming_batch=None) -> LLMResponseBundle:
        return LLMResponseBundle(
            text="",
            actions=[],
            facts=[],
            participant_hints=[],
            summary=None,
        )


class SummaryCaptureNeverAddressedLLM(LLMClient):
    def __init__(self) -> None:
        self.summarize_calls: list[list[str]] = []

    async def is_addressed_to_me(self, chat, persona, environment=None, participants=None, context=None) -> bool:
        return False

    async def generate_response(self, chat, context) -> LLMResponse:  # pragma: no cover - unused
        raise NotImplementedError

    async def extract_facts(self, chat, context) -> list[ExtractedFact]:  # pragma: no cover - unused
        return []

    async def summarize_overflow(self, summary, messages) -> str:
        self.summarize_calls.append([str(msg.text) for msg in messages])
        return summary or "summary"


class SummarySenderCaptureLLM(LLMClient):
    def __init__(self) -> None:
        self.summarize_calls: list[list[dict[str, str]]] = []

    async def is_addressed_to_me(self, chat, persona, environment=None, participants=None, context=None) -> bool:
        return False

    async def generate_response(self, chat, context) -> LLMResponse:  # pragma: no cover - unused
        raise NotImplementedError

    async def extract_facts(self, chat, context) -> list[ExtractedFact]:  # pragma: no cover - unused
        return []

    async def summarize_overflow(self, summary, messages) -> str:
        self.summarize_calls.append(
            [
                {
                    "sender_id": str(msg.sender_id or ""),
                    "sender_name": str(msg.sender_name or ""),
                    "text": str(msg.text or ""),
                }
                for msg in messages
            ]
        )
        return summary or "summary"


class ReasoningTraceLLM(EchoLLMClient):
    def __init__(self) -> None:
        super().__init__()
        self._consumed = False

    def consume_reasoning_trace(self, label: str):
        if label != "bundle" or self._consumed:
            return None
        self._consumed = True
        return {
            "label": "bundle",
            "request_type": "structured.parse",
            "model": "moonshotai/kimi-k2.5",
            "include_reasoning": True,
            "reasoning_effort": "low",
            "reasoning_tokens": 64,
            "prompt_tokens": 1200,
            "completion_tokens": 220,
            "total_tokens": 1420,
            "reasoning_text": "The message directly addresses the persona.",
            "reasoning_details": [{"type": "reasoning.text", "text": "Direct mention"}],
            "has_reasoning": True,
        }



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
        facts_config=FactsConfig(
            mode="periodic",
            interval_seconds=1,
            evidence_max_messages=12,
            in_bundle=False,
            min_pending_messages=1,
        ),
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
    assert isinstance(payload["incoming"]["timestamp"], str)
    assert payload["incoming"]["timestamp"]
    assert payload["incoming_sender_known"] is False
    assert payload["incoming_sender_id"] == "user-1"
    assert payload["recent_messages"] == []
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


def test_prompt_bundle_environment_entities_use_uuid_only():
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
    )
    pipeline.update_environment(
        {
            "location": "Test Zone",
            "agents": [
                {"target_uuid": "agent-b", "name": "B"},
                {"uuid": "agent-a", "target_uuid": "agent-a", "name": "A"},
            ],
            "objects": [
                {"target_uuid": "obj-b", "name": "B", "distance": 1.2},
                {"uuid": "obj-a", "target_uuid": "obj-a", "name": "A", "distance": 5.0},
            ],
        }
    )

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "hello",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 123.0,
            }
        )
    )

    prompt_payloads = [payload for name, payload in tracer.events if name == "llm_prompt_bundle"]
    assert prompt_payloads
    environment = prompt_payloads[-1]["environment"]
    assert [agent.get("uuid") for agent in environment["agents"]] == ["agent-b", "agent-a"]
    assert all("target_uuid" not in agent for agent in environment["agents"])
    assert [obj.get("uuid") for obj in environment["objects"]] == ["obj-b", "obj-a"]
    assert all("target_uuid" not in obj for obj in environment["objects"])


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
    assert pipeline.consume_autonomy_delay_hint_seconds() == 900.0
    assert pipeline.consume_autonomy_delay_hint_seconds() is None


def test_chat_bundle_can_override_autonomy_scheduler_hint():
    pipeline = MessagePipeline(
        persona="Isabella",
        user_id="ai-1",
        llm=IncomingSchedulerHintLLM(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
    )

    actions = asyncio.run(
        pipeline.process_chat(
            {
                "text": "Sleep well tonight.",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 10.0,
            }
        )
    )
    assert actions == []
    assert pipeline.activity_snapshot(1.0)["autonomy_decision"] == "sleep"
    assert pipeline.consume_autonomy_delay_hint_seconds() == 3600.0


def test_chat_disabled_does_not_override_autonomy_scheduler_hint():
    pipeline = MessagePipeline(
        persona="Isabella",
        user_id="ai-1",
        llm=IncomingSchedulerHintLLM(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
    )
    pipeline.set_llm_chat_enabled(False)

    actions = asyncio.run(
        pipeline.process_chat(
            {
                "text": "No need to reply, just rest.",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 20.0,
            }
        )
    )
    assert actions == []
    assert pipeline.activity_snapshot(1.0)["autonomy_decision"] == "wait"
    assert pipeline.consume_autonomy_delay_hint_seconds() is None


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


def test_prompt_payload_moves_last_seen_into_live_people_recency():
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
        facts_config=FactsConfig(
            mode="periodic",
            interval_seconds=1,
            evidence_max_messages=12,
            in_bundle=False,
            min_pending_messages=1,
        ),
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
    assert "last_seen_ts" not in evie
    assert "last_seen_seconds_ago" not in evie
    recency = payload["people_recency"]["user-2"]
    assert recency["last_seen_seconds_ago"] >= 0
    assert recency["last_seen_bucket"]


def test_prompt_payload_marks_reappearance_after_long_absence():
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
    )

    base_ts = time.time()
    pipeline.update_environment(
        {
            "agents": [{"name": "Evie", "uuid": "user-2"}],
            "objects": [],
            "location": "Test Zone",
            "timestamp": base_ts - 400.0,
        }
    )
    pipeline.update_environment(
        {
            "agents": [],
            "objects": [],
            "location": "Test Zone",
            "timestamp": base_ts - 399.0,
        }
    )
    pipeline.update_environment(
        {
            "agents": [{"name": "Evie", "uuid": "user-2"}],
            "objects": [],
            "location": "Test Zone",
            "timestamp": base_ts,
        }
    )

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "Hello again",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": base_ts + 10.0,
                "participants": [{"user_id": "user-2", "name": "Evie"}],
            }
        )
    )

    prompt_payloads = [payload for name, payload in tracer.events if name == "llm_prompt_bundle"]
    assert prompt_payloads
    recency = prompt_payloads[-1]["people_recency"]["user-2"]
    assert recency["reappeared_after_seconds"] >= 300
    assert recency["reappeared_after_bucket"] in {"5-15m", "15-60m"}


def test_prompt_payload_marks_mentioned_absent_person_as_seen_today():
    tracer = CaptureTracer()
    store = InMemoryKnowledgeStore()
    store.upsert_person_facts("user-42", "Kei", ["likes tea"])
    base_ts = 1_000_000.0
    store.update_last_seen("user-42", "Kei", base_ts - 1800.0)
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=EchoLLMClient(),
        knowledge_store=store,
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=tracer,
    )

    with patch("gpt5_roleplay_system.pipeline.time.time", return_value=base_ts):
        asyncio.run(
            pipeline.process_chat(
                {
                    "text": "Did you see Kei today?",
                    "from_name": "User",
                    "from_id": "user-1",
                    "timestamp": base_ts,
                }
            )
        )

    prompt_payloads = [payload for name, payload in tracer.events if name == "llm_prompt_bundle"]
    assert prompt_payloads
    payload = prompt_payloads[-1]
    assert payload["people_facts"]["user-42"]["match_type"] == "text_mention"
    recency = payload["people_recency"]["user-42"]
    assert recency["last_seen_day_relation"] == "today"
    assert recency["last_seen_seconds_ago"] == 1800


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


def test_pipeline_ignores_configured_user_id_message():
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
    )

    actions = asyncio.run(
        pipeline.process_chat(
            {
                "text": "system message",
                "from_name": "System",
                "from_id": IGNORED_USER_ID,
                "timestamp": 2,
            }
        )
    )

    assert actions == []
    assert pipeline._memory.recent() == []  # type: ignore[attr-defined]
    assert pipeline._rolling_buffer.items() == []  # type: ignore[attr-defined]
    assert not [payload for name, payload in tracer.events if name == "llm_prompt_bundle"]


def test_pipeline_ignores_configured_user_id_in_mixed_batch():
    tracer = CaptureTracer()
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=tracer,
    )

    actions = asyncio.run(
        pipeline.process_chat_batch(
            [
                {
                    "text": "system message",
                    "from_name": "System",
                    "from_id": IGNORED_USER_ID,
                    "timestamp": 2,
                },
                {
                    "text": "hello",
                    "from_name": "Evie",
                    "from_id": "user-2",
                    "timestamp": 3,
                },
            ]
        )
    )

    assert [action.content for action in actions] == ["Echo: hello"]
    recent = pipeline._memory.recent()  # type: ignore[attr-defined]
    user_messages = [item for item in recent if item.sender_id != "ai"]
    assert len(user_messages) == 1
    assert user_messages[0].sender_id == "user-2"
    assert user_messages[0].text == "hello"

    prompt_payloads = [payload for name, payload in tracer.events if name == "llm_prompt_bundle"]
    assert prompt_payloads
    incoming_batch = prompt_payloads[-1]["incoming_batch"]
    assert [entry.get("sender_id") for entry in incoming_batch] == ["user-2"]
    participant_ids = {entry.get("user_id") for entry in prompt_payloads[-1]["participants"]}
    assert IGNORED_USER_ID not in participant_ids


def test_pipeline_blocks_exact_ignored_message_payload():
    tracer = CaptureTracer()
    pipeline = MessagePipeline(
        persona="tiffanylynn03",
        user_id="ai-1",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=tracer,
    )

    actions = asyncio.run(
        pipeline.process_chat(
            {
                "text": "Roleplay AI chat output enabled",
                "sender_id": IGNORED_USER_ID,
                "sender_name": IGNORED_USER_ID,
                "timestamp": 1772969267.5980396,
                "metadata": {},
            }
        )
    )

    assert actions == []
    assert pipeline._rolling_buffer.items() == []  # type: ignore[attr-defined]
    assert pipeline._memory.recent() == []  # type: ignore[attr-defined]
    assert not [payload for name, payload in tracer.events if name == "llm_address_check"]
    assert not [payload for name, payload in tracer.events if name == "llm_prompt_bundle"]


def test_pipeline_treats_exact_ai_payload_as_self_and_seeds_history():
    tracer = CaptureTracer()
    pipeline = MessagePipeline(
        persona="tiffanylynn03",
        user_id="ai-1",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=tracer,
    )

    actions = asyncio.run(
        pipeline.process_chat(
            {
                "text": "/me wiggles in her sleep and squeezes her newborn brother tightly",
                "sender_id": "ai",
                "sender_name": "tiffanylynn03",
                "timestamp": 1772976378.5799742,
                "metadata": {},
            }
        )
    )

    assert actions == []
    rolling = pipeline._rolling_buffer.items()  # type: ignore[attr-defined]
    assert len(rolling) == 1
    assert rolling[0].sender_id == "ai"
    assert rolling[0].sender_name == "tiffanylynn03"
    assert rolling[0].text == "/me wiggles in her sleep and squeezes her newborn brother tightly"
    recent = pipeline._memory.recent()  # type: ignore[attr-defined]
    assert len(recent) == 1
    assert recent[0].sender_id == "ai"
    assert recent[0].sender_name == "tiffanylynn03"
    assert not [payload for name, payload in tracer.events if name == "llm_address_check"]
    assert not [payload for name, payload in tracer.events if name == "llm_prompt_bundle"]


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

    async def summarize_overflow(self, summary, messages):
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

    async def summarize_overflow(self, summary, messages):
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


class MentionLookupCountingStore(InMemoryKnowledgeStore):
    def __init__(self) -> None:
        super().__init__()
        self.fetch_by_name_calls = 0
        self.fetch_by_partial_calls = 0
        self.fetch_name_index_calls = 0

    def fetch_people_by_name(self, names):
        self.fetch_by_name_calls += 1
        return super().fetch_people_by_name(names)

    def fetch_people_by_partial_name(self, names):
        self.fetch_by_partial_calls += 1
        return super().fetch_people_by_partial_name(names)

    def fetch_people_name_index(self):
        self.fetch_name_index_calls += 1
        return super().fetch_people_name_index()


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
        facts_config=FactsConfig(
            mode="periodic",
            interval_seconds=1,
            evidence_max_messages=12,
            in_bundle=False,
            min_pending_messages=1,
        ),
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
        facts_config=FactsConfig(
            mode="periodic",
            interval_seconds=1,
            evidence_max_messages=12,
            in_bundle=False,
            min_pending_messages=1,
        ),
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


def test_llm_summarize_receives_overflow_only_not_current_incoming():
    llm = SummaryCaptureNeverAddressedLLM()
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=2, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        summary_strategy="llm",
    )

    asyncio.run(pipeline.process_chat({"text": "one", "from_name": "User", "from_id": "user-1", "timestamp": 10.0}))
    asyncio.run(pipeline.process_chat({"text": "two", "from_name": "User", "from_id": "user-1", "timestamp": 11.0}))
    asyncio.run(pipeline.process_chat({"text": "three", "from_name": "User", "from_id": "user-1", "timestamp": 12.0}))
    asyncio.run(pipeline.process_chat({"text": "four", "from_name": "User", "from_id": "user-1", "timestamp": 13.0}))

    assert llm.summarize_calls
    last_call = llm.summarize_calls[-1]
    assert "one" in last_call
    assert "four" not in last_call


def test_pipeline_summarize_overflow_preserves_sender_name_not_uuid():
    llm = SummarySenderCaptureLLM()
    sender_uuid = "4b7e7cb8-f2e1-49c9-9a57-aec11ad15535"
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=1, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        summary_strategy="llm",
    )

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "/me kisses Tiffany, holds her gently, and begins to nurse her",
                "from_name": "Rene",
                "from_id": sender_uuid,
                "timestamp": 10.0,
            }
        )
    )
    asyncio.run(
        pipeline.process_chat(
            {
                "text": "hello",
                "from_name": "Tiffany",
                "from_id": "user-2",
                "timestamp": 11.0,
            }
        )
    )
    asyncio.run(
        pipeline.process_chat(
            {
                "text": "still here",
                "from_name": "Tiffany",
                "from_id": "user-2",
                "timestamp": 12.0,
            }
        )
    )

    assert llm.summarize_calls
    assert llm.summarize_calls[-1] == [
        {
            "sender_id": sender_uuid,
            "sender_name": "Rene",
            "text": "/me kisses Tiffany, holds her gently, and begins to nurse her",
        }
    ]


def test_pipeline_filters_overflow_already_covered_by_summary_meta():
    llm = SummaryCaptureNeverAddressedLLM()
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=1, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        summary_strategy="llm",
    )

    pipeline._memory.add_message(  # type: ignore[attr-defined]
        InboundChat(text="one", sender_id="user-1", sender_name="User", timestamp=10.0, raw={})
    )
    pipeline._memory.add_message(  # type: ignore[attr-defined]
        InboundChat(text="two", sender_id="user-1", sender_name="User", timestamp=11.0, raw={})
    )
    pipeline._memory._summary = "summary"  # type: ignore[attr-defined]
    pipeline._memory._summary_meta = {  # type: ignore[attr-defined]
        "last_updated_ts": 20.0,
        "range_start_ts": 10.0,
        "range_end_ts": 10.0,
    }

    asyncio.run(pipeline.process_chat({"text": "three", "from_name": "User", "from_id": "user-1", "timestamp": 12.0}))
    asyncio.run(pipeline.process_chat({"text": "four", "from_name": "User", "from_id": "user-1", "timestamp": 13.0}))

    assert llm.summarize_calls == [["two"]]
    assert pipeline._memory.summary_meta()["range_end_ts"] == 11.0  # type: ignore[attr-defined]


def test_not_addressed_llm_summary_empty_requeues_overflow_without_simple_fallback():
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=NeverAddressedLLM(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=1, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        summary_strategy="llm",
    )

    asyncio.run(pipeline.process_chat({"text": "one", "from_name": "User", "from_id": "user-1", "timestamp": 10.0}))
    asyncio.run(pipeline.process_chat({"text": "two", "from_name": "User", "from_id": "user-1", "timestamp": 11.0}))
    asyncio.run(pipeline.process_chat({"text": "three", "from_name": "User", "from_id": "user-1", "timestamp": 12.0}))

    assert pipeline._memory.summary() == ""  # type: ignore[attr-defined]
    overflow = [item.text for item in pipeline._memory.drain_overflow()]  # type: ignore[attr-defined]
    assert overflow == ["one", "two"]


def test_addressed_llm_summary_empty_requeues_overflow_without_simple_fallback():
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=AddressedEmptySummaryLLM(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=1, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        summary_strategy="llm",
    )

    asyncio.run(pipeline.process_chat({"text": "one", "from_name": "User", "from_id": "user-1", "timestamp": 10.0}))
    asyncio.run(pipeline.process_chat({"text": "two", "from_name": "User", "from_id": "user-1", "timestamp": 11.0}))
    asyncio.run(pipeline.process_chat({"text": "three", "from_name": "User", "from_id": "user-1", "timestamp": 12.0}))

    assert pipeline._memory.summary() == ""  # type: ignore[attr-defined]
    overflow = [item.text for item in pipeline._memory.drain_overflow()]  # type: ignore[attr-defined]
    assert overflow == ["one", "two"]


def test_simple_summary_strategy_keeps_overflow_when_no_summary_is_produced():
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=NeverAddressedLLM(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=1, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        summary_strategy="simple",
    )

    asyncio.run(pipeline.process_chat({"text": "one", "from_name": "User", "from_id": "user-1", "timestamp": 10.0}))
    asyncio.run(pipeline.process_chat({"text": "two", "from_name": "User", "from_id": "user-1", "timestamp": 11.0}))
    asyncio.run(pipeline.process_chat({"text": "three", "from_name": "User", "from_id": "user-1", "timestamp": 12.0}))

    assert pipeline._memory.summary() == ""  # type: ignore[attr-defined]
    assert pipeline._memory.summary_meta() == {}  # type: ignore[attr-defined]
    overflow = [item.text for item in pipeline._memory.drain_overflow()]  # type: ignore[attr-defined]
    assert overflow == ["one", "two"]


def test_pipeline_ignores_bundle_summary_when_no_overflow():
    class BundleSummaryOnlyLLM(LLMClient):
        async def is_addressed_to_me(self, chat, persona, environment=None, participants=None, context=None) -> bool:
            return True

        async def generate_response(self, chat, context) -> LLMResponse:  # pragma: no cover - unused
            raise NotImplementedError

        async def extract_facts(self, chat, context) -> list[ExtractedFact]:  # pragma: no cover - unused
            return []

        async def summarize_overflow(self, summary, messages) -> str:  # pragma: no cover - unused
            return "overflow-summary"

        async def generate_bundle(self, chat, context, overflow=None, incoming_batch=None) -> LLMResponseBundle:
            return LLMResponseBundle(
                text="",
                actions=[],
                facts=[],
                participant_hints=[],
                summary="model-summary-should-be-ignored",
            )

    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=BundleSummaryOnlyLLM(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        summary_strategy="llm",
    )

    asyncio.run(pipeline.process_chat({"text": "hello", "from_name": "User", "from_id": "user-1", "timestamp": 10.0}))

    assert pipeline._memory.summary() == ""  # type: ignore[attr-defined]


def test_pipeline_summary_updates_from_overflow_not_bundle_summary():
    class BundleAndOverflowSummaryLLM(LLMClient):
        def __init__(self) -> None:
            self.summarize_calls = 0

        async def is_addressed_to_me(self, chat, persona, environment=None, participants=None, context=None) -> bool:
            return True

        async def generate_response(self, chat, context) -> LLMResponse:  # pragma: no cover - unused
            raise NotImplementedError

        async def extract_facts(self, chat, context) -> list[ExtractedFact]:  # pragma: no cover - unused
            return []

        async def summarize_overflow(self, summary, messages) -> str:
            self.summarize_calls += 1
            return "overflow-summary"

        async def generate_bundle(self, chat, context, overflow=None, incoming_batch=None) -> LLMResponseBundle:
            return LLMResponseBundle(
                text="",
                actions=[],
                facts=[],
                participant_hints=[],
                summary="model-summary-should-be-ignored",
            )

    llm = BundleAndOverflowSummaryLLM()
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=1, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        summary_strategy="llm",
    )

    asyncio.run(pipeline.process_chat({"text": "one", "from_name": "User", "from_id": "user-1", "timestamp": 10.0}))
    asyncio.run(pipeline.process_chat({"text": "two", "from_name": "User", "from_id": "user-1", "timestamp": 11.0}))
    asyncio.run(pipeline.process_chat({"text": "three", "from_name": "User", "from_id": "user-1", "timestamp": 12.0}))

    assert llm.summarize_calls >= 1
    assert pipeline._memory.summary() == "overflow-summary"  # type: ignore[attr-defined]


def test_pipeline_clamps_summary_meta_range_end_to_oldest_recent_and_logs_event():
    tracer = CaptureTracer()
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=10, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=10),
        experience_store=ExperienceStore(),
        tracer=tracer,
    )
    pipeline._memory.add_message(  # type: ignore[attr-defined]
        InboundChat(text="older", sender_id="user-2", sender_name="Evie", timestamp=100.0, raw={})
    )
    pipeline._memory._summary = "summary"  # type: ignore[attr-defined]
    pipeline._memory._summary_meta = {  # type: ignore[attr-defined]
        "last_updated_ts": 120.0,
        "range_start_ts": 50.0,
        "range_end_ts": 999.0,
    }

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "hello",
                "from_name": "Alice",
                "from_id": "user-3",
                "timestamp": 101.0,
            }
        )
    )

    summary_meta = pipeline._memory.summary_meta()  # type: ignore[attr-defined]
    recent = pipeline._memory.recent()  # type: ignore[attr-defined]
    oldest_recent = min(float(item.timestamp or 0.0) for item in recent if float(item.timestamp or 0.0) > 0.0)
    assert float(summary_meta.get("range_end_ts", 0.0)) <= oldest_recent
    assert any(name == "summary_boundary_violation" for name, _payload in tracer.events)


def test_pipeline_clamps_tiny_summary_boundary_drift_without_logging_event():
    tracer = CaptureTracer()
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=10, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=10),
        experience_store=ExperienceStore(),
        tracer=tracer,
    )
    pipeline._memory.add_message(  # type: ignore[attr-defined]
        InboundChat(text="older", sender_id="user-2", sender_name="Evie", timestamp=100.0, raw={})
    )
    pipeline._memory._summary = "summary"  # type: ignore[attr-defined]
    pipeline._memory._summary_meta = {  # type: ignore[attr-defined]
        "last_updated_ts": 120.0,
        "range_start_ts": 50.0,
        "range_end_ts": 100.0005,
    }

    pipeline._enforce_summary_before_recent(reason="chat_disabled")  # type: ignore[attr-defined]

    summary_meta = pipeline._memory.summary_meta()  # type: ignore[attr-defined]
    assert float(summary_meta.get("range_end_ts", 0.0)) <= 100.0
    assert not any(name == "summary_boundary_violation" for name, _payload in tracer.events)


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
        facts_config=FactsConfig(
            mode="periodic",
            interval_seconds=1,
            evidence_max_messages=12,
            in_bundle=False,
            min_pending_messages=1,
        ),
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


def test_periodic_facts_batch_until_min_pending():
    llm = PeriodicFactsLLM(
        [ExtractedFact(user_id="user-2", name="Evie", facts=["likes tea"])]
    )
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=10),
        rolling_buffer=RollingBuffer(max_items=10),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        facts_config=FactsConfig(
            mode="periodic",
            interval_seconds=120,
            evidence_max_messages=12,
            in_bundle=False,
            min_pending_messages=3,
            max_pending_age_seconds=120,
        ),
    )

    async def run_batch():
        for idx in range(3):
            await pipeline.process_chat(
                {
                    "text": f"msg-{idx}",
                    "from_name": "User",
                    "from_id": "user-1",
                    "timestamp": 100 + idx,
                    "participants": [{"user_id": "user-2", "name": "Evie"}],
                }
            )
        await asyncio.sleep(0.05)

    asyncio.run(run_batch())

    assert len(llm.evidence_calls) == 1
    assert llm.evidence_calls[0] == ["msg-0", "msg-1", "msg-2"]


def test_periodic_facts_overflow_does_not_force_flush_by_default():
    llm = PeriodicFactsLLM(
        [ExtractedFact(user_id="user-2", name="Evie", facts=["likes tea"])]
    )
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=2, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=10),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        facts_config=FactsConfig(
            mode="periodic",
            interval_seconds=120,
            evidence_max_messages=12,
            in_bundle=False,
            min_pending_messages=10,
            max_pending_age_seconds=120,
            flush_on_overflow=False,
        ),
    )

    async def run_once():
        await pipeline.process_chat_batch(
            [
                {"text": "one", "from_name": "User", "from_id": "user-1", "timestamp": 200},
                {"text": "two", "from_name": "User", "from_id": "user-1", "timestamp": 201},
                {"text": "three", "from_name": "User", "from_id": "user-1", "timestamp": 202},
            ]
        )
        await pipeline.process_chat(
            {
                "text": "four",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 203,
            }
        )
        await asyncio.sleep(0.05)

    asyncio.run(run_once())

    assert llm.evidence_calls == []


def test_periodic_facts_overflow_can_force_flush_when_enabled():
    llm = PeriodicFactsLLM(
        [ExtractedFact(user_id="user-2", name="Evie", facts=["likes tea"])]
    )
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=2, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=10),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        facts_config=FactsConfig(
            mode="periodic",
            interval_seconds=120,
            evidence_max_messages=12,
            in_bundle=False,
            min_pending_messages=10,
            max_pending_age_seconds=120,
            flush_on_overflow=True,
        ),
    )

    async def run_once():
        await pipeline.process_chat_batch(
            [
                {"text": "one", "from_name": "User", "from_id": "user-1", "timestamp": 300},
                {"text": "two", "from_name": "User", "from_id": "user-1", "timestamp": 301},
                {"text": "three", "from_name": "User", "from_id": "user-1", "timestamp": 302},
            ]
        )
        await pipeline.process_chat(
            {
                "text": "four",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 303,
            }
        )
        await asyncio.sleep(0.05)

    asyncio.run(run_once())

    assert llm.evidence_calls
    assert any("one" in call for call in llm.evidence_calls)


def test_periodic_facts_restore_preserves_pending_queue():
    llm_seed = PeriodicFactsLLM(
        [ExtractedFact(user_id="user-2", name="Evie", facts=["likes tea"])]
    )
    pipeline_seed = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm_seed,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=10),
        rolling_buffer=RollingBuffer(max_items=10),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        facts_config=FactsConfig(
            mode="periodic",
            interval_seconds=120,
            evidence_max_messages=12,
            in_bundle=False,
            min_pending_messages=3,
            max_pending_age_seconds=120,
        ),
    )

    async def run_seed():
        for idx in range(2):
            await pipeline_seed.process_chat(
                {
                    "text": f"msg-{idx}",
                    "from_name": "User",
                    "from_id": "user-1",
                    "timestamp": 400 + idx,
                }
            )
        await asyncio.sleep(0.05)

    asyncio.run(run_seed())

    snapshot = pipeline_seed.snapshot_state()
    facts_snapshot = snapshot.get("facts", {})
    assert isinstance(facts_snapshot, dict)
    assert len(facts_snapshot.get("pending_messages", [])) == 2

    llm_restore = PeriodicFactsLLM(
        [ExtractedFact(user_id="user-2", name="Evie", facts=["likes tea"])]
    )
    pipeline_restore = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm_restore,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=10),
        rolling_buffer=RollingBuffer(max_items=10),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        facts_config=FactsConfig(
            mode="periodic",
            interval_seconds=120,
            evidence_max_messages=12,
            in_bundle=False,
            min_pending_messages=3,
            max_pending_age_seconds=120,
        ),
    )
    pipeline_restore.restore_state(snapshot)

    async def run_restore():
        await pipeline_restore.process_chat(
            {
                "text": "msg-2",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 402,
            }
        )
        await asyncio.sleep(0.05)

    asyncio.run(run_restore())

    assert len(llm_restore.evidence_calls) == 1
    assert llm_restore.evidence_calls[0] == ["msg-0", "msg-1", "msg-2"]


def test_pipeline_snapshot_restores_display_name_cache_for_nameless_incoming():
    tracer = CaptureTracer()
    pipeline_seed = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
    )
    pipeline_seed.update_environment(
        {
            "agents": [{"name": "Rene", "uuid": "user-2"}],
            "objects": [],
            "location": "Test Zone",
            "timestamp": 10.0,
        }
    )

    snapshot = pipeline_seed.snapshot_state()
    assert snapshot["identity"]["display_names_by_id"]["user-2"] == "Rene"

    pipeline_restore = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=tracer,
    )
    pipeline_restore.restore_state(snapshot)

    asyncio.run(
        pipeline_restore.process_chat(
            {
                "text": "hello again",
                "from_id": "user-2",
                "timestamp": 11.0,
            }
        )
    )

    prompt_payloads = [payload for name, payload in tracer.events if name == "llm_prompt_bundle"]
    assert prompt_payloads
    incoming = prompt_payloads[-1]["incoming"]
    assert incoming["sender"] == "Rene"
    assert incoming["sender_id"] == "user-2"


def test_periodic_facts_restore_requeues_recent_when_snapshot_has_no_facts_state():
    llm_seed = PeriodicFactsLLM(
        [ExtractedFact(user_id="user-2", name="Evie", facts=["likes tea"])]
    )
    pipeline_seed = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm_seed,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=10),
        rolling_buffer=RollingBuffer(max_items=10),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        facts_config=FactsConfig(
            mode="periodic",
            interval_seconds=120,
            evidence_max_messages=12,
            in_bundle=False,
            min_pending_messages=3,
            max_pending_age_seconds=120,
        ),
    )

    async def run_seed():
        for idx in range(2):
            await pipeline_seed.process_chat(
                {
                    "text": f"msg-{idx}",
                    "from_name": "User",
                    "from_id": "user-1",
                    "timestamp": 500 + idx,
                }
            )
        await asyncio.sleep(0.05)

    asyncio.run(run_seed())

    snapshot = pipeline_seed.snapshot_state()
    snapshot.pop("facts", None)

    llm_restore = PeriodicFactsLLM(
        [ExtractedFact(user_id="user-2", name="Evie", facts=["likes tea"])]
    )
    pipeline_restore = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm_restore,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=10),
        rolling_buffer=RollingBuffer(max_items=10),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        facts_config=FactsConfig(
            mode="periodic",
            interval_seconds=120,
            evidence_max_messages=12,
            in_bundle=False,
            min_pending_messages=3,
            max_pending_age_seconds=120,
        ),
    )
    pipeline_restore.restore_state(snapshot)
    assert llm_restore.evidence_calls == []

    async def run_restore():
        await pipeline_restore.process_chat(
            {
                "text": "msg-2",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 502,
            }
        )
        await asyncio.sleep(0.05)

    asyncio.run(run_restore())

    assert len(llm_restore.evidence_calls) == 1
    assert llm_restore.evidence_calls[0] == ["msg-0", "msg-1", "msg-2"]


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


def test_pipeline_uses_cached_display_name_when_incoming_name_missing():
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
    )
    pipeline.update_environment(
        {
            "agents": [{"name": "Rene", "uuid": "user-2"}],
            "objects": [],
            "location": "Test Zone",
            "timestamp": 10.0,
        }
    )

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "hello there",
                "from_id": "user-2",
                "timestamp": 11.0,
            }
        )
    )

    prompt_payloads = [payload for name, payload in tracer.events if name == "llm_prompt_bundle"]
    assert prompt_payloads
    incoming = prompt_payloads[-1]["incoming"]
    assert incoming["sender"] == "Rene"
    assert incoming["sender_id"] == "user-2"


def test_pipeline_text_mention_lookup_marks_context():
    llm = ContextCaptureLLM()
    store = InMemoryKnowledgeStore()
    store.upsert_person_facts("user-42", "James Allardyce", ["is a father"])
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
                "text": "James is sick and will be offline for a week.",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 7,
            }
        )
    )

    assert llm.contexts
    people_facts = llm.contexts[-1].people_facts
    assert "user-42" in people_facts
    assert people_facts["user-42"].get("match_type") == "text_mention"
    assert people_facts["user-42"].get("matched_query") == "james"


def test_pipeline_text_mention_uses_name_index_without_word_probing():
    llm = ContextCaptureLLM()
    store = MentionLookupCountingStore()
    store.upsert_person_facts("user-42", "James Allardyce", ["is a father"])
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
                "text": "Did you see James today?",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 8,
            }
        )
    )

    assert llm.contexts
    people_facts = llm.contexts[-1].people_facts
    assert people_facts["user-42"].get("match_type") == "text_mention"
    assert store.fetch_by_name_calls == 0
    assert store.fetch_by_partial_calls == 0
    assert store.fetch_name_index_calls >= 1


def test_pipeline_text_mention_skips_ambiguous_name_tokens():
    llm = ContextCaptureLLM()
    store = InMemoryKnowledgeStore()
    store.upsert_person_facts("user-1", "Alex Smith", [])
    store.upsert_person_facts("user-2", "Alex Johnson", [])
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
                "text": "Did Alex come by earlier?",
                "from_name": "User",
                "from_id": "user-9",
                "timestamp": 9,
            }
        )
    )

    assert llm.contexts
    people_facts = llm.contexts[-1].people_facts
    assert "user-1" not in people_facts
    assert "user-2" not in people_facts


def test_pipeline_text_mention_matches_multiword_name():
    llm = ContextCaptureLLM()
    store = InMemoryKnowledgeStore()
    store.upsert_person_facts("user-5", "Anna Marie", ["likes drawing"])
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
                "text": "Is Anna Marie around today?",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 10,
            }
        )
    )

    assert llm.contexts
    people_facts = llm.contexts[-1].people_facts
    assert people_facts["user-5"].get("match_type") == "text_mention"
    assert people_facts["user-5"].get("matched_query") == "anna marie"


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


def test_pipeline_adds_routine_summaries_non_destructively():
    llm = ContextCaptureLLM()
    store = ExperienceStore()
    store.add(
        "We had ramen together on a rainy evening near the station.",
        {
            "experience_id": "r1",
            "timestamp_end": "2026-02-14 20:11:00",
            "timestamp": 1707941460.0,
        },
        persona_id="TestPersona",
    )
    store.add(
        "We had ramen together on a rainy evening near the station and chatted quietly.",
        {
            "experience_id": "r2",
            "timestamp_end": "2026-02-16 20:25:00",
            "timestamp": 1708115100.0,
        },
        persona_id="TestPersona",
    )
    store.add(
        "We visited a bookstore at noon.",
        {
            "experience_id": "other",
            "timestamp_end": "2026-02-16 12:00:00",
            "timestamp": 1708084800.0,
        },
        persona_id="TestPersona",
    )

    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=store,
        tracer=NoOpTracer(),
        routine_summary_enabled=True,
        routine_summary_limit=2,
        routine_summary_min_count=2,
    )

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "What do you remember about ramen nights?",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 9,
            }
        )
    )

    assert llm.contexts
    related = llm.contexts[-1].related_experiences
    assert any(item.get("metadata", {}).get("source") == "routine_summary" for item in related)
    routine_texts = [
        item.get("text", "")
        for item in related
        if item.get("metadata", {}).get("source") == "routine_summary"
    ]
    assert any("similar experiences" in text for text in routine_texts)
    raw_texts = [
        item.get("text", "")
        for item in related
        if item.get("metadata", {}).get("source") != "routine_summary"
    ]
    assert any("ramen" in text.lower() for text in raw_texts)


def test_pipeline_collapses_near_duplicate_related_experiences():
    llm = ContextCaptureLLM()
    store = ExperienceStore()
    store.add(
        "We had ramen together on a rainy evening near the station.",
        {
            "experience_id": "dup-1",
            "timestamp_end": "2026-02-14 20:11:00",
            "timestamp": 1707941460.0,
        },
        persona_id="TestPersona",
    )
    store.add(
        "We had ramen together on a rainy evening near the station!",
        {
            "experience_id": "dup-2",
            "timestamp_end": "2026-02-16 20:25:00",
            "timestamp": 1708115100.0,
        },
        persona_id="TestPersona",
    )

    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=llm,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=store,
        tracer=NoOpTracer(),
        experience_top_k=4,
        near_duplicate_collapse_enabled=True,
        near_duplicate_similarity=0.9,
    )

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "Tell me what you remember about ramen near the station.",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 10,
            }
        )
    )

    assert llm.contexts
    related = llm.contexts[-1].related_experiences
    assert len(related) == 1
    metadata = related[0].get("metadata", {})
    assert metadata.get("near_duplicate_count") == 2
    assert metadata.get("near_duplicate_first_seen") == "2024-02-14"
    assert metadata.get("near_duplicate_last_seen") == "2024-02-16"


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


def test_pipeline_logs_reasoning_trace_for_bundle():
    tracer = CaptureTracer()
    pipeline = MessagePipeline(
        persona="TracePersona",
        user_id="ai-1",
        llm=ReasoningTraceLLM(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=tracer,
    )

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "Trace this reasoning",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 99,
            }
        )
    )

    reasoning_payloads = [payload for name, payload in tracer.events if name == "llm_reasoning_bundle"]
    assert reasoning_payloads
    payload = reasoning_payloads[-1]
    assert payload["has_reasoning"] is True
    assert payload["reasoning_tokens"] == 64
    assert payload["reasoning_effort"] == "low"


def test_pipeline_logs_quality_fallback_events_from_llm():
    tracer = CaptureTracer()
    pipeline = MessagePipeline(
        persona="TracePersona",
        user_id="ai-1",
        llm=QualityFallbackEventLLM(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=tracer,
    )

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "hello",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 100,
            }
        )
    )

    fallback_payloads = [payload for name, payload in tracer.events if name == "llm_quality_fallback"]
    assert fallback_payloads
    payload = fallback_payloads[-1]
    assert payload["mode"] == "bundle"
    assert payload["reason"] == "test_fallback"
    assert payload["fallback_client"] == "RuleBasedLLMClient"
    assert payload["model"] == "test-model"


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


def test_experience_created_via_autonomy_tick_when_chat_disabled():
    pipeline = MessagePipeline(
        persona="EpisodePersona",
        user_id="ai-1",
        llm=NeverAddressedLLM(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=20, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=20),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        episode_config=EpisodeConfig(
            enabled=True,
            min_messages=2,
            max_messages=2,
            inactivity_seconds=300,
            forced_interval_seconds=0,
            overlap_messages=0,
        ),
    )

    batch = [
        {"text": "one", "from_name": "User", "from_id": "user-1", "timestamp": 1},
        {"text": "two", "from_name": "User", "from_id": "user-1", "timestamp": 2},
    ]

    async def run() -> None:
        pipeline.set_llm_chat_enabled(False)
        await pipeline.process_chat_batch(batch)
        await pipeline.generate_autonomous_actions(45.0)
        # Allow the scheduled episode task to complete.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(run())

    experiences = pipeline._experience_store.all()  # type: ignore[attr-defined]
    assert len(experiences) == 1
    summary_text = experiences[0].text
    assert "one" in summary_text
    assert "two" in summary_text


def test_episode_summary_uses_dedicated_episode_summarizer():
    class EpisodeSummaryMethodLLM(LLMClient):
        def __init__(self) -> None:
            self.called_episode = False

        async def is_addressed_to_me(self, chat, persona, environment=None, participants=None, context=None) -> bool:
            return False

        async def generate_response(self, chat, context) -> LLMResponse:  # pragma: no cover - unused
            raise NotImplementedError

        async def extract_facts(self, chat, context) -> list[ExtractedFact]:  # pragma: no cover - unused
            return []

        async def summarize_overflow(self, summary, messages) -> str:
            raise AssertionError("overflow summarize should not be used for episodic summaries")

        async def summarize_episode(self, messages) -> str:
            self.called_episode = True
            return "episode-via-dedicated-method"

    llm = EpisodeSummaryMethodLLM()
    pipeline = MessagePipeline(
        persona="EpisodePersona",
        user_id="ai-1",
        llm=llm,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=20, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=20),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        episode_config=EpisodeConfig(
            enabled=True,
            min_messages=2,
            max_messages=2,
            inactivity_seconds=300,
            forced_interval_seconds=0,
            overlap_messages=0,
        ),
    )

    batch = [
        {"text": "one", "from_name": "User", "from_id": "user-1", "timestamp": 1},
        {"text": "two", "from_name": "User", "from_id": "user-1", "timestamp": 2},
    ]

    async def run() -> None:
        pipeline.set_llm_chat_enabled(False)
        await pipeline.process_chat_batch(batch)
        await pipeline.generate_autonomous_actions(45.0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(run())

    assert llm.called_episode is True
    experiences = pipeline._experience_store.all()  # type: ignore[attr-defined]
    assert len(experiences) == 1
    assert experiences[0].text == "episode-via-dedicated-method"


def test_pipeline_merges_name_only_and_id_participant():
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

    participants = pipeline._normalize_participants(  # type: ignore[attr-defined]
        [Participant(user_id="", name="Evie"), Participant(user_id="user-2", name="Evie")]
    )
    assert len(participants) == 1
    assert participants[0].user_id == "user-2"


def test_pipeline_prompt_participants_cover_recent_and_incoming_active_speakers():
    tracer = CaptureTracer()
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=tracer,
    )
    pipeline._memory.add_message(  # type: ignore[attr-defined]
        InboundChat(text="old", sender_id="user-2", sender_name="Evie", timestamp=10.0, raw={})
    )

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "new",
                "from_name": "Alice",
                "from_id": "user-3",
                "timestamp": 11.0,
            }
        )
    )

    prompt_payloads = [payload for name, payload in tracer.events if name == "llm_prompt_bundle"]
    assert prompt_payloads
    participant_ids = {entry.get("user_id") for entry in prompt_payloads[-1]["participants"]}
    assert "user-2" in participant_ids
    assert "user-3" in participant_ids


def test_pipeline_promotes_self_uuid_and_uses_persona_sender():
    tracer = CaptureTracer()
    pipeline = MessagePipeline(
        persona="isabella.elara",
        user_id="default_user",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=8, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=8),
        experience_store=ExperienceStore(),
        tracer=tracer,
    )
    promoted_uuid = "11111111-2222-3333-4444-555555555555"

    # Confirmed self-message should promote placeholder self id to UUID.
    asyncio.run(
        pipeline.process_chat(
            {
                "text": "self line",
                "from_name": "isabella.elara",
                "from_id": promoted_uuid,
                "timestamp": 20.0,
            }
        )
    )
    asyncio.run(
        pipeline.process_chat(
            {
                "text": "hello",
                "from_name": "Evie",
                "from_id": "user-2",
                "timestamp": 21.0,
            }
        )
    )

    prompt_payloads = [payload for name, payload in tracer.events if name == "llm_prompt_bundle"]
    assert prompt_payloads
    recent = prompt_payloads[-1]["recent_messages"]
    self_messages = [msg for msg in recent if msg.get("sender") == "isabella.elara"]
    assert self_messages
    assert all(msg.get("sender_id") == promoted_uuid for msg in self_messages)


def test_pipeline_clamps_summary_range_end_to_recent_start():
    tracer = CaptureTracer()
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-uuid",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=tracer,
    )
    pipeline._memory.add_message(  # type: ignore[attr-defined]
        InboundChat(text="old", sender_id="user-2", sender_name="Evie", timestamp=100.0, raw={})
    )
    pipeline._memory._summary = "summary"  # type: ignore[attr-defined]
    pipeline._memory._summary_meta = {  # type: ignore[attr-defined]
        "last_updated_ts": 120.0,
        "range_start_ts": 50.0,
        "range_end_ts": 999.0,
    }

    asyncio.run(
        pipeline.process_chat(
            {
                "text": "hello",
                "from_name": "Alice",
                "from_id": "user-3",
                "timestamp": 101.0,
            }
        )
    )

    prompt_payloads = [payload for name, payload in tracer.events if name == "llm_prompt_bundle"]
    assert prompt_payloads
    payload = prompt_payloads[-1]
    summary_meta = payload.get("summary_meta", {})
    assert float(summary_meta.get("range_end_ts", 0.0)) <= 100.0


def test_pipeline_incoming_matches_last_incoming_batch_entry():
    tracer = CaptureTracer()
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5, defer_compression=True),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=tracer,
    )

    asyncio.run(
        pipeline.process_chat_batch(
            [
                {"text": "first", "from_name": "Alpha", "from_id": "user-a", "timestamp": 200.0},
                {"text": "second", "from_name": "Beta", "from_id": "user-b", "timestamp": 200.0},
            ]
        )
    )

    prompt_payloads = [payload for name, payload in tracer.events if name == "llm_prompt_bundle"]
    assert prompt_payloads
    payload = prompt_payloads[-1]
    latest = payload["incoming_batch"][-1]
    assert payload["incoming"]["sender_id"] == latest["sender_id"]
    assert payload["incoming"]["text"] == latest["latest_text"]


def test_pipeline_posture_out_of_order_ignored_and_stale_becomes_unknown():
    pipeline = MessagePipeline(
        persona="TestPersona",
        user_id="ai-1",
        llm=EchoLLMClient(),
        knowledge_store=InMemoryKnowledgeStore(),
        memory=ConversationMemory(SimpleMemoryCompressor(), max_recent=5),
        rolling_buffer=RollingBuffer(max_items=5),
        experience_store=ExperienceStore(),
        tracer=NoOpTracer(),
        posture_stale_seconds=1.0,
    )

    pipeline.update_environment(
        {
            "agents": [],
            "objects": [],
            "location": "Zone",
            "is_sitting": True,
            "timestamp": 100.0,
        }
    )
    pipeline.update_environment(
        {
            "agents": [],
            "objects": [],
            "location": "Zone",
            "is_sitting": False,
            "timestamp": 99.0,
        }
    )

    assert pipeline._environment.is_sitting is True  # type: ignore[attr-defined]
    assert pipeline._effective_posture_is_sitting(now_ts=101.5) is None  # type: ignore[attr-defined]


def test_payload_contract_emits_expected_soft_heal_categories():
    payload = {
        "persona": "TestPersona",
        "user_id": "default_user",
        "participants": [{"user_id": "", "name": "Evie"}, {"user_id": "user-2", "name": "Evie"}],
        "recent_messages": [{"sender": "Evie", "sender_id": "user-2", "text": "hi", "timestamp": "2026-01-01 10:00:00"}],
        "overflow_messages": [],
        "incoming_batch": [
            {
                "sender_id": "user-3",
                "sender_name": "Alice",
                "latest_text": "latest",
                "last_timestamp": 200.0,
                "arrival_order": 1,
            }
        ],
        "incoming": {"sender_id": "user-x", "sender": "Wrong", "text": "old", "timestamp": "2026-01-01 10:00:01"},
        "summary_meta": {"range_start_ts": 300.0, "range_end_ts": 100.0, "last_updated_ts": 90.0},
        "recent_time_range": {"start": "2026-01-01 10:00:00", "end": "2026-01-01 10:00:00"},
        "now_timestamp": "2026-01-01 10:00:02",
    }

    _repaired, warnings = normalize_and_validate_payload(payload)
    categories = {warning.get("category") for warning in warnings}
    assert "identity_merge" in categories
    assert "participant_coverage" in categories
    assert "summary_range_clamp" in categories
    assert "incoming_repair" in categories
    assert "self_id_missing_uuid" in categories
