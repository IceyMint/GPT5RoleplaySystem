import asyncio
import json

from gpt5_roleplay_system.config import EpisodeConfig, load_config
from gpt5_roleplay_system.controller import SessionController
from gpt5_roleplay_system.llm import EchoLLMClient
from gpt5_roleplay_system.neo4j_store import InMemoryKnowledgeStore
from gpt5_roleplay_system.observability import NoOpTracer


class CaptureTracer(NoOpTracer):
    def __init__(self) -> None:
        self.events = []

    def log_event(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


def test_logged_in_agent_updates_persona():
    controller = SessionController(
        persona="DefaultPersona",
        user_id="ai-1",
        knowledge_store=InMemoryKnowledgeStore(),
        llm=EchoLLMClient(),
        tracer=NoOpTracer(),
        episode_config=EpisodeConfig(persist_state=False),
    )

    async def run() -> None:
        await controller.process_chat(
            {
                "text": "Hello",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 1,
                "logged_in_agent": "Isabella",
            }
        )

    asyncio.run(run())
    assert controller.persona() == "Isabella"


def test_environment_update_logged_in_agent_updates_default_persona():
    controller = SessionController(
        persona="DefaultPersona",
        user_id="ai-1",
        knowledge_store=InMemoryKnowledgeStore(),
        llm=EchoLLMClient(),
        tracer=NoOpTracer(),
        episode_config=EpisodeConfig(persist_state=False),
    )

    controller.update_environment(
        {
            "agents": [],
            "objects": [],
            "location": "Region",
            "avatar_position": "(0,0,0)",
            "timestamp": 1,
            "logged_in_agent": "Isabella",
        }
    )

    assert controller.persona() == "Isabella"


def test_environment_update_does_not_override_non_default_persona():
    controller = SessionController(
        persona="Isabella",
        user_id="ai-1",
        knowledge_store=InMemoryKnowledgeStore(),
        llm=EchoLLMClient(),
        tracer=NoOpTracer(),
        episode_config=EpisodeConfig(persist_state=False),
    )

    controller.update_environment(
        {
            "agents": [],
            "objects": [],
            "location": "Region",
            "avatar_position": "(0,0,0)",
            "timestamp": 1,
            "logged_in_agent": "SomeoneElse",
        }
    )

    assert controller.persona() == "Isabella"


def test_default_persona_does_not_persist_state_files(tmp_path):
    controller = SessionController(
        persona="DefaultPersona",
        user_id="ai-1",
        knowledge_store=InMemoryKnowledgeStore(),
        llm=EchoLLMClient(),
        tracer=NoOpTracer(),
        episode_config=EpisodeConfig(persist_state=True, state_dir=str(tmp_path)),
    )

    async def run() -> None:
        await controller.process_chat(
            {
                "text": "Hello",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 1,
            }
        )
        await controller.flush_state()

    asyncio.run(run())
    assert list(tmp_path.glob("session_state_*.json")) == []


def test_non_default_persona_still_persists_state_files(tmp_path):
    controller = SessionController(
        persona="Isabella",
        user_id="ai-1",
        knowledge_store=InMemoryKnowledgeStore(),
        llm=EchoLLMClient(),
        tracer=NoOpTracer(),
        episode_config=EpisodeConfig(persist_state=True, state_dir=str(tmp_path)),
    )

    async def run() -> None:
        await controller.process_chat(
            {
                "text": "Hello",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 1,
            }
        )
        await controller.flush_state()

    asyncio.run(run())
    state_files = list(tmp_path.glob("session_state_*.json"))
    assert len(state_files) == 1
    assert state_files[0].name == "session_state_Isabella.json"


def test_self_uuid_promotion_keeps_persona_only_state_filename(tmp_path):
    controller = SessionController(
        persona="Isabella",
        user_id="default_user",
        knowledge_store=InMemoryKnowledgeStore(),
        llm=EchoLLMClient(),
        tracer=NoOpTracer(),
        episode_config=EpisodeConfig(persist_state=True, state_dir=str(tmp_path)),
    )
    promoted_uuid = "11111111-2222-3333-4444-555555555555"

    async def run() -> None:
        await controller.process_chat(
            {
                "text": "self line",
                "from_name": "Isabella",
                "from_id": promoted_uuid,
                "timestamp": 1,
            }
        )
        await controller.flush_state()

    asyncio.run(run())

    assert controller.user_id() == promoted_uuid
    state_files = list(tmp_path.glob("session_state_*.json"))
    assert len(state_files) == 1
    assert state_files[0].name == "session_state_Isabella.json"


def test_persona_only_store_loads_legacy_persona_user_file(tmp_path):
    legacy_path = tmp_path / "session_state_Isabella_default_user.json"
    legacy_payload = {
        "rolling_buffer": [],
        "memory": {
            "recent": [],
            "summary": "legacy summary",
            "summary_meta": {},
        },
        "facts": {
            "cursor_ts": 0.0,
            "last_sweep_ts": 0.0,
            "pending_since_ts": 0.0,
            "pending_messages": [],
            "pending_participants": [],
        },
        "environment": {
            "last_environment_update_ts": 0.0,
            "last_posture_update_ts": 0.0,
            "posture_known": False,
            "posture_is_sitting": False,
            "posture_stale_seconds": 6.0,
        },
        "episode": {"last_episode_ts": 0.0, "last_episode_size": 0},
        "status": {
            "mood": "neutral",
            "mood_ts": 0.0,
            "mood_source": "init",
            "status": "idle",
            "status_ts": 0.0,
            "status_source": "init",
        },
    }
    legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    controller = SessionController(
        persona="Isabella",
        user_id="default_user",
        knowledge_store=InMemoryKnowledgeStore(),
        llm=EchoLLMClient(),
        tracer=NoOpTracer(),
        episode_config=EpisodeConfig(persist_state=True, state_dir=str(tmp_path)),
    )

    assert controller.memory_summary() == "legacy summary"


def test_logged_in_agent_clones_default_persona_profile_and_uses_it_immediately(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
ai_name: DefaultPersona
user_id: default_user
persona_profiles:
  DefaultPersona: |
    Be warm.
    Stay brief.
""".strip(),
        encoding="utf-8",
    )
    config = load_config(str(config_path))
    tracer = CaptureTracer()
    controller = SessionController(
        persona=config.persona,
        user_id=config.user_id,
        knowledge_store=InMemoryKnowledgeStore(),
        llm=EchoLLMClient(),
        tracer=tracer,
        episode_config=EpisodeConfig(persist_state=False),
        persona_profiles=config.persona_profiles,
        config_path=config.config_path,
    )

    async def run() -> None:
        await controller.process_chat(
            {
                "text": "Hello",
                "from_name": "User",
                "from_id": "user-1",
                "timestamp": 1,
                "logged_in_agent": "Isabella",
            }
        )

    asyncio.run(run())

    assert config.config_path == str(config_path)
    assert controller.persona() == "Isabella"
    assert config.persona_profiles["isabella"] == "Be warm.\nStay brief."
    payloads = [payload for name, payload in tracer.events if name == "llm_prompt_bundle"]
    assert payloads
    assert payloads[-1]["persona"] == "Isabella"
    assert payloads[-1]["persona_instructions"] == "Be warm.\nStay brief."
    saved = config_path.read_text(encoding="utf-8")
    assert "Isabella: |" in saved
    assert "Be warm." in saved
    assert "Stay brief." in saved
