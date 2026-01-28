import asyncio
import logging

from gpt5_roleplay_system.config import AutonomyConfig, ServerConfig
from gpt5_roleplay_system.models import Action, CommandType
from gpt5_roleplay_system.neo4j_store import InMemoryKnowledgeStore
from gpt5_roleplay_system.observability import NoOpTracer
from gpt5_roleplay_system.server import GPT5RoleplayServer, _compute_autonomy_delay, _status_loop
from gpt5_roleplay_system.session import ClientSession


class StubController:
    def __init__(self) -> None:
        self.autonomy_windows = []

    async def generate_autonomous_actions(self, window: float):
        self.autonomy_windows.append(window)
        return [Action(command_type=CommandType.EMOTE, content="idle")]

    def activity_snapshot(self, window: float):
        return {
            "seconds_since_activity": 120.0,
            "recent_messages": 2,
            "recent_activity_window_seconds": window,
            "last_inbound_ts": 10.0,
            "last_response_ts": 20.0,
            "mood": "content",
            "mood_ts": 30.0,
            "mood_source": "autonomy",
            "status": "patrolling",
            "status_ts": 40.0,
            "status_source": "autonomy",
        }

    def persona(self) -> str:
        return "Persona"

    def user_id(self) -> str:
        return "ai-1"


class DummyWriter:
    def write(self, data: bytes) -> None:  # pragma: no cover - trivial
        self.last = data

    async def drain(self) -> None:  # pragma: no cover - trivial
        return None


def test_compute_autonomy_delay_recent_activity_backoff():
    autonomy = AutonomyConfig(
        enabled=True,
        base_delay_seconds=60.0,
        min_delay_seconds=10.0,
        max_delay_seconds=300.0,
        recent_activity_window_seconds=45.0,
        recent_activity_multiplier=3.0,
    )
    snapshot = {"seconds_since_activity": 5.0}
    delay = _compute_autonomy_delay(snapshot, autonomy)
    assert delay == 180.0


def test_autonomous_tick_routes_to_pipeline():
    controller = StubController()
    session = ClientSession(
        session_id="s1",
        controller=controller,
        writer=DummyWriter(),
        batch_window_seconds=0.0,
        batch_max_size=1,
    )
    actions = asyncio.run(
        session.handle_message("autonomous_tick", {"recent_activity_window_seconds": 33})
    )
    assert controller.autonomy_windows == [33.0]
    assert actions and actions[0].command_type == CommandType.EMOTE


def test_status_loop_emits_payload_and_exits_on_disconnect():
    controller = StubController()
    session = ClientSession(
        session_id="s-status",
        controller=controller,
        writer=DummyWriter(),
        batch_window_seconds=0.0,
        batch_max_size=1,
    )
    payloads = []

    async def send_status(payload):
        payloads.append(payload)
        raise ConnectionError()

    session.send_status = send_status  # type: ignore[method-assign]
    autonomy = AutonomyConfig(enabled=True, status_interval_seconds=0.01)
    asyncio.run(_status_loop(session, autonomy))
    assert payloads, "status payload was not emitted"
    payload = payloads[0]
    assert payload["session_id"] == "s-status"
    assert payload["persona"] == "Persona"
    assert payload["autonomy_enabled"] is True
    assert payload["mood"] == "content"
    assert payload["status"] == "patrolling"


def test_status_loop_emits_chat_on_configured_channel():
    controller = StubController()
    session = ClientSession(
        session_id="s-status-chat",
        controller=controller,
        writer=DummyWriter(),
        batch_window_seconds=0.0,
        batch_max_size=1,
    )
    actions_sent = []

    async def send_status(_payload):
        return None

    async def send_actions(actions):
        actions_sent.extend(actions)
        raise ConnectionError()

    session.send_status = send_status  # type: ignore[method-assign]
    session.send_actions = send_actions  # type: ignore[method-assign]
    autonomy = AutonomyConfig(enabled=True, status_interval_seconds=0.01, status_channel=-9001)
    asyncio.run(_status_loop(session, autonomy))
    assert actions_sent, "status chat action was not emitted"
    params = actions_sent[0].parameters
    assert params.get("channel") == -9001


def test_error_logs_broadcast_to_status_channel():
    class DummySession:
        def __init__(self) -> None:
            self.actions = []

        async def send_actions(self, actions):
            self.actions.extend(actions)

    async def run() -> None:
        config = ServerConfig()
        config.autonomy = AutonomyConfig(enabled=False, status_channel=-9001)
        server = GPT5RoleplayServer(
            host="127.0.0.1",
            port=9999,
            config=config,
            knowledge_store=InMemoryKnowledgeStore(),
            llm_client=StubController(),  # unused in this test
            tracer=NoOpTracer(),
        )
        session = DummySession()
        server._sessions["s1"] = session  # type: ignore[assignment]
        server._loop = asyncio.get_running_loop()
        server._install_error_handler()
        try:
            logging.getLogger("test.error").error("boom")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        finally:
            server._remove_error_handler()

        assert session.actions, "no error action was broadcast"
        action = session.actions[0]
        assert action.command_type == CommandType.CHAT
        assert action.parameters.get("channel") == -9001
        assert "[error]" in action.content

    asyncio.run(run())
