import asyncio
import contextlib

from gpt5_roleplay_system.session import ClientSession


class _DummyWriter:
    def write(self, _data: bytes) -> None:
        return None

    async def drain(self) -> None:
        return None


class _CaptureController:
    def __init__(self) -> None:
        self.batch_calls = []
        self.single_calls = []

    async def process_chat_batch(self, batch):
        self.batch_calls.append(batch)
        return []

    async def process_chat(self, data):
        self.single_calls.append(data)
        return []

    def update_environment(self, _data):
        return None

    def set_user_id(self, _user_id):
        return None

    def set_llm_chat_enabled(self, _enabled):
        return None

    async def generate_autonomous_actions(self, _window):
        return []


def test_process_queue_batches_rapid_process_chat_messages():
    async def run() -> None:
        controller = _CaptureController()
        session = ClientSession(
            session_id="batch-test",
            controller=controller,
            writer=_DummyWriter(),
            batch_window_seconds=0.05,
            batch_max_size=10,
        )
        queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        worker = asyncio.create_task(session.process_queue(queue))
        try:
            await queue.put(("process_chat", {"text": "one", "from_name": "User", "from_id": "u", "timestamp": 1}))
            await queue.put(("process_chat", {"text": "two", "from_name": "User", "from_id": "u", "timestamp": 2}))
            await queue.put(("process_chat", {"text": "three", "from_name": "User", "from_id": "u", "timestamp": 3}))

            await asyncio.wait_for(queue.join(), timeout=1.0)

            assert len(controller.batch_calls) == 1
            assert len(controller.batch_calls[0]) == 3
            assert controller.single_calls == []
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker

    asyncio.run(run())
