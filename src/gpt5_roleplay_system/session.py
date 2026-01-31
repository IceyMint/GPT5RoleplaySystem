from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .controller import SessionController
from .models import Action
from .protocol import build_chat_response, encode_message


@dataclass
class ClientSession:
    session_id: str
    controller: SessionController
    writer: asyncio.StreamWriter
    batch_window_seconds: float = 0.0
    batch_max_size: int = 1

    def __post_init__(self) -> None:
        self._write_lock = asyncio.Lock()

    async def handle_message(self, msg_type: str, data: Dict[str, Any]) -> List[Action]:
        if msg_type == "process_chat":
            return await self.controller.process_chat(data)
        if msg_type == "environment_update":
            self.controller.update_environment(data)
            return []
        if msg_type == "set_user_id":
            user_id = data.get("user_id", "") or data.get("value", "")
            if user_id:
                self.controller.set_user_id(user_id)
            return []
        if msg_type == "set_llm_chat_enabled":
            raw_value = data.get("enabled", data.get("value", True))
            enabled = _parse_bool(raw_value)
            self.controller.set_llm_chat_enabled(enabled)
            return []
        if msg_type == "autonomous_tick":
            window = float(data.get("recent_activity_window_seconds", 45.0) or 45.0)
            return await self.controller.generate_autonomous_actions(window)
        return []

    async def send_actions(self, actions: List[Action]) -> None:
        if not actions:
            return
        payload = build_chat_response(actions)
        message = encode_message("chat_response", payload)
        async with self._write_lock:
            self.writer.write(message.encode("utf-8"))
            await self.writer.drain()

    async def send_status(self, payload: Dict[str, Any]) -> None:
        message = encode_message("status", payload)
        async with self._write_lock:
            self.writer.write(message.encode("utf-8"))
            await self.writer.drain()

    async def process_queue(self, queue: asyncio.Queue[Tuple[str, Dict[str, Any]]]) -> None:
        while True:
            msg_type, data = await queue.get()
            if msg_type != "process_chat" or self.batch_window_seconds <= 0 or self.batch_max_size <= 1:
                try:
                    actions = await self.handle_message(msg_type, data)
                    await self.send_actions(actions)
                finally:
                    queue.task_done()
                continue

            batch = [data]
            processed_count = 1
            loop = asyncio.get_running_loop()
            start = loop.time()

            while len(batch) < self.batch_max_size:
                remaining = self.batch_window_seconds - (loop.time() - start)
                if remaining <= 0:
                    break
                try:
                    next_msg = await asyncio.wait_for(queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                next_type, next_data = next_msg
                if next_type == "process_chat":
                    batch.append(next_data)
                    processed_count += 1
                    continue
                try:
                    actions = await self.handle_message(next_type, next_data)
                    await self.send_actions(actions)
                finally:
                    queue.task_done()

            try:
                actions = await self.controller.process_chat_batch(batch)
                await self.send_actions(actions)
            finally:
                for _ in range(processed_count):
                    queue.task_done()


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}
    return bool(value)
