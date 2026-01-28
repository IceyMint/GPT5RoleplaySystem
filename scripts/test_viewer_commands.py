import asyncio
import logging
import re
import math
from typing import Any, Dict, List, Optional

from gpt5_roleplay_system.server import GPT5RoleplayServer, _build_knowledge_store, _build_tracer
from gpt5_roleplay_system.config import load_config
from gpt5_roleplay_system.llm import LLMClient, LLMResponseBundle, ExtractedFact
from gpt5_roleplay_system.models import Action, CommandType, InboundChat

logger = logging.getLogger("test_viewer_commands")

class MockLLMClient(LLMClient):
    def __init__(self):
        self.next_bundle: Optional[LLMResponseBundle] = None

    async def is_addressed_to_me(self, *args, **kwargs) -> bool:
        return True

    async def generate_bundle(self, *args, **kwargs) -> LLMResponseBundle:
        if self.next_bundle:
            b = self.next_bundle
            self.next_bundle = None
            return b
        return LLMResponseBundle(text="Default mock response", actions=[])

    async def generate_autonomous_bundle(self, *args, **kwargs) -> LLMResponseBundle:
        return await self.generate_bundle()

    async def summarize(self, summary: str, messages: List[InboundChat]) -> str:
        return summary

    async def extract_facts(self, *args, **kwargs) -> List[ExtractedFact]:
        return []

async def test_coordinator(server: GPT5RoleplayServer, mock_llm: MockLLMClient):
    logger.info("Test Coordinator: Waiting for a viewer to connect...")
    while not server._sessions:
        await asyncio.sleep(1)
    
    # Give it a moment to stabilize
    await asyncio.sleep(2)
    if not server._sessions:
        return
        
    session_id = next(iter(server._sessions))
    session = server._sessions[session_id]
    logger.info(f"Test Coordinator: Viewer connected: {session_id}. Starting command tests in 5 seconds...")
    await asyncio.sleep(5)

    # Access environment to find an object
    pipeline = session.controller._pipeline
    env = pipeline._environment
    
    logger.info("Test Coordinator: Waiting for environment update with objects...")
    while not env.objects:
        await asyncio.sleep(1)
        if session_id not in server._sessions:
            return

    obj_name = "Target Object"
    target_uuid = "00000000-0000-0000-0000-000000000000"
    target_pos = {"x": 128.0, "y": 128.0, "z": 25.0}

    # Find the closest object
    if env.objects:
        avatar_pos_str = env.avatar_position or "(0,0,0)"
        avatar_pos_match = re.findall(r"[-+]?\d*\.\d+|\d+", avatar_pos_str)
        avatar_pos = tuple(float(p) for p in avatar_pos_match[:3]) if avatar_pos_match else (0.0, 0.0, 0.0)

        def get_dist(o):
            try:
                ox = float(o.get("x", 0.0))
                oy = float(o.get("y", 0.0))
                oz = float(o.get("z", 0.0))
                return math.sqrt((ox - avatar_pos[0])**2 + (oy - avatar_pos[1])**2 + (oz - avatar_pos[2])**2)
            except (TypeError, ValueError):
                return 999999.0

        sorted_objects = sorted(env.objects, key=get_dist)
        obj = sorted_objects[0]
        obj_name = obj.get("name", "Target Object")
        target_uuid = obj.get("uuid") or obj.get("target_uuid") or target_uuid
        if "x" in obj and "y" in obj and "z" in obj:
            target_pos["x"] = float(obj["x"])
            target_pos["y"] = float(obj["y"])
            target_pos["z"] = float(obj["z"])
        logger.info(f"Test Coordinator: Found closest object '{obj_name}' at distance {get_dist(obj):.2f}m")
    else:
        logger.warning("Test Coordinator: No objects found in environment. Using dummy data.")

    # Specific sequence requested by user
    # CHAT, EMOTE, MOVE, SIT, TOUCH, STAND
    commands_to_test = [
        (CommandType.CHAT, {
            "content": f"I am going to sit on {obj_name}.",
            "parameters": {"content": f"I am going to sit on {obj_name}."}
        }),
        (CommandType.EMOTE, {
            "content": f"walks over to {obj_name} and prepares to sit.",
            "parameters": {"content": f"walks over to {obj_name} and prepares to sit."}
        }),
        (CommandType.MOVE, {
            "x": target_pos["x"], "y": target_pos["y"], "z": target_pos["z"],
            "parameters": {"x": str(target_pos["x"]), "y": str(target_pos["y"]), "z": str(target_pos["z"])}
        }),
        (CommandType.SIT, {
            "target_uuid": target_uuid
        }),
        (CommandType.TOUCH, {
            "target_uuid": target_uuid
        }),
        (CommandType.STAND, {}),
    ]

    for cmd_type, params in commands_to_test:
        if session_id not in server._sessions:
            logger.warning("Session lost during tests.")
            break
            
        test_desc = f"Testing command: {cmd_type.value}"
        logger.info(f">>> {test_desc}")
        
        # Announce via status channel -9001
        status_action = Action(
            command_type=CommandType.CHAT,
            content=f"[TEST] {test_desc}",
            parameters={"channel": -9001, "content": f"[TEST] {test_desc}"}
        )
        await session.send_actions([status_action])
        await asyncio.sleep(1) # Small pause after announcement

        # Prepare mock response
        action = Action(command_type=cmd_type, **params)
        mock_llm.next_bundle = LLMResponseBundle(
            text=f"Executing {cmd_type.value}",
            actions=[action],
            facts=[],
            participant_hints=[]
        )

        # Trigger pipeline
        # We call handle_message directly to avoid needing access to the private queue
        actions = await session.handle_message("process_chat", {
            "text": f"Trigger {cmd_type.value}",
            "from_name": "TestRunner",
            "from_id": "test-uuid"
        })
        await session.send_actions(actions)

        # Wait as requested
        await asyncio.sleep(5)

    if session_id in server._sessions:
        logger.info("All tests completed.")
        await session.send_actions([Action(
            command_type=CommandType.CHAT,
            content="[TEST] All command tests completed.",
            parameters={"channel": -9001, "content": "[TEST] All command tests completed."}
        )])

async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    # Disable autonomy and persistence for clean testing
    config.autonomy.enabled = False
    config.episode.persist_state = False
    
    knowledge_store = _build_knowledge_store(config)
    mock_llm = MockLLMClient()
    tracer = _build_tracer(config)
    
    server = GPT5RoleplayServer(config.host, config.port, config, knowledge_store, mock_llm, tracer)
    
    logger.info(f"Starting Test Server on {config.host}:{config.port}")
    
    # Start server and coordinator
    server_task = asyncio.create_task(server.start())
    coordinator_task = asyncio.create_task(test_coordinator(server, mock_llm))
    
    try:
        await asyncio.gather(server_task, coordinator_task)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down...")
    finally:
        server_task.cancel()
        coordinator_task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
