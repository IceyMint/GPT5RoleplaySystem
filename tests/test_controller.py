import asyncio

from gpt5_roleplay_system.config import EpisodeConfig
from gpt5_roleplay_system.controller import SessionController
from gpt5_roleplay_system.llm import EchoLLMClient
from gpt5_roleplay_system.neo4j_store import InMemoryKnowledgeStore
from gpt5_roleplay_system.observability import NoOpTracer


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

