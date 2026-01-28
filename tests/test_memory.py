import time

from gpt5_roleplay_system.memory import ConversationMemory, RollingBuffer, SimpleMemoryCompressor
from gpt5_roleplay_system.models import InboundChat


def _chat(text: str) -> InboundChat:
    return InboundChat(
        text=text,
        sender_id="u1",
        sender_name="User",
        timestamp=time.time(),
        raw={},
    )


def _chat_at(text: str, timestamp: float) -> InboundChat:
    return InboundChat(
        text=text,
        sender_id="u1",
        sender_name="User",
        timestamp=timestamp,
        raw={},
    )


def test_rolling_buffer_trims():
    buffer = RollingBuffer(max_items=2)
    buffer.add_user_message(_chat("one"))
    buffer.add_user_message(_chat("two"))
    buffer.add_user_message(_chat("three"))
    assert len(buffer.items()) == 2
    assert buffer.items()[-1].text == "three"


def test_conversation_memory_compresses():
    memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=2)
    memory.add_message(_chat("one"))
    memory.add_message(_chat("two"))
    memory.add_message(_chat("three"))
    assert len(memory.recent()) == 2
    assert "one" in memory.summary()


def test_summary_meta_tracks_overflow_timestamps():
    memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=1, defer_compression=False)
    memory.add_message(_chat_at("earlier", 100.0))
    memory.add_message(_chat_at("later", 200.0))
    meta = memory.summary_meta()
    assert meta
    assert meta.get("range_start_ts") == 100.0
    assert meta.get("range_end_ts") == 100.0
    assert float(meta.get("last_updated_ts", 0.0) or 0.0) > 0.0
