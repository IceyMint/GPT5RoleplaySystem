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
    # Overflow is accumulated without naive compression
    assert len(memory.drain_overflow()) == 1


def test_summary_meta_tracks_overflow_timestamps():
    memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=1, defer_compression=False)
    memory.add_message(_chat_at("earlier", 100.0))
    memory.add_message(_chat_at("later", 200.0))
    # Naive compression is no longer applied implicitly; overflow is accumulated instead.
    assert memory.summary() == ""
    
    # We can explicitly test updating summary meta
    memory.apply_summary("new summary", [100.0, 200.0])
    meta = memory.summary_meta()
    assert meta
    assert meta.get("range_start_ts") == 100.0
    assert meta.get("range_end_ts") == 200.0
    assert float(meta.get("last_updated_ts", 0.0) or 0.0) > 0.0


def test_simple_compress_overflow_requeues_drained_items_without_advancing_summary_meta():
    memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=1, defer_compression=True)
    memory.add_message(_chat_at("one", 1.0))
    memory.add_message(_chat_at("two", 2.0))

    drained = memory.drain_overflow()
    assert [item.text for item in drained] == ["one"]

    memory.compress_overflow(drained)

    assert memory.summary() == ""
    assert memory.summary_meta() == {}
    assert [item.text for item in memory.drain_overflow()] == ["one"]


def test_overflow_slicing_preserves_tied_timestamp_group_deterministically():
    def run_once() -> tuple[list[str], list[str]]:
        memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=2, defer_compression=True)
        memory.add_message(_chat_at("a", 10.0))
        memory.add_message(_chat_at("b", 10.0))
        memory.add_message(_chat_at("c", 10.0))
        # Tie preservation allows temporary overage.
        assert [item.text for item in memory.recent()] == ["a", "b", "c"]
        memory.add_message(_chat_at("d", 11.0))
        memory.add_message(_chat_at("e", 11.0))
        overflow = [item.text for item in memory.drain_overflow()]
        recent = [item.text for item in memory.recent()]
        return overflow, recent

    first_overflow, first_recent = run_once()
    second_overflow, second_recent = run_once()
    assert first_overflow == ["a", "b", "c"]
    assert first_recent == ["d", "e"]
    assert second_overflow == first_overflow
    assert second_recent == first_recent


def test_requeue_overflow_prepends_drained_items_before_newer_overflow():
    memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=1, defer_compression=True)
    memory.add_message(_chat_at("one", 1.0))
    memory.add_message(_chat_at("two", 2.0))
    drained = memory.drain_overflow()
    assert [item.text for item in drained] == ["one"]
    memory.add_message(_chat_at("three", 3.0))
    memory.requeue_overflow(drained)
    assert [item.text for item in memory.drain_overflow()] == ["one", "two"]
