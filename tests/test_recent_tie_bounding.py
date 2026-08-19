"""Bounded tie-preservation in the recent-message window.

Invariants under test (from docs/PAYLOAD_DATA_QUALITY_REMEDIATION_PLAN.md section 5):

* A tied-timestamp group is not split arbitrarily across recent/overflow when avoidable.
* Ordering is deterministic: identical input always produces an identical boundary.
* Overage is *slight and temporary* -- recent() must stay bounded even when every
  inbound message carries the same timestamp, because recent() is sent to the model
  on every turn.
* No message is ever destroyed: everything is either in recent() or in overflow.
"""

from __future__ import annotations

from gpt5_roleplay_system.memory import (
    MAX_RECENT_TIE_OVERAGE_FACTOR,
    ConversationMemory,
    SimpleMemoryCompressor,
)
from gpt5_roleplay_system.models import InboundChat


def _chat(text, ts):
    return InboundChat(text=text, sender_id="u", sender_name="U", timestamp=ts, raw={})


def test_recent_stays_bounded_when_every_timestamp_is_identical():
    max_recent = 5
    memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=max_recent, defer_compression=True)
    for i in range(500):
        memory.add_message(_chat(f"m{i}", 1000.0))  # one shared timestamp

    limit = max_recent * MAX_RECENT_TIE_OVERAGE_FACTOR
    assert len(memory.recent()) <= limit, "recent() must not grow without bound"

    # Nothing may be lost: every message is either recent or pending summarization.
    alive = [i.text for i in memory.drain_overflow()] + [i.text for i in memory.recent()]
    assert alive == [f"m{i}" for i in range(500)]


def test_identical_timestamps_across_ai_messages_stay_bounded():
    """add_ai_message() stamps time.time(); same-tick appends can tie."""
    memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=3, defer_compression=True)
    for i in range(200):
        memory.add_ai_message(f"a{i}", "P")
    assert len(memory.recent()) <= 3 * MAX_RECENT_TIE_OVERAGE_FACTOR


def test_small_tie_group_is_still_kept_together():
    """The documented behaviour -- slight overage to avoid splitting a tie -- is preserved."""
    memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=2, defer_compression=True)
    memory.add_message(_chat("a", 10.0))
    memory.add_message(_chat("b", 10.0))
    memory.add_message(_chat("c", 10.0))
    # Temporary overage: the tie group is held together rather than split.
    assert [i.text for i in memory.recent()] == ["a", "b", "c"]
    assert memory.drain_overflow() == []


def test_forced_split_is_deterministic():
    def run_once():
        memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=4, defer_compression=True)
        for i in range(60):
            memory.add_message(_chat(f"m{i}", 7.0))
        return (
            [i.text for i in memory.drain_overflow()],
            [i.text for i in memory.recent()],
        )

    first_overflow, first_recent = run_once()
    second_overflow, second_recent = run_once()
    assert first_overflow == second_overflow
    assert first_recent == second_recent
    # Boundary must respect chronological order across the split.
    assert first_overflow + first_recent == [f"m{i}" for i in range(60)]


def test_distinct_timestamps_are_unaffected():
    """The normal path must keep exactly max_recent messages."""
    memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=5, defer_compression=True)
    for i in range(50):
        memory.add_message(_chat(f"m{i}", 1000.0 + i))
    assert [i.text for i in memory.recent()] == [f"m{i}" for i in range(45, 50)]
