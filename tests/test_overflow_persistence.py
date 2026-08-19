"""Round-trip persistence of un-summarized overflow history.

Invariant under test (independent of implementation): a message that has been ingested
but is not yet described by the summary must still be recoverable after a restart.
Overflow is the only record of such messages, so it has to survive snapshot/restore
and the on-disk state file.
"""

from __future__ import annotations

import json

from gpt5_roleplay_system.memory import ConversationMemory, SimpleMemoryCompressor
from gpt5_roleplay_system.models import InboundChat
from gpt5_roleplay_system.session_state import SessionStateStore


def _chat(text, ts):
    return InboundChat(text=text, sender_id="user-1", sender_name="User", timestamp=ts, raw={})


def _fill(memory, count, start=0):
    for i in range(count):
        memory.add_message(_chat(f"m{start + i}", 1000.0 + start + i))


def test_unsummarized_messages_survive_snapshot_restore():
    memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=2)
    _fill(memory, 20)
    assert memory.summary() == ""  # nothing has been summarized yet

    restored = ConversationMemory(SimpleMemoryCompressor(), max_recent=2)
    restored.restore(memory.snapshot())

    before = [i.text for i in memory._overflow] + [i.text for i in memory.recent()]
    after = [i.text for i in restored._overflow] + [i.text for i in restored.recent()]
    assert after == before
    assert after == [f"m{i}" for i in range(20)]


def test_unsummarized_messages_survive_a_real_state_file(tmp_path):
    memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=2)
    _fill(memory, 30)

    store = SessionStateStore(state_dir=str(tmp_path), persona="P", user_id="u")
    store.save({"rolling_buffer": [], "memory": memory.snapshot()})

    # The state file must be valid JSON containing the pending backlog.
    written = list(tmp_path.glob("session_state_*.json"))
    assert written, "no state file was written"
    raw = json.loads(written[0].read_text())
    assert len(raw["memory"]["overflow"]) == len(memory._overflow)

    reloaded = ConversationMemory(SimpleMemoryCompressor(), max_recent=2)
    reloaded.restore(store.load()["memory"])

    after = [i.text for i in reloaded._overflow] + [i.text for i in reloaded.recent()]
    assert after == [f"m{i}" for i in range(30)]


def test_save_does_not_mutate_the_live_snapshot(tmp_path):
    """save() runs on a background thread against live pipeline state."""
    memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=2)
    _fill(memory, 10)
    snapshot = memory.snapshot()

    store = SessionStateStore(state_dir=str(tmp_path), persona="P", user_id="u")
    store.save({"rolling_buffer": [], "memory": snapshot})

    # The caller's snapshot must still hold MemoryItem objects, not serialized dicts.
    assert all(hasattr(item, "text") for item in snapshot["overflow"])
    assert all(hasattr(item, "text") for item in snapshot["recent"])


def test_restore_tolerates_legacy_state_without_overflow_key():
    """State files written before overflow was persisted must still load."""
    memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=2)
    memory.restore({"recent": [], "summary": "old summary", "summary_meta": {}})
    assert memory.summary() == "old summary"
    assert memory._overflow == []


def test_backlog_still_drains_normally_after_restore(tmp_path):
    """A restored backlog is ordinary pending work, not a permanently stuck queue."""
    memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=2)
    _fill(memory, 12)

    restored = ConversationMemory(SimpleMemoryCompressor(), max_recent=2)
    restored.restore(memory.snapshot())

    drained = restored.drain_overflow()
    assert [i.text for i in drained] == [f"m{i}" for i in range(10)]
    assert restored._overflow == []
