"""Regression tests for confirmed data-loss bugs.

Each test asserts an invariant that was VIOLATED by the code before the fix, and each
was proven load-bearing by re-introducing the original bug and confirming the test
goes red. Naming maps to the audit findings.

The unifying invariant: any queue that is the ONLY copy of un-processed history
(overflow, rolling buffer, pending fact evidence) must never be truncated or dropped
on a failure path -- it may only shrink after the content has actually been
summarized/stored.
"""
from __future__ import annotations

import asyncio
import threading

from gpt5_roleplay_system.config import EpisodeConfig, FactsConfig
from gpt5_roleplay_system.controller import SessionController
from gpt5_roleplay_system.llm import LLMClient, LLMResponseBundle
from gpt5_roleplay_system.memory import (
    ConversationMemory,
    ExperienceStore,
    MemoryItem,
    RollingBuffer,
    SimpleMemoryCompressor,
)
from gpt5_roleplay_system.models import Action, CommandType, InboundChat, Participant
from gpt5_roleplay_system.neo4j_store import InMemoryKnowledgeStore
from gpt5_roleplay_system.observability import NoOpTracer
from gpt5_roleplay_system.pipeline import MessagePipeline
from gpt5_roleplay_system.session_state import SessionStateStore


def _chat(text: str, ts: float, sid: str = "u", name: str = "User") -> InboundChat:
    return InboundChat(text=text, sender_id=sid, sender_name=name, timestamp=ts, raw={})


def _payload(text: str, ts: float) -> dict:
    return {"text": text, "from_name": "User", "from_id": "u", "timestamp": ts}


class _BaseLLM(LLMClient):
    async def is_addressed_to_me(self, *a, **k):
        return True

    async def generate_response(self, chat, context):
        raise NotImplementedError

    async def extract_facts(self, chat, context):
        return []

    async def summarize_overflow(self, summary, messages):
        return "SUMMARY"

    async def summarize_episode(self, messages):
        return "EPISODE"

    async def generate_bundle(self, chat, context, overflow=None, incoming_batch=None):
        action = Action(command_type=CommandType.CHAT, content="ok", parameters={"content": "ok"})
        return LLMResponseBundle(text="ok", actions=[action], facts=[], participant_hints=[])


def _pipeline(llm, *, memory=None, buf=None, facts=None, episode=None, tracer=None):
    return MessagePipeline(
        persona="P",
        user_id="ai-1",
        llm=llm,
        knowledge_store=InMemoryKnowledgeStore(),
        memory=memory or ConversationMemory(SimpleMemoryCompressor(), max_recent=2, defer_compression=True),
        rolling_buffer=buf or RollingBuffer(max_items=100),
        experience_store=ExperienceStore(),
        tracer=tracer or NoOpTracer(),
        facts_config=facts or FactsConfig(enabled=False),
        episode_config=episode or EpisodeConfig(enabled=False),
    )


# ---------------------------------------------------------------- V1
def test_V1_overflow_survives_a_failing_bundle_call():
    class _FailingBundleLLM(_BaseLLM):
        async def generate_bundle(self, chat, context, overflow=None, incoming_batch=None):
            raise RuntimeError("bundle provider 500")

    async def main():
        memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=1, defer_compression=True)
        pipe = _pipeline(_FailingBundleLLM(), memory=memory)
        pipe.set_llm_chat_enabled(True)
        for i in range(4):
            memory.add_message(_chat(f"old-{i}", 1000.0 + i))
        before = [m.text for m in memory._overflow]

        try:
            await pipe.process_chat_batch([_payload("new message", 2000.0)])
        except RuntimeError as exc:
            print("V1 bundle raised:", exc)

        after = [m.text for m in memory._overflow]
        print("V1 overflow_before=", before, "overflow_after=", after)
        lost = [t for t in before if t not in after]
        print("V1 LOST=", lost)
        assert not lost, "drained overflow was destroyed by a failing LLM call"

    asyncio.run(main())


# ---------------------------------------------------------------- V3
def test_V3_persona_switch_does_not_leak_history(tmp_path):
    async def main():
        controller = SessionController(
            persona="PersonaA",
            user_id="ai-1",
            episode_config=EpisodeConfig(persist_state=True, state_dir=str(tmp_path)),
        )
        controller._memory.add_message(_chat("A-PRIVATE-HISTORY", 1.0))

        controller.set_persona("PersonaB")
        after = [m.text for m in controller._memory.recent()]
        await controller.flush_state()
        b_disk = [
            m.text
            for m in SessionStateStore(str(tmp_path), "PersonaB", "ai-1").load()["memory"]["recent"]
        ]
        print("V3 after_switch=", after, "B_disk=", b_disk)

        assert "A-PRIVATE-HISTORY" not in after, "persona switch kept the previous persona's history live"
        assert "A-PRIVATE-HISTORY" not in b_disk, "persona A history was written into persona B's state file"

    asyncio.run(main())


# ---------------------------------------------------------------- V4
def test_V4_batch_arrivals_all_reach_episodic_memory():
    class _RecordingLLM(_BaseLLM):
        def __init__(self):
            self.summarized: list[str] = []

        async def summarize_episode(self, messages):
            self.summarized = [m.text for m in messages]
            return "EPISODE"

    async def main():
        buf = RollingBuffer(max_items=5)
        llm = _RecordingLLM()
        pipe = _pipeline(
            llm,
            buf=buf,
            episode=EpisodeConfig(
                enabled=True, min_messages=2, max_messages=5,
                inactivity_seconds=0, forced_interval_seconds=0, overlap_messages=1,
            ),
        )
        pipe.set_llm_chat_enabled(False)

        await pipe.process_chat_batch([_payload(f"b{i}", 1000.0 + i) for i in range(8)])
        if pipe._episode_task:
            await pipe._episode_task

        retained = [m.text for m in buf.items()]
        lost = [f"b{i}" for i in range(8) if f"b{i}" not in llm.summarized and f"b{i}" not in retained]
        print("V4 summarized=", llm.summarized, "retained=", retained, "LOST=", lost)
        assert not lost, "batched messages were evicted before episodic summarization"

    asyncio.run(main())


# ---------------------------------------------------------------- V6
def test_V6_fact_worker_requeues_tail_on_any_exception():
    class _FactsLLM(_BaseLLM):
        def extract_facts_from_evidence_sync(self, context, evidence, participants):
            return []

        def consume_reasoning_trace(self, label):
            return None

        def consume_quality_fallback_events(self):
            return []

    class _ExplodingTracer(NoOpTracer):
        def log_event(self, name, payload):
            if name == "facts_sweep":
                raise RuntimeError("tracer exploded after extraction")

    async def main():
        pipe = _pipeline(
            _FactsLLM(),
            facts=FactsConfig(enabled=True, mode="periodic", evidence_max_messages=4, min_pending_messages=1),
            tracer=_ExplodingTracer(),
        )
        participant = Participant(user_id="u", name="User")
        messages = [_chat(f"e{i}", 1000.0 + i) for i in range(9)]
        pipe._enqueue_fact_messages(messages, [participant])

        pipe._maybe_start_facts_worker()
        task = pipe._facts_task
        try:
            await task
        except RuntimeError as exc:
            print("V6 worker raised:", exc)

        pending = [m.text for m in pipe._facts_pending_messages]
        lost = [m.text for m in messages[4:] if m.text not in pending]
        print("V6 pending=", pending, "LOST_TAIL=", lost)
        assert not lost, "unprocessed fact evidence was destroyed by a post-extraction exception"

    asyncio.run(main())


# ---------------------------------------------------------------- V7
def test_V7_state_save_requested_during_inflight_save_is_not_dropped(tmp_path):
    async def main():
        controller = SessionController(
            persona="Isabella",
            user_id="ai-1",
            episode_config=EpisodeConfig(persist_state=True, state_dir=str(tmp_path)),
        )
        store = controller._state_store
        release = threading.Event()
        original_save = store.save
        saved_versions: list[list[str]] = []

        def blocking_save(state):
            saved_versions.append([m.text for m in state["memory"]["recent"]])
            release.wait(timeout=5)
            original_save(state)

        store.save = blocking_save  # type: ignore[method-assign]

        controller._memory.add_message(_chat("FIRST", 1.0))
        controller._schedule_state_save()
        await asyncio.sleep(0.05)

        controller._memory.add_message(_chat("SECOND-WHILE-SAVE-INFLIGHT", 2.0))
        controller._schedule_state_save()

        release.set()
        await controller._state_task

        live = [m.text for m in controller._memory.recent()]
        print("V7 saved_versions=", saved_versions, "live=", live)
        assert any("SECOND-WHILE-SAVE-INFLIGHT" in version for version in saved_versions), (
            "a save requested during an in-flight save was silently dropped"
        )

    asyncio.run(main())


# ---------------------------------------------------------------- V8
def test_V8_episode_watermark_does_not_stall_later_episodes():
    class _RecordingLLM(_BaseLLM):
        def __init__(self):
            self.episodes: list[list[str]] = []

        async def summarize_episode(self, messages):
            self.episodes.append([m.text for m in messages])
            return "EPISODE"

    async def main():
        buf = RollingBuffer(max_items=5)
        llm = _RecordingLLM()
        pipe = _pipeline(
            llm,
            buf=buf,
            episode=EpisodeConfig(
                enabled=True, min_messages=3, max_messages=5,
                inactivity_seconds=0, forced_interval_seconds=0, overlap_messages=4,
            ),
        )
        seen = set()
        for i in range(11):
            buf.add_user_message(_chat(f"w{i}", 1000.0 + i))
            await pipe._maybe_finalize_episode()
            for item in buf.items():
                seen.add(item.text)

        summarized = {text for episode in llm.episodes for text in episode}
        retained = {m.text for m in buf.items()}
        lost = [f"w{i}" for i in range(11) if f"w{i}" not in summarized and f"w{i}" not in retained]
        print("V8 episodes=", llm.episodes)
        print("V8 last_episode_size=", pipe._last_episode_size, "retained=", sorted(retained), "LOST=", lost)
        assert not lost, "episode watermark arithmetic stalled and messages were evicted unsummarized"

    asyncio.run(main())


# ------------------------------------------------ previously confirmed set
def test_V2_set_user_id_preserves_live_state(tmp_path):
    async def main():
        stale = ConversationMemory(SimpleMemoryCompressor(), max_recent=20, defer_compression=True)
        stale.add_message(_chat("stale-on-disk", 1.0))
        SessionStateStore(str(tmp_path), "P", "old-id").save({"rolling_buffer": [], "memory": stale.snapshot()})

        controller = SessionController(
            persona="P",
            user_id="old-id",
            episode_config=EpisodeConfig(persist_state=True, state_dir=str(tmp_path)),
        )
        controller._memory.add_message(_chat("LIVE-UNSAVED", 2.0))
        controller.set_user_id("new-id")
        after = [m.text for m in controller._memory.recent()]
        print("V2 after=", after)
        assert "LIVE-UNSAVED" in after, "set_user_id reloaded disk state over live memory"

    asyncio.run(main())


def test_V5_restore_is_lossless():
    memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=2, defer_compression=True)
    saved = [MemoryItem(f"r{i}", "u", "User", float(i)) for i in range(5)]
    memory.restore({"recent": saved, "overflow": [], "summary": "", "summary_meta": {}})
    alive = [m.text for m in memory.drain_overflow()] + [m.text for m in memory.recent()]
    lost = [m.text for m in saved if m.text not in alive]
    print("V5 alive=", alive, "LOST=", lost)
    assert not lost, "restore discarded saved history"


def test_V9_fact_cursor_keeps_tied_timestamp_message():
    pipe = _pipeline(_BaseLLM(), facts=FactsConfig(enabled=True, mode="periodic"))
    pipe._facts_cursor_ts = 100.0
    tied = _chat("distinct message arriving later", 100.0)
    pipe._enqueue_fact_messages([tied], [Participant(user_id="u", name="User")])
    alive = [m.text for m in pipe._facts_pending_messages]
    print("V9 pending=", alive)
    assert tied.text in alive, "timestamp-only fact cursor dropped distinct evidence"


def test_V10_fact_queue_cap_keeps_oldest_unswept_evidence():
    pipe = _pipeline(
        _BaseLLM(),
        facts=FactsConfig(enabled=True, mode="periodic", evidence_max_messages=4, min_pending_messages=1),
    )
    pipe._facts_retry_not_before_ts = 10**20
    participant = Participant(user_id="u", name="User")
    for i in range(13):
        pipe._enqueue_fact_messages([_chat(f"fact{i}", float(i + 1))], [participant])
        pipe._maybe_start_facts_worker()
    alive = [m.text for m in pipe._facts_pending_messages]
    lost = [f"fact{i}" for i in range(13) if f"fact{i}" not in alive]
    print("V10 pending_count=", len(alive), "LOST=", lost)
    assert not lost, "fact pending cap discarded the oldest unswept evidence"


def test_V11_summary_watermark_keeps_tied_timestamp_messages():
    memory = ConversationMemory(SimpleMemoryCompressor(), max_recent=2, defer_compression=True)
    memory.apply_summary("summary of old event", [100.0])
    pipe = _pipeline(_BaseLLM(), memory=memory)
    for i in range(5):
        memory.add_message(_chat(f"NEW-SAME-TS-{i}", 100.0))
    drained = memory.drain_overflow()
    filtered = pipe._filter_overflow_against_summary(drained)
    print("V11 drained=", [m.text for m in drained], "filtered=", [m.text for m in filtered])
    assert len(filtered) == len(drained), "summary watermark dropped later tied-timestamp messages"


def test_V12_rolling_buffer_cap_does_not_precede_episode_trigger():
    class _RecordingLLM(_BaseLLM):
        def __init__(self):
            self.summarized: list[str] = []

        async def summarize_episode(self, messages):
            self.summarized = [m.text for m in messages]
            return "EPISODE"

    async def main():
        buf = RollingBuffer(max_items=30)
        llm = _RecordingLLM()
        pipe = _pipeline(
            llm,
            buf=buf,
            episode=EpisodeConfig(
                enabled=True, min_messages=8, max_messages=30,
                inactivity_seconds=0, forced_interval_seconds=0, overlap_messages=3,
            ),
        )
        for i in range(31):
            buf.add_user_message(_chat(f"m{i}", float(i)))
        await pipe._maybe_finalize_episode()
        lost = [f"m{i}" for i in range(31) if f"m{i}" not in llm.summarized]
        print("V12 summarized=", len(llm.summarized), "LOST=", lost)
        assert not lost, "rolling-buffer cap dropped items before episode finalization"

    asyncio.run(main())
