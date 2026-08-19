"""Supervised background tasks: no silent failures, no GC loss, no added latency.

Background work is deliberately off the response path (~150ms/turn saved). These tests
pin the two properties that bare `asyncio.create_task` did not provide.
"""

from __future__ import annotations

import asyncio
import gc
import logging

from gpt5_roleplay_system.background import (
    drain_background_tasks,
    pending_background_tasks,
    spawn_supervised,
)


def test_failing_background_task_is_logged(caplog):
    async def main():
        async def boom():
            raise RuntimeError("neo4j write failed")

        with caplog.at_level(logging.ERROR, logger="gpt5_roleplay_background"):
            spawn_supervised(boom(), label="experience_embedding", session_id="session_1")
            await drain_background_tasks()

    asyncio.run(main())
    assert any("experience_embedding" in r.message for r in caplog.records)
    assert any("session_1" in r.message for r in caplog.records)
    assert any("neo4j write failed" in str(r.message) for r in caplog.records)


def test_quiet_tasks_do_not_log_at_error(caplog):
    """The error-broadcast path must not log its own failure at ERROR (feedback loop)."""
    async def main():
        async def boom():
            raise RuntimeError("broadcast failed")

        with caplog.at_level(logging.ERROR, logger="gpt5_roleplay_background"):
            spawn_supervised(boom(), label="error_broadcast", quiet=True)
            await drain_background_tasks()

    asyncio.run(main())
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_task_reference_is_held_until_completion():
    """asyncio keeps only a weak ref; an unreferenced task can be GC'd mid-flight."""
    async def main():
        phases = []

        async def slow():
            phases.append("start")
            await asyncio.sleep(0.05)
            phases.append("end")

        spawn_supervised(slow(), label="slow")   # handle deliberately discarded
        assert pending_background_tasks() >= 1
        gc.collect()                              # force the risk window
        await drain_background_tasks()
        return phases

    phases = asyncio.run(main())
    assert phases == ["start", "end"]


def test_completed_tasks_are_released():
    """The strong-reference set must not leak."""
    async def main():
        async def noop():
            return None

        for _ in range(50):
            spawn_supervised(noop(), label="noop")
        await drain_background_tasks()
        return pending_background_tasks()

    assert asyncio.run(main()) == 0


def test_cancellation_is_not_reported_as_failure(caplog):
    async def main():
        async def forever():
            await asyncio.sleep(3600)

        with caplog.at_level(logging.ERROR, logger="gpt5_roleplay_background"):
            task = spawn_supervised(forever(), label="autonomy")
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.sleep(0)

    asyncio.run(main())
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_spawning_does_not_block_the_caller():
    """Supervision must not put the work back on the response path."""
    async def main():
        async def slow():
            await asyncio.sleep(0.2)

        loop = asyncio.get_running_loop()
        start = loop.time()
        spawn_supervised(slow(), label="slow")
        elapsed = loop.time() - start
        await drain_background_tasks()
        return elapsed

    assert asyncio.run(main()) < 0.05


def test_success_path_logs_nothing(caplog):
    async def main():
        async def ok():
            return "fine"

        with caplog.at_level(logging.DEBUG, logger="gpt5_roleplay_background"):
            spawn_supervised(ok(), label="store_facts")
            await drain_background_tasks()

    asyncio.run(main())
    assert not caplog.records
