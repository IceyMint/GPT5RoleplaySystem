"""Supervised background tasks.

Background work (Neo4j writes, embeddings, fact storage, state saves) is deliberately
kept off the request path -- awaiting it inline costs ~150ms per turn. That design is
correct; this module fixes the two hazards of doing it with bare `asyncio.create_task`:

1. **Garbage collection.** asyncio only holds a *weak* reference to a running task.
   A task whose handle is discarded may be collected mid-execution, silently dropping
   the write. We keep a strong reference until the task completes.

2. **Silent failure.** An exception in a fire-and-forget task surfaces nowhere the
   caller can see; Python prints "Task exception was never retrieved" at GC time, on
   stderr, with no context. We attach a done-callback that logs the failure with a
   label so it lands in normal logs.

Supervision costs no latency: attaching a callback does not touch the response path.

This module is a leaf -- it imports nothing from the package -- so it is safe to use
from any layer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Set

logger = logging.getLogger("gpt5_roleplay_background")

# Strong references to in-flight tasks, so they cannot be garbage collected while
# running. Entries are removed as soon as each task completes.
_background_tasks: Set[asyncio.Task] = set()


def spawn_supervised(
    coro: Awaitable[Any],
    *,
    label: str,
    session_id: str = "",
    quiet: bool = False,
    on_error: Callable[[BaseException], None] | None = None,
) -> asyncio.Task:
    """Schedule `coro` in the background with a strong reference and error logging.

    Args:
        label: short identifier used in log messages (e.g. "experience_embedding").
        session_id: optional session identifier for correlating logs.
        quiet: log failures at DEBUG instead of ERROR. Use for tasks that are
            themselves part of error reporting, to avoid a logging feedback loop.
        on_error: optional extra callback invoked with the exception.

    Returns:
        The created task. Callers that need to await or cancel it may keep the handle;
        callers that don't can safely discard it.
    """
    task = asyncio.ensure_future(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    task.add_done_callback(
        lambda finished: _report(finished, label=label, session_id=session_id, quiet=quiet, on_error=on_error)
    )
    return task


def _report(
    task: asyncio.Task,
    *,
    label: str,
    session_id: str,
    quiet: bool,
    on_error: Callable[[BaseException], None] | None,
) -> None:
    if task.cancelled():
        # Cancellation is normal during shutdown/disconnect; not a failure.
        return
    try:
        error = task.exception()
    except asyncio.CancelledError:
        return
    if error is None:
        return
    where = f" [{session_id}]" if session_id else ""
    if quiet:
        logger.debug("Background task %s%s failed: %s", label, where, error, exc_info=error)
    else:
        logger.error("Background task %s%s failed: %s", label, where, error, exc_info=error)
    if on_error is not None:
        try:
            on_error(error)
        except Exception:  # pragma: no cover - defensive
            logger.debug("on_error handler for %s raised", label, exc_info=True)


def pending_background_tasks() -> int:
    """Number of supervised tasks currently in flight (diagnostics/tests)."""
    return len(_background_tasks)


async def drain_background_tasks(timeout: float = 5.0) -> None:
    """Wait for in-flight supervised tasks to finish. Intended for shutdown/tests."""
    if not _background_tasks:
        return
    pending = list(_background_tasks)
    done, still_pending = await asyncio.wait(pending, timeout=timeout)
    for task in still_pending:
        logger.warning("Background task did not finish within %.1fs during drain", timeout)
