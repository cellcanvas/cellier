"""Cancellable one-off reads: the pick-value reader.

Every visual loads through the chunk scheduler, which never cancels; the
only reads left that a caller cancels are the pick-value reads of a
non-memory store (a pointer moved on before the values arrived).
`CancellableReads` runs each as one asyncio task, named by a read id.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from uuid import UUID

_LOGGER = logging.getLogger("cellier.render.pick_reads")


class CancellableReads:
    """One asyncio task per read id; cancelling a read drops its result."""

    def __init__(self) -> None:
        self._tasks: dict[UUID, asyncio.Task] = {}
        self._coros: dict[UUID, Coroutine] = {}

    def submit(
        self,
        read_id: UUID,
        coro: Coroutine[Any, Any, Any],
        on_done: Callable[[Any], None],
    ) -> UUID:
        """Run *coro*; call ``on_done(result)`` when it finishes, never if cancelled.

        A read already running under *read_id* is cancelled first.  A read
        that raises is logged and its ``on_done`` is not called.
        """
        self.cancel(read_id)
        task = asyncio.ensure_future(self._run(read_id, coro, on_done))
        self._tasks[read_id] = task
        self._coros[read_id] = coro
        return read_id

    async def _run(self, read_id, coro, on_done) -> None:
        try:
            result = await coro
        except asyncio.CancelledError:
            raise
        except Exception:
            _LOGGER.exception("pick read %s failed", read_id)
            return
        finally:
            if self._tasks.get(read_id) is asyncio.current_task():
                del self._tasks[read_id]
                self._coros.pop(read_id, None)
        on_done(result)

    def _stop(self, task: asyncio.Task, coro: Coroutine | None) -> bool:
        if task.done():
            return False
        task.cancel()
        if coro is not None and inspect.getcoroutinestate(coro) == "CORO_CREATED":
            # Cancelled before it ran: the read never started.
            coro.close()
        return True

    def cancel(self, read_id: UUID | None) -> bool:
        """Cancel *read_id*; ``False`` when it is unknown or already done."""
        if read_id is None:
            return False
        task = self._tasks.pop(read_id, None)
        coro = self._coros.pop(read_id, None)
        return task is not None and self._stop(task, coro)

    def cancel_all(self) -> int:
        """Cancel every read in flight; return how many were running."""
        cancelled = 0
        for read_id, task in self._tasks.items():
            cancelled += self._stop(task, self._coros.get(read_id))
        self._tasks.clear()
        self._coros.clear()
        return cancelled

    def in_flight(self) -> list[asyncio.Task]:
        """The reads still running."""
        return [task for task in self._tasks.values() if not task.done()]

    def close(self) -> None:
        """Cancel every read; the reader stays usable."""
        self.cancel_all()
