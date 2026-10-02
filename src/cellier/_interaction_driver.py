"""Timers and bookkeeping for the controller's interaction trackers."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from cellier._interaction import InteractionState, InteractionTracker, Transition

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID


class _InteractionDriver:
    """Own the trackers of one kind and their stillness timers.

    The controller keeps one driver for the dims (keyed by scene id) and one
    for the camera (keyed by canvas id).  The driver feeds inputs to the
    tracker of a key, hands every transition to *on_transition*, and keeps
    one ``asyncio`` task per ``ACTIVE`` tracker that sleeps until the
    tracker's deadline.  A tick only moves the deadline; the task re-reads it
    when it wakes.  A tracker that leaves ``ACTIVE`` any other way (release,
    jump, cancel) has its task cancelled, so no timer outlives its
    interaction.

    With no running event loop there is no timer: the caller ends the
    interaction with :meth:`settle_without_loop` once it has acted on the
    tick.

    Parameters
    ----------
    settle_s : Callable[[], float]
        The stillness time, read at every tick so a changed setting applies
        to the next one.
    on_transition : Callable[[UUID, Transition], None]
        Called with the key and each transition, in order, before the input
        method returns.
    clock : Callable[[], float]
        Monotonic time in seconds, shared with the trackers.
    """

    def __init__(
        self,
        settle_s: Callable[[], float],
        on_transition: Callable[[UUID, Transition], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settle_s = settle_s
        self._on_transition = on_transition
        self._clock = clock
        self._trackers: dict[UUID, InteractionTracker] = {}
        self._timers: dict[UUID, asyncio.Task] = {}

    # -- queries --------------------------------------------------------------

    def is_active(self, key: UUID) -> bool:
        """Whether *key*'s tracker is ``ACTIVE``."""
        tracker = self._trackers.get(key)
        return tracker is not None and tracker.state is InteractionState.ACTIVE

    def scope_open(self, key: UUID) -> bool:
        """Whether any source holds a scope open on *key*."""
        tracker = self._trackers.get(key)
        return tracker is not None and tracker.scope_open()

    def state(self, key: UUID) -> InteractionState:
        """The state of *key*'s tracker (``IDLE`` if it has none yet)."""
        tracker = self._trackers.get(key)
        return InteractionState.IDLE if tracker is None else tracker.state

    def tasks(self) -> list[asyncio.Task]:
        """The timer tasks still pending."""
        return [task for task in self._timers.values() if not task.done()]

    # -- inputs ---------------------------------------------------------------

    def tick(self, key: UUID, source_id: UUID, *, interactive: bool) -> None:
        """Feed one tick to *key*'s tracker."""
        tracker = self._tracker(key)
        tracker.settle_s = self._settle_s()
        self._apply(key, tracker, tracker.tick(source_id, interactive=interactive))

    def begin_scope(self, key: UUID, source_id: UUID) -> None:
        """Open *source_id*'s scope on *key*."""
        tracker = self._tracker(key)
        self._apply(key, tracker, tracker.begin_scope(source_id))

    def end_scope(self, key: UUID, source_id: UUID) -> None:
        """Close *source_id*'s scope on *key*."""
        tracker = self._trackers.get(key)
        if tracker is not None:
            self._apply(key, tracker, tracker.end_scope(source_id))

    def cancel(self, key: UUID, source_id: UUID) -> None:
        """Cancel *key*'s interaction, if one is active."""
        tracker = self._trackers.get(key)
        if tracker is not None:
            self._apply(key, tracker, tracker.cancel(source_id))

    def settle_without_loop(self, key: UUID) -> None:
        """End *key*'s interaction at once when no event loop can time it.

        Does nothing while a loop is running: the timer task ends it.
        """
        tracker = self._trackers.get(key)
        if tracker is None or tracker.state is not InteractionState.ACTIVE:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            tracker.deadline = self._clock()
            self._apply(key, tracker, tracker.expire())

    # -- teardown -------------------------------------------------------------

    def drop(self, key: UUID) -> None:
        """Forget *key*: no transition is reported, and its timer is cancelled."""
        self._trackers.pop(key, None)
        self._cancel_timer(key)

    def close(self) -> None:
        """Forget every key."""
        for key in list(self._trackers):
            self.drop(key)
        for key in list(self._timers):
            self._cancel_timer(key)

    # -- internals ------------------------------------------------------------

    def _tracker(self, key: UUID) -> InteractionTracker:
        tracker = self._trackers.get(key)
        if tracker is None:
            tracker = InteractionTracker(self._settle_s(), self._clock)
            self._trackers[key] = tracker
        return tracker

    def _apply(
        self, key: UUID, tracker: InteractionTracker, transitions: list[Transition]
    ) -> None:
        # The timer first, so a callback that asks for the pending tasks (or
        # re-enters the driver) sees the state the transition left.
        if tracker.state is InteractionState.ACTIVE:
            self._arm(key, tracker)
        else:
            self._cancel_timer(key)
        for transition in transitions:
            self._on_transition(key, transition)

    def _arm(self, key: UUID, tracker: InteractionTracker) -> None:
        existing = self._timers.get(key)
        if existing is not None and not existing.done():
            return  # the running task re-reads the deadline
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop: settle_without_loop ends it
        self._timers[key] = loop.create_task(self._timer(key, tracker))

    async def _timer(self, key: UUID, tracker: InteractionTracker) -> None:
        while tracker.state is InteractionState.ACTIVE and tracker.deadline is not None:
            delay = tracker.deadline - self._clock()
            if delay > 0:
                await asyncio.sleep(delay)
                continue
            if self._trackers.get(key) is tracker:
                self._timers.pop(key, None)
                self._apply(key, tracker, tracker.expire())
            return

    def _cancel_timer(self, key: UUID) -> None:
        task = self._timers.pop(key, None)
        if task is None or task.done():
            return
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if task is not current:
            task.cancel()
