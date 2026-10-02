"""The interaction tracker: "the user is scrubbing the dims / moving the camera".

Pure: no asyncio and no cellier imports.  The controller keeps one tracker
per scene for the dims (a *scrub*) and one per canvas for the camera (a
*motion*), and acts on the transitions a tracker returns.  The timers live in
``cellier._interaction_driver``.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID


class InteractionState(Enum):
    """Whether an interaction is in progress."""

    IDLE = "idle"
    ACTIVE = "active"


#: Why an interaction ended.  ``"release"``: the last open scope closed.
#: ``"settle"``: no tick for the stillness time.  ``"jump"``: a tick that was
#: not interactive.  ``"cancel"``: the view it belonged to was replaced.
InteractionEndReason = Literal["release", "settle", "jump", "cancel"]


class Transition(NamedTuple):
    """A change of state returned by an :class:`InteractionTracker`.

    Attributes
    ----------
    phase : {"start", "end"}
        ``"start"`` for ``IDLE -> ACTIVE``, ``"end"`` for ``ACTIVE -> IDLE``.
    source_id : UUID
        The source of the tick or scope that caused it.  A ``"settle"`` end
        carries the source of the interaction's last tick.
    reason : {"release", "settle", "jump", "cancel"} or None
        Why the interaction ended.  ``None`` for a start.
    """

    phase: Literal["start", "end"]
    source_id: UUID
    reason: InteractionEndReason | None = None


class InteractionTracker:
    """Two states, ``IDLE`` and ``ACTIVE``, driven by ticks and scopes.

    A tick is **interactive** when it is marked so or while a scope is open;
    otherwise it is a **jump**.

    - An interactive tick starts an interaction in ``IDLE`` and sets the
      stillness deadline; in ``ACTIVE`` it only moves the deadline.
    - A jump does nothing in ``IDLE`` and ends an interaction (``"jump"``).
    - Closing the last open scope ends an interaction (``"release"``).
    - ``expire()`` at or after the deadline ends it (``"settle"``); before
      the deadline (a stale timer) it does nothing.
    - ``cancel()`` ends it (``"cancel"``).

    Opening a scope never starts an interaction; the first interactive tick
    does.  An open scope does not stop the stillness timer: holding the input
    still settles, the scope stays open, and the next tick starts a new
    interaction.  Scopes are counted per source: beginning twice from one
    source is one scope, and ending a scope that is not open does nothing.

    Parameters
    ----------
    settle_s : float
        Stillness, in seconds, after which :meth:`expire` ends an
        interaction.  May be reassigned; it is read at each interactive tick.
    clock : Callable[[], float]
        Monotonic time in seconds.

    Attributes
    ----------
    state : InteractionState
        The current state.
    deadline : float or None
        Clock time at which stillness ends the interaction.  ``None`` in
        ``IDLE``.
    """

    def __init__(
        self, settle_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.settle_s = settle_s
        self._clock = clock
        self.state = InteractionState.IDLE
        self.deadline: float | None = None
        self._scopes: set[UUID] = set()
        self._last_source: UUID | None = None

    def scope_open(self) -> bool:
        """Whether any source holds a scope open."""
        return bool(self._scopes)

    def tick(self, source_id: UUID, *, interactive: bool) -> list[Transition]:
        """Record one change of the tracked input.

        Parameters
        ----------
        source_id : UUID
            Who made the change.
        interactive : bool
            Whether the change is part of an interaction.  A tick inside an
            open scope is interactive whatever this says.

        Returns
        -------
        list[Transition]
            A start for the first interactive tick, an end (``"jump"``) for a
            jump during an interaction, otherwise nothing.
        """
        if interactive or self._scopes:
            self._last_source = source_id
            self.deadline = self._clock() + self.settle_s
            if self.state is InteractionState.IDLE:
                self.state = InteractionState.ACTIVE
                return [Transition("start", source_id)]
            return []
        if self.state is InteractionState.ACTIVE:
            return self._end(source_id, "jump")
        return []

    def begin_scope(self, source_id: UUID) -> list[Transition]:
        """Open *source_id*'s scope.  Never a transition by itself."""
        self._scopes.add(source_id)
        return []

    def end_scope(self, source_id: UUID) -> list[Transition]:
        """Close *source_id*'s scope; ends an interaction if it was the last."""
        if source_id not in self._scopes:
            return []
        self._scopes.discard(source_id)
        if not self._scopes and self.state is InteractionState.ACTIVE:
            return self._end(source_id, "release")
        return []

    def cancel(self, source_id: UUID) -> list[Transition]:
        """End an interaction because its view was replaced.  Scopes stay."""
        if self.state is InteractionState.ACTIVE:
            return self._end(source_id, "cancel")
        return []

    def expire(self) -> list[Transition]:
        """The stillness timer fired: end the interaction if it is due.

        A stale timer (one that wakes before the deadline, or after the
        interaction already ended) does nothing.
        """
        if self.state is not InteractionState.ACTIVE or self.deadline is None:
            return []
        if self._clock() < self.deadline:
            return []
        assert self._last_source is not None
        return self._end(self._last_source, "settle")

    def _end(self, source_id: UUID, reason: InteractionEndReason) -> list[Transition]:
        self.state = InteractionState.IDLE
        self.deadline = None
        return [Transition("end", source_id, reason)]
