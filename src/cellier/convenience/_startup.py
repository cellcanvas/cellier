"""Observable state for a viewer's startup sequence.

Showing a viewer is not one step but four: the canvas has to reach the browser
or the screen, render a frame, load its data, and commit it.  Until now the
only thing a caller could observe was the happy ending -- ``on_ready`` fired
once everything had worked, and nothing at all fired when it had not.  A
viewer that never finished starting looked exactly like one still working.

This module makes the sequence answerable.  :class:`StartupTracker` records
where each scene has got to, so ``viewer.startup_state`` can be *read* at any
moment with no callback, no event loop and no front end -- which is what makes
a blank panel diagnosable from a notebook cell instead of from a debugger.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from uuid import UUID


class StartupState(str, Enum):
    """How far a scene (or a whole viewer) has got through startup.

    Ordered: a viewer's aggregate state is the least-advanced of its scenes,
    so one stuck panel is never hidden behind three finished ones.
    """

    IDLE = "idle"
    """Nothing armed yet -- ``display``/``launch`` has not run."""

    WAITING_FOR_CANVAS = "waiting_for_canvas"
    """Armed, waiting for the front end to report the canvas is live.

    The state an anywidget viewer sits in when the browser never mounted its
    canvas.  On Qt it is normally over in a single event-loop turn.
    """

    WAITING_FOR_FRAME = "waiting_for_frame"
    """The canvas is live; waiting for it to render its first frame.

    Startup defers the camera fit and the initial load to the first frame,
    because only then does the canvas have its final size.
    """

    LOADING = "loading"
    """The first frame arrived; the initial reslice is in flight."""

    READY = "ready"
    """Data is committed to the GPU and the camera has been fitted."""

    STALLED = "stalled"
    """No progress within the timeout.  Records where it stopped.

    Not an error -- slow remote data legitimately takes a while -- but the
    signal that something is worth looking at.  A stalled scene keeps its
    :attr:`StartupTracker.stalled_at` state so the report can say *what* it
    was waiting for.
    """


#: Ordering used to reduce per-scene states to one viewer-level state.
_ORDER = [
    StartupState.IDLE,
    StartupState.WAITING_FOR_CANVAS,
    StartupState.WAITING_FOR_FRAME,
    StartupState.LOADING,
    StartupState.READY,
]


class StartupTracker:
    """Per-scene startup state, plus the callbacks that observe it.

    One per viewer, created when ``display``/``launch`` arms startup.  Every
    transition is recorded against a scene key, so an ``OrthoViewer`` with one
    stuck panel reports which panel rather than just "not ready".
    """

    def __init__(self, scene_keys: list[str]) -> None:
        self._states: dict[str, StartupState] = dict.fromkeys(
            scene_keys, StartupState.IDLE
        )
        self._stalled_at: dict[str, StartupState] = {}
        self._scene_callbacks: dict[str, list[Callable[[], None]]] = {}
        self._progress_callbacks: list[Callable[[int, int], None]] = []
        self._stalled_callbacks: list[Callable[[dict], None]] = []
        self._canvas_ids: dict[str, UUID] = {}

    # ── reading ──────────────────────────────────────────────────────────────

    @property
    def state(self) -> StartupState:
        """The least-advanced scene's state, or ``STALLED`` if any stalled."""
        if self._stalled_at:
            return StartupState.STALLED
        if not self._states:
            return StartupState.READY
        return min(self._states.values(), key=_ORDER.index)

    @property
    def scene_states(self) -> dict[str, StartupState]:
        """Each scene's state, keyed as the viewer keys its scenes."""
        return {
            key: StartupState.STALLED if key in self._stalled_at else value
            for key, value in self._states.items()
        }

    @property
    def stalled_at(self) -> dict[str, StartupState]:
        """For each stalled scene, the state it stopped in."""
        return dict(self._stalled_at)

    @property
    def progress(self) -> tuple[int, int]:
        """``(scenes ready, scenes total)``."""
        ready = sum(1 for v in self._states.values() if v is StartupState.READY)
        return ready, len(self._states)

    def describe(self) -> str:
        """A one-line human summary, for a notebook cell or a log line."""
        ready, total = self.progress
        detail = ", ".join(
            f"{key}={value.value}" for key, value in sorted(self.scene_states.items())
        )
        return f"{self.state.value} ({ready}/{total} scenes ready) [{detail}]"

    # ── writing ──────────────────────────────────────────────────────────────

    def advance(self, key: str, state: StartupState) -> None:
        """Record that *key* reached *state*, and fire whatever that unblocks.

        Never moves a scene backwards: a late duplicate signal -- a second
        Show event, a redundant frame -- must not undo progress.
        """
        current = self._states.get(key, StartupState.IDLE)
        if _ORDER.index(state) <= _ORDER.index(current):
            return
        self._states[key] = state
        self._stalled_at.pop(key, None)
        if state is StartupState.READY:
            for callback in self._scene_callbacks.get(key, []):
                callback()
            ready, total = self.progress
            for callback in self._progress_callbacks:
                callback(ready, total)

    def mark_stalled(self) -> None:
        """Record every scene that has not finished as stalled, and report."""
        stalled = {
            key: value
            for key, value in self._states.items()
            if value is not StartupState.READY
        }
        if not stalled:
            return
        self._stalled_at = stalled
        for callback in self._stalled_callbacks:
            callback(dict(stalled))

    def set_canvas(self, key: str, canvas_id: UUID) -> None:
        """Record which canvas serves *key*, for diagnostics."""
        self._canvas_ids[key] = canvas_id

    @property
    def canvas_ids(self) -> dict[str, UUID]:
        """Scene key to canvas id, for correlating with canvas events."""
        return dict(self._canvas_ids)

    # ── subscribing ──────────────────────────────────────────────────────────

    def on_scene_ready(self, key: str, callback: Callable[[], None]) -> None:
        """Fire *callback* when scene *key* is ready; immediately if it is."""
        if self._states.get(key) is StartupState.READY:
            callback()
            return
        self._scene_callbacks.setdefault(key, []).append(callback)

    def on_progress(self, callback: Callable[[int, int], None]) -> None:
        """Fire ``callback(ready, total)`` each time a scene becomes ready."""
        self._progress_callbacks.append(callback)

    def on_stalled(self, callback: Callable[[dict], None]) -> None:
        """Fire ``callback({scene key: state})`` if startup times out."""
        self._stalled_callbacks.append(callback)
