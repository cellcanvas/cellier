"""Observable state for a viewer's startup sequence.

Showing a viewer is not one step but four: the canvas has to reach the browser
or the screen, render a frame, load its data, and commit it.  Until now the
only thing a caller could observe was the happy ending -- ``on_ready`` fired
once everything had worked, and nothing at all fired when it had not.  A
viewer that never finished starting looked exactly like one still working.

This module makes the sequence answerable.  `StartupTracker` records
where each scene has got to, so ``viewer.startup_state`` can be *read* at any
moment with no callback, no event loop and no front end -- which is what makes
a blank panel diagnosable from a notebook cell instead of from a debugger.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from cellier.events._bus import fire_registered, inactive_handle, register_callback

if TYPE_CHECKING:
    from uuid import UUID

    from cellier.events import SubscriptionHandle


class StartupState(str, Enum):
    """How far a scene (or a whole viewer) has got through startup.

    Read it from ``Viewer.startup_state`` (the whole viewer) or
    ``Viewer.scene_startup_states`` (per scene; one per panel of an
    ``OrthoViewer``).  A scene goes ``IDLE`` -> ``WAITING_FOR_CANVAS`` ->
    ``WAITING_FOR_FRAME`` -> ``LOADING`` -> ``READY``, or ends in
    ``STALLED`` when it makes no progress within the stall window: 30 s by
    default, set with ``display(..., stall_timeout=)``.  ``run``, ``launch``
    and ``show`` always use the default.

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
    `StartupTracker.stalled_at` state so the report can say *what* it
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
            fire_registered(self._scene_callbacks.get(key, []), once=True)
            ready, total = self.progress
            fire_registered(self._progress_callbacks, ready, total)

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
        fire_registered(self._stalled_callbacks, dict(stalled))

    def forget(self, key: str) -> None:
        """Drop scene *key*, which was removed: it no longer counts.

        The aggregate state, progress and stall report then cover the
        remaining scenes only, and *key*'s pending ``on_scene_ready``
        callbacks are dropped unfired (the controller cancels its own
        readiness callbacks for a removed scene the same way).  Progress
        callbacks fire with the new totals.
        """
        if key not in self._states:
            return
        del self._states[key]
        self._stalled_at.pop(key, None)
        self._canvas_ids.pop(key, None)
        # Clear in place: the handles hold this list, so they go inactive.
        self._scene_callbacks.pop(key, []).clear()
        ready, total = self.progress
        fire_registered(self._progress_callbacks, ready, total)

    def set_canvas(self, key: str, canvas_id: UUID) -> None:
        """Record which canvas serves *key*, for diagnostics."""
        self._canvas_ids[key] = canvas_id

    @property
    def canvas_ids(self) -> dict[str, UUID]:
        """Scene key to canvas id, for correlating with canvas events."""
        return dict(self._canvas_ids)

    # ── subscribing ──────────────────────────────────────────────────────────

    def on_scene_ready(
        self, key: str, callback: Callable[[], None]
    ) -> SubscriptionHandle:
        """Fire *callback* once when scene *key* is ready; now if it is.

        The handle is active until the callback runs.
        """
        if self._states.get(key) is StartupState.READY:
            callback()
            return inactive_handle()
        return register_callback(self._scene_callbacks.setdefault(key, []), callback)

    def on_progress(self, callback: Callable[[int, int], None]) -> SubscriptionHandle:
        """Fire ``callback(ready, total)`` each time a scene becomes ready."""
        return register_callback(self._progress_callbacks, callback)

    def on_stalled(self, callback: Callable[[dict], None]) -> SubscriptionHandle:
        """Fire ``callback({scene key: state})`` if startup times out."""
        return register_callback(self._stalled_callbacks, callback)


# ── the viewers' hooks ───────────────────────────────────────────────────────


def started_tracker(viewer: Any, hook: str) -> StartupTracker:
    """The viewer's startup tracker; raise if the viewer has not started.

    Shared by the ``Viewer`` and ``OrthoViewer`` hooks that need a tracker.
    """
    tracker = getattr(viewer, "_startup", None)
    if tracker is None:
        raise RuntimeError(
            f"{hook} requires a started viewer; call it after "
            "display()/launch()/show()."
        )
    return tracker


def on_viewer_ready(viewer: Any, callback: Callable[[], None]) -> SubscriptionHandle:
    """Register *callback* for when every scene of *viewer* is ready.

    Before startup has finished the callback joins the viewer's
    ``_ready_callbacks``, which the launcher fires once, when the last scene
    is ready.  After that it runs now, and the handle is already inactive.
    """
    tracker = getattr(viewer, "_startup", None)
    if tracker is not None and tracker.state is StartupState.READY:
        callback()
        return inactive_handle()
    return register_callback(viewer._ready_callbacks, callback)
