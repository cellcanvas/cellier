"""``InteractionTracker``: one test per row of the design's 4.1 table."""

from __future__ import annotations

from uuid import uuid4

from cellier._interaction import InteractionState, InteractionTracker, Transition

SETTLE = 0.15


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _tracker() -> tuple[InteractionTracker, _Clock]:
    clock = _Clock()
    return InteractionTracker(SETTLE, clock), clock


def test_starts_idle_with_no_deadline() -> None:
    tracker, _clock = _tracker()
    assert tracker.state is InteractionState.IDLE
    assert tracker.deadline is None
    assert not tracker.scope_open()


# -- interactive ticks --------------------------------------------------------


def test_an_interactive_tick_starts_and_sets_the_deadline() -> None:
    tracker, clock = _tracker()
    source = uuid4()
    assert tracker.tick(source, interactive=True) == [Transition("start", source)]
    assert tracker.state is InteractionState.ACTIVE
    assert tracker.deadline == clock.now + SETTLE


def test_an_interactive_tick_while_active_only_moves_the_deadline() -> None:
    tracker, clock = _tracker()
    source = uuid4()
    tracker.tick(source, interactive=True)
    clock.now += 0.1
    assert tracker.tick(source, interactive=True) == []
    assert tracker.state is InteractionState.ACTIVE
    assert tracker.deadline == clock.now + SETTLE


def test_the_settle_time_is_read_at_each_tick() -> None:
    tracker, clock = _tracker()
    tracker.tick(uuid4(), interactive=True)
    tracker.settle_s = 0.5
    tracker.tick(uuid4(), interactive=True)
    assert tracker.deadline == clock.now + 0.5


# -- jumps --------------------------------------------------------------------


def test_a_jump_while_idle_does_nothing() -> None:
    tracker, _clock = _tracker()
    assert tracker.tick(uuid4(), interactive=False) == []
    assert tracker.state is InteractionState.IDLE
    assert tracker.deadline is None


def test_a_jump_while_active_ends_with_jump() -> None:
    tracker, _clock = _tracker()
    tracker.tick(uuid4(), interactive=True)
    jumper = uuid4()
    assert tracker.tick(jumper, interactive=False) == [
        Transition("end", jumper, "jump")
    ]
    assert tracker.state is InteractionState.IDLE
    assert tracker.deadline is None


# -- scopes -------------------------------------------------------------------


def test_a_scope_makes_a_plain_tick_interactive() -> None:
    tracker, _clock = _tracker()
    holder, ticker = uuid4(), uuid4()
    assert tracker.begin_scope(holder) == []
    assert tracker.state is InteractionState.IDLE  # a scope starts nothing
    assert tracker.tick(ticker, interactive=False) == [Transition("start", ticker)]


def test_closing_the_last_scope_ends_with_release() -> None:
    tracker, _clock = _tracker()
    holder = uuid4()
    tracker.begin_scope(holder)
    tracker.tick(uuid4(), interactive=False)
    assert tracker.end_scope(holder) == [Transition("end", holder, "release")]
    assert tracker.state is InteractionState.IDLE
    assert not tracker.scope_open()


def test_a_scope_with_no_tick_emits_nothing() -> None:
    tracker, _clock = _tracker()
    holder = uuid4()
    assert tracker.begin_scope(holder) == []
    assert tracker.end_scope(holder) == []
    assert tracker.state is InteractionState.IDLE


def test_scopes_are_counted_per_source() -> None:
    tracker, _clock = _tracker()
    first, second = uuid4(), uuid4()
    tracker.begin_scope(first)
    tracker.begin_scope(first)  # the same source twice is one scope
    tracker.begin_scope(second)
    tracker.tick(uuid4(), interactive=False)
    assert tracker.end_scope(first) == []  # the other is still open
    assert tracker.state is InteractionState.ACTIVE
    assert tracker.end_scope(first) == []  # not open: a no-op
    assert tracker.end_scope(second) == [Transition("end", second, "release")]


def test_ending_a_scope_that_is_not_open_does_nothing() -> None:
    tracker, _clock = _tracker()
    tracker.tick(uuid4(), interactive=True)
    assert tracker.end_scope(uuid4()) == []
    assert tracker.state is InteractionState.ACTIVE


def test_an_interactive_tick_with_no_scope_is_not_ended_by_another_scope() -> None:
    """A scope opened after the first tick still ends the scrub on release."""
    tracker, _clock = _tracker()
    holder = uuid4()
    tracker.tick(uuid4(), interactive=True)
    tracker.begin_scope(holder)
    assert tracker.end_scope(holder) == [Transition("end", holder, "release")]


# -- the timer ----------------------------------------------------------------


def test_expire_at_the_deadline_ends_with_settle() -> None:
    tracker, clock = _tracker()
    source = uuid4()
    tracker.tick(source, interactive=True)
    clock.now += SETTLE
    assert tracker.expire() == [Transition("end", source, "settle")]
    assert tracker.state is InteractionState.IDLE
    assert tracker.deadline is None


def test_a_stale_timer_does_nothing() -> None:
    tracker, clock = _tracker()
    tracker.tick(uuid4(), interactive=True)
    clock.now += SETTLE / 2
    assert tracker.expire() == []  # before the deadline
    assert tracker.state is InteractionState.ACTIVE
    tracker.tick(uuid4(), interactive=True)  # moves the deadline
    clock.now += SETTLE / 2 + 0.01  # past the old deadline, not the new one
    assert tracker.expire() == []
    assert tracker.state is InteractionState.ACTIVE


def test_expire_while_idle_does_nothing() -> None:
    tracker, clock = _tracker()
    assert tracker.expire() == []
    tracker.tick(uuid4(), interactive=True)
    tracker.cancel(uuid4())
    clock.now += 10
    assert tracker.expire() == []


def test_an_open_scope_does_not_stop_the_timer() -> None:
    """Holding still settles; the scope stays, and the next tick starts again."""
    tracker, clock = _tracker()
    holder, ticker = uuid4(), uuid4()
    tracker.begin_scope(holder)
    tracker.tick(ticker, interactive=False)
    clock.now += SETTLE
    assert tracker.expire() == [Transition("end", ticker, "settle")]
    assert tracker.scope_open()
    assert tracker.tick(ticker, interactive=False) == [Transition("start", ticker)]
    assert tracker.end_scope(holder) == [Transition("end", holder, "release")]


# -- cancel -------------------------------------------------------------------


def test_cancel_while_active_ends_with_cancel() -> None:
    tracker, _clock = _tracker()
    tracker.tick(uuid4(), interactive=True)
    canceller = uuid4()
    assert tracker.cancel(canceller) == [Transition("end", canceller, "cancel")]
    assert tracker.state is InteractionState.IDLE


def test_cancel_while_idle_does_nothing_and_keeps_scopes() -> None:
    tracker, _clock = _tracker()
    holder = uuid4()
    tracker.begin_scope(holder)
    assert tracker.cancel(uuid4()) == []
    assert tracker.scope_open()


def test_no_input_returns_an_end_and_a_start_together() -> None:
    """The list return leaves room for it; nothing produces it today."""
    tracker, clock = _tracker()
    holder = uuid4()
    results = [
        tracker.begin_scope(holder),
        tracker.tick(uuid4(), interactive=True),
        tracker.tick(uuid4(), interactive=False),
        tracker.cancel(uuid4()),
        tracker.tick(uuid4(), interactive=True),
        tracker.end_scope(holder),
        tracker.tick(uuid4(), interactive=True),
        tracker.tick(uuid4(), interactive=False),
    ]
    clock.now += 1
    results.append(tracker.expire())
    assert all(len(transitions) <= 1 for transitions in results)
