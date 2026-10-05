"""``LevelResidency`` on the real scheduler core, driven synchronously.

``plans/mesh_refactor_v3.md`` L2 and L3: the plan's keys, the release of a
level whose key leaves the plan (D9), the keep rule (D23), and the display
rule (D8): a level is drawable only while it holds the newest plan's key.
"""

from __future__ import annotations

import pytest

from cellier.render import SchedulerConfig
from cellier.render._level_residency import LEVEL_CACHE_POLICY, LevelResidency
from cellier.render.scheduling import SchedulerCore

FINE, COARSE = 0, 1


class Store:
    id = "store"


class Rig:
    """One level cache on a scheduler core, with reads completed by hand."""

    def __init__(
        self, n_levels: int = 1, fail_upload: bool = False, keeps_previous=None
    ) -> None:
        self.uploads: list[tuple[int, object, object]] = []
        self.releases: list[int] = []
        self.changes = 0
        self.fail_upload = fail_upload
        self.residency = LevelResidency(
            n_levels,
            upload=self._upload,
            release=self.releases.append,
            on_change=self._changed,
            keeps_previous=keeps_previous,
        )
        self.core = SchedulerCore(SchedulerConfig())
        self.core.register(self.residency.cache_id, self.residency, scene="scene")
        self.levels = list(range(n_levels))
        self.tickets: list = []

    def _upload(self, level, request_key, data) -> None:
        if self.fail_upload:
            raise ValueError("refused")
        self.uploads.append((level, request_key, data))

    def _changed(self) -> None:
        self.changes += 1

    def plan(self, position, asked=None) -> None:
        """Plan every level at *position*; *asked* defaults to all levels."""
        asked = self.levels if asked is None else asked
        desired = self.residency.desired(
            dict.fromkeys(self.levels, position),
            asked,
            {level: (level, position) for level in self.levels},
        )
        self.desired = desired
        self.core.set_desired(_with_store(desired))
        self.core.process()
        self.tickets.extend(self.core.next_reads())

    def land(self, index: int = 0, error: BaseException | None = None) -> None:
        """Complete one issued read (its data is its request), then commit."""
        ticket = self.tickets.pop(index)
        data = None if error is not None else ("data", *ticket.request)
        self.core.complete_read(ticket, data=data, error=error)
        self.core.commit_round("scene")
        self.tickets.extend(self.core.next_reads())

    def requests_in_flight(self) -> list:
        return [ticket.request for ticket in self.tickets]


def _with_store(desired):
    import dataclasses

    return dataclasses.replace(desired, store=Store())


def test_the_policy_is_the_mesh_policy():
    rig = Rig()
    assert rig.residency.policy is LEVEL_CACHE_POLICY
    assert rig.core.policy_of(rig.residency.cache_id).max_target_fetching == 1
    assert rig.residency.n_slots == 2


def test_a_planned_result_is_uploaded_and_drawable():
    rig = Rig()
    rig.plan("a")
    assert rig.residency.level_to_draw() is None
    rig.land()
    assert rig.uploads == [(FINE, "a", ("data", FINE, "a"))]
    assert rig.residency.is_drawable(FINE)
    assert rig.residency.level_to_draw() == FINE


def test_an_unchanged_plan_reads_nothing():
    rig = Rig()
    rig.plan("a")
    rig.land()
    rig.plan("a")
    assert rig.tickets == []
    assert len(rig.uploads) == 1
    assert rig.core.is_complete(rig.residency.cache_id)


def test_a_new_plan_releases_the_level_when_it_is_made():
    """D9: the release happens in ``desired``, before the pass is applied."""
    rig = Rig()
    rig.plan("a")
    rig.land()
    # Only the plan: nothing has been handed to the scheduler yet.
    rig.residency.desired({FINE: "b"}, [FINE], {FINE: (FINE, "b")})
    assert rig.releases == [FINE]
    assert rig.residency.level_to_draw() is None


def test_returning_to_a_position_reads_again():
    rig = Rig()
    rig.plan("a")
    rig.land()
    rig.plan("b")
    rig.land()
    rig.plan("a")
    assert rig.requests_in_flight() == [(FINE, "a")]
    rig.land()
    assert [upload[1] for upload in rig.uploads] == ["a", "b", "a"]


def test_an_arrival_for_a_position_the_slider_left_is_never_uploaded():
    rig = Rig()
    rig.plan("a")
    rig.plan("b")
    # One fine read at a time: b waits for a.
    assert rig.requests_in_flight() == [(FINE, "a")]
    rig.land()
    assert rig.uploads == []
    assert rig.residency.level_to_draw() is None
    rig.land()
    assert [upload[1] for upload in rig.uploads] == ["b"]


def test_a_read_still_in_flight_is_reused_when_its_position_returns():
    rig = Rig()
    rig.plan("a")
    rig.plan("b")
    rig.plan("a")
    assert rig.requests_in_flight() == [(FINE, "a")]
    rig.land()
    assert [upload[1] for upload in rig.uploads] == ["a"]
    assert rig.tickets == []


def test_the_keep_rule_keeps_a_level_whose_key_is_unchanged():
    """D23: a level the mode omits stays planned while its key is the same."""
    rig = Rig(n_levels=2)
    rig.plan("a")
    rig.land()
    rig.land()
    assert set(rig.residency.held) == {FINE, COARSE}

    rig.plan("a", asked=[COARSE])
    assert rig.tickets == []
    assert rig.releases == []
    assert set(rig.residency.planned) == {FINE, COARSE}
    assert rig.residency.is_drawable(FINE)


def test_the_keep_rule_lets_go_of_a_level_whose_key_changed():
    rig = Rig(n_levels=2)
    rig.plan("a")
    rig.land()
    rig.land()
    rig.plan("b", asked=[COARSE])
    assert set(rig.residency.planned) == {COARSE}
    assert sorted(rig.releases) == [FINE, COARSE]
    assert rig.requests_in_flight() == [(COARSE, "b")]


def test_coarse_first_then_fine_and_never_both():
    rig = Rig(n_levels=2)
    rig.plan("a")
    assert sorted(rig.requests_in_flight()) == [(FINE, "a"), (COARSE, "a")]
    coarse = rig.requests_in_flight().index((COARSE, "a"))
    rig.land(coarse)
    assert rig.residency.level_to_draw() == COARSE
    rig.land()
    assert rig.residency.level_to_draw() == FINE
    assert rig.residency.level_to_draw(prefer_coarse=True) == COARSE


def test_the_preferred_level_falls_back_to_the_drawable_one():
    rig = Rig(n_levels=2)
    rig.plan("a")
    fine = rig.requests_in_flight().index((FINE, "a"))
    rig.land(fine)
    assert rig.residency.level_to_draw(prefer_coarse=True) == FINE


def test_an_invalidation_releases_and_the_reread_is_uploaded_again():
    rig = Rig()
    rig.plan("a")
    rig.land()
    rig.core.invalidate("store")
    assert rig.releases == [FINE]
    assert rig.residency.level_to_draw() is None
    rig.tickets.extend(rig.core.next_reads())
    rig.land()
    assert [upload[1] for upload in rig.uploads] == ["a", "a"]
    assert rig.residency.is_drawable(FINE)


def test_a_failed_read_is_not_drawable_and_not_tried_again():
    rig = Rig()
    rig.plan("a")
    rig.land(error=RuntimeError("boom"))
    assert rig.core.is_complete(rig.residency.cache_id)
    assert rig.residency.level_to_draw() is None
    rig.plan("a")
    assert rig.tickets == []


def test_a_failed_level_leaves_the_other_drawable():
    rig = Rig(n_levels=2)
    rig.plan("a")
    fine = rig.requests_in_flight().index((FINE, "a"))
    rig.land(fine, error=RuntimeError("boom"))
    rig.land()
    assert rig.residency.level_to_draw() == COARSE


def test_an_upload_that_raises_leaves_the_level_empty():
    rig = Rig(fail_upload=True)
    rig.plan("a")
    rig.land()
    assert rig.residency.held == {}
    assert rig.residency.level_to_draw() is None
    # Complete all the same: the read itself succeeded.
    assert rig.core.is_complete(rig.residency.cache_id)


def test_a_retired_cache_keeps_what_it_holds():
    rig = Rig()
    rig.plan("a")
    rig.land()
    rig.core.retire(rig.residency.cache_id)
    rig.core.process()
    assert rig.residency.is_drawable(FINE)
    rig.plan("a")
    assert rig.tickets == []
    assert len(rig.uploads) == 1


def test_keys_are_forgotten_with_what_they_named():
    """The interner holds only keys the registry or the plan still names."""
    rig = Rig()
    for position in range(50):
        rig.plan(position)
        rig.land()
    assert len(rig.residency._items) <= rig.residency.n_slots + 1
    assert len(rig.residency._requests) <= rig.residency.n_slots + 1


def test_changes_are_announced():
    rig = Rig()
    before = rig.changes
    rig.plan("a")
    assert rig.changes > before
    before = rig.changes
    rig.land()
    assert rig.changes > before


def test_n_levels_must_be_positive():
    with pytest.raises(ValueError, match="n_levels"):
        LevelResidency(0, upload=lambda *a: None, release=lambda level: None)


# ---------------------------------------------------------------------------
# keeps_previous: a level stays on screen across a change that leaves its
# result in the right place (clipping planes design 5.2, D31)
# ---------------------------------------------------------------------------


def _only_the_clip_changed(held, planned) -> bool:
    """Request keys here are ``(view, clip)``."""
    return held[0] == planned[0] and held[1] != planned[1]


def test_a_kept_level_stays_drawable_until_the_new_result_lands():
    rig = Rig(keeps_previous=_only_the_clip_changed)
    rig.plan(("view", "clip 1"))
    rig.land()
    rig.plan(("view", "clip 2"))
    assert rig.releases == []
    assert rig.residency.is_drawable(FINE)
    assert rig.residency.level_to_draw() == FINE
    assert not rig.residency.awaiting
    rig.land()
    assert [upload[1] for upload in rig.uploads] == [
        ("view", "clip 1"),
        ("view", "clip 2"),
    ]
    assert rig.residency.level_to_draw() == FINE
    assert rig.releases == []


def test_a_change_of_view_still_releases_the_level():
    rig = Rig(keeps_previous=_only_the_clip_changed)
    rig.plan(("view", "clip 1"))
    rig.land()
    rig.plan(("other view", "clip 1"))
    assert rig.releases == [FINE]
    assert rig.residency.level_to_draw() is None
    assert rig.residency.awaiting


def test_a_kept_level_is_released_when_the_view_then_changes():
    rig = Rig(keeps_previous=_only_the_clip_changed)
    rig.plan(("view", "clip 1"))
    rig.land()
    rig.plan(("view", "clip 2"))
    rig.plan(("other view", "clip 2"))
    assert rig.releases == [FINE]
    assert rig.residency.level_to_draw() is None


def test_a_superseded_arrival_of_the_same_view_is_shown():
    """A drag faster than the reads still updates the picture."""
    rig = Rig(keeps_previous=_only_the_clip_changed)
    rig.plan(("view", "clip 1"))
    rig.land()
    rig.plan(("view", "clip 2"))
    rig.plan(("view", "clip 3"))
    # One fine read at a time: clip 3 waits for clip 2, which is superseded.
    assert rig.requests_in_flight() == [(FINE, ("view", "clip 2"))]
    rig.land()
    assert [upload[1][1] for upload in rig.uploads] == ["clip 1", "clip 2"]
    assert rig.residency.level_to_draw() == FINE
    rig.land()
    assert [upload[1][1] for upload in rig.uploads] == ["clip 1", "clip 2", "clip 3"]
    assert rig.residency.is_drawable(FINE)
    assert rig.releases == []


def test_a_superseded_arrival_of_another_view_is_not_shown():
    rig = Rig(keeps_previous=_only_the_clip_changed)
    rig.plan(("view", "clip 1"))
    rig.plan(("other view", "clip 1"))
    rig.land()
    assert rig.uploads == []
    assert rig.residency.level_to_draw() is None
