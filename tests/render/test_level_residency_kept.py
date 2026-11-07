"""``LevelResidency`` for whole-item caches under a change stream.

``plans/scheduler_only_loading_design_v2.md`` 4.2: the ``policy`` argument,
``keeps_through_invalidation`` (D12), ``write_kept`` and the step that draws
a kept stale result (D14).  The mesh's behaviour (all of it off) is
``test_level_residency.py``'s, unchanged.
"""

from __future__ import annotations

import dataclasses

from cellier.render import SchedulerConfig
from cellier.render._level_residency import (
    LEVEL_CACHE_POLICY,
    MEMORY_GEOMETRY_POLICY,
    MEMORY_RASTER_POLICY,
    LevelResidency,
)
from cellier.render.scheduling import ReadOutcome, SchedulerCore

FINE = 0


class Store:
    id = "store"


class Rig:
    """A single-level whole-item cache on a core, reads completed by hand."""

    def __init__(self, *, keeps: bool = True, policy=MEMORY_RASTER_POLICY) -> None:
        self.uploads: list[tuple[object, object]] = []
        self.releases: list[int] = []
        self.residency = LevelResidency(
            1,
            upload=lambda level, key, data: self.uploads.append((key, data)),
            release=self.releases.append,
            keeps_previous=lambda held, planned: True,
            policy=policy,
            keeps_through_invalidation=keeps,
        )
        self.core = SchedulerCore(SchedulerConfig())
        self.core.register(self.residency.cache_id, self.residency, scene="scene")
        self.tickets: list = []

    def plan(self, position) -> None:
        desired = self.residency.desired({FINE: position}, [FINE], {FINE: position})
        self.core.set_desired(dataclasses.replace(desired, store=Store()))
        self.core.process()
        self.tickets.extend(self.core.next_reads())

    def land(self, data, index: int = 0, commit: bool = True) -> str:
        ticket = self.tickets.pop(index)
        outcome = self.core.complete_read(ticket, data=data)
        if commit:
            self.commit()
        return outcome

    def commit(self) -> None:
        self.core.commit_round("scene")
        self.tickets.extend(self.core.next_reads())

    def rebuild(self) -> None:
        """Rebuild the draw outside a round (as a pass or invalidation does)."""
        self.core._settle(self.core._caches[self.residency.cache_id])

    def invalidate(self) -> None:
        self.core.invalidate("store")
        self.tickets.extend(self.core.next_reads())

    @property
    def drawn(self) -> bool:
        return self.residency.is_drawable(FINE)


def test_the_policy_is_an_argument() -> None:
    assert LevelResidency(1, lambda *a: None, lambda level: None).policy is (
        LEVEL_CACHE_POLICY
    )
    rig = Rig()
    assert rig.residency.policy is MEMORY_RASTER_POLICY
    assert rig.core.policy_of(rig.residency.cache_id) is MEMORY_RASTER_POLICY


def test_the_memory_policies() -> None:
    """D17: cap 1 for raster and 2 for geometry; compute, one attempt."""
    for policy, cap in ((MEMORY_RASTER_POLICY, 1), (MEMORY_GEOMETRY_POLICY, 2)):
        assert policy.max_target_fetching == cap
        assert policy.resource == "compute"
        assert policy.retry_max_attempts == 1
        assert policy.retry_on_pass is False
        assert policy.commit_stale is True
    assert LEVEL_CACHE_POLICY.commit_stale is False


def test_an_invalidation_keeps_the_held_result_drawn() -> None:
    """D12: the planned key is invalidated; its upload stays until the re-read."""
    rig = Rig()
    rig.plan("a")
    rig.land("a1")
    rig.invalidate()
    assert rig.drawn
    assert rig.releases == []
    assert not rig.residency.awaiting
    rig.land("a2")
    assert rig.uploads == [("a", "a1"), ("a", "a2")]
    assert rig.drawn


def test_without_the_flag_an_invalidation_releases() -> None:
    rig = Rig(keeps=False, policy=LEVEL_CACHE_POLICY)
    rig.plan("a")
    rig.land("a1")
    rig.invalidate()
    assert not rig.drawn
    assert rig.releases == [FINE]


def test_a_stale_read_of_the_planned_key_is_drawn_as_kept() -> None:
    """D14: the read in flight lands stale and is drawn at the next round."""
    rig = Rig()
    rig.plan("a")
    rig.land("a1")
    rig.invalidate()  # requeued and issued again at once
    rig.invalidate()  # the re-read in flight is now stale
    assert rig.land("a2", commit=False) == ReadOutcome.STALE_KEPT
    assert rig.uploads == [("a", "a1")]
    rig.commit()
    assert rig.uploads == [("a", "a1"), ("a", "a2")]
    assert rig.drawn
    rig.land("a3")
    assert rig.uploads[-1] == ("a", "a3")
    assert rig.drawn


def test_write_kept_of_a_superseded_key_is_dropped() -> None:
    rig = Rig()
    rig.plan("a")
    rig.land("a1")
    key_a = rig.residency.key_of(FINE, "a")
    rig.plan("b")
    rig.residency.write_kept(key_a, "a-stale")
    rig.rebuild()
    assert ("a", "a-stale") not in rig.uploads


def test_write_kept_of_an_unknown_key_is_ignored() -> None:
    rig = Rig()
    rig.plan("a")
    rig.residency.write_kept(12345, "junk")
    rig.rebuild()
    assert rig.uploads == []


def test_a_newer_kept_payload_replaces_an_older_one() -> None:
    rig = Rig()
    rig.plan("a")
    key = rig.residency.key_of(FINE, "a")
    rig.residency.write_kept(key, "older")
    rig.residency.write_kept(key, "newer")
    rig.rebuild()
    assert rig.uploads == [("a", "newer")]
    assert rig.drawn


def test_a_kept_result_does_not_replace_a_newer_resident_one() -> None:
    """A fresh arrival committed in the same round wins over the stale one."""
    rig = Rig()
    rig.plan("a")
    rig.land("a1")
    rig.invalidate()
    ticket = rig.tickets.pop(0)
    key = ticket.key
    rig.core.complete_read(ticket, data="fresh")
    # A stale copy handed over in the same round (written after the arrival).
    rig.residency.write_kept(key, "stale")
    rig.commit()
    assert rig.uploads[-1] == ("a", "fresh")
    assert rig.drawn


def test_a_retired_cache_shown_again_reads_nothing_and_stays_drawn() -> None:
    """D18: hide (retire) then show at an unchanged key."""
    rig = Rig()
    rig.plan("a")
    rig.land("a1")
    rig.core.retire(rig.residency.cache_id)
    rig.core.process()
    assert rig.drawn
    rig.plan("a")
    assert rig.tickets == []
    assert rig.drawn
    assert rig.uploads == [("a", "a1")]
