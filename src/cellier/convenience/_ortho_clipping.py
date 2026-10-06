"""Link clipping planes across the OrthoViewer panels of one dataset.

See ``plans/clipping_planes_linking_design.md`` (sections 4 and 5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

from psygnal import Signal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cellier.controller import CellierController
    from cellier.events import (
        ClippingPlanesChangedEvent,
        PlaneInteractionEvent,
        VisualRemovedEvent,
    )

ClippingLinkMode = Literal["all", "2d"] | None
"""Which panels of one ``add_*`` call carry one tuple of clipping planes."""

_PANEL_KEYS = ("xy", "xz", "yz", "vol")
_2D_KEYS = ("xy", "xz", "yz")
_LINKED_KEYS: dict[ClippingLinkMode, tuple[tuple[str, ...], ...]] = {
    "all": (_PANEL_KEYS,),
    "2d": (_2D_KEYS,),
    None: (),
}


def _check_mode(mode: object) -> None:
    if mode not in _LINKED_KEYS:
        raise ValueError(
            f'Unknown clipping link mode {mode!r}. Expected "all", "2d" or None.'
        )


class OrthoClippingController:
    """Keep the clipping planes of linked panel visuals equal.

    The panel visuals of one ``OrthoViewer.add_*`` call read one store and
    share its data coordinate system, so one tuple of planes means the same
    cut in each of them.  ``mode`` says which of them carry one tuple:

    ========== ======================================================
    ``"all"``  xy, xz, yz and vol.
    ``"2d"``   xy, xz and yz.  vol is independent.
    ``None``   No link.  Each panel's visual has its own planes.
    ========== ======================================================

    The mode applies to every dataset.  Links are always within the panels
    of one ``add_*`` call.

    This holds **no plane state**.  Each visual's ``clipping_planes`` stays
    the source of truth and is what is serialized; this copies between
    them, through the controller's public API.  A change of one visual is
    written to the visuals linked to it whose tuple differs, stamped with
    the ``source_id`` of the change that caused it, so each linked visual
    emits one ``ClippingPlanesChangedEvent`` per change.  A tuple the
    controller refuses changes no visual.

    A **drag** is forwarded too.  When a clipping plane drag starts on one
    visual, this opens a clipping interaction scope on every visual linked
    to it and closes them when that drag ends, so each linked visual
    reports one start and one end per drag.  The linked visuals end with
    ``"release"`` whatever ended the origin's drag: when the origin ends
    with ``"settle"`` (the handle was held still), the next change starts a
    new drag on all of them.  With no event loop there is no stillness
    timer and every change is a drag of its own, on every linked visual.

    Event order: the bus calls subscribers in registration order, and this
    subscribes when a dataset is added.  A subscriber registered later
    hears the events of the linked visuals before the event of the visual
    that was edited.  Every one of them carries the same tuple.

    Parameters
    ----------
    controller : CellierController
        The controller owning the visuals.
    mode : {"all", "2d"} or None
        The starting mode.

    Attributes
    ----------
    mode_changed : psygnal.Signal
        Emitted with the new mode after :attr:`mode` changed.
    groups_changed : psygnal.Signal
        Emitted after :attr:`groups` changed: a dataset was added, or a
        visual was removed.
    """

    mode_changed = Signal(object)
    groups_changed = Signal()

    def __init__(
        self, controller: CellierController, mode: ClippingLinkMode = "all"
    ) -> None:
        _check_mode(mode)
        self._id: UUID = uuid4()
        self._controller = controller
        self._mode: ClippingLinkMode = mode
        #: One dict per ``add_*`` call: panel key -> visual id.
        self._groups: list[dict[str, UUID]] = []
        # One id per visual: the owner of the visual's subscriptions, so
        # they can be dropped when it is removed, and the source of the
        # scopes forwarded from it, so two visuals dragged at once hold two
        # scopes on the others.  And the visuals each origin currently
        # holds a scope on.
        self._scope_ids: dict[UUID, UUID] = {}
        self._forwarded: dict[UUID, list[UUID]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def id(self) -> UUID:
        """The source id stamped on the writes of a mode change."""
        return self._id

    @property
    def mode(self) -> ClippingLinkMode:
        """Which panels of one dataset carry one tuple of planes.

        Setting it to a less linked mode writes nothing: the visuals keep
        the planes they have and are free to differ from then on.  Setting
        it to a more linked mode makes the newly linked visuals agree: the
        3D view's planes are given to the others (it has the gizmo), or
        the xy panel's when only 2D panels are linked.  Those writes are
        stamped with :attr:`id`.  A drag being forwarded is closed on the
        linked visuals first; the newly linked visuals join at the next
        drag.  ``mode_changed`` is emitted once, after the writes.  Setting
        the current mode does nothing.
        """
        return self._mode

    @mode.setter
    def mode(self, mode: ClippingLinkMode) -> None:
        _check_mode(mode)
        if mode == self._mode:
            return
        for origin in list(self._forwarded):
            self._end_forwarded(origin)
        self._mode = mode
        for group in list(self._groups):
            self._reconcile(group)
        self.mode_changed.emit(mode)

    @property
    def groups(self) -> list[dict[str, UUID]]:
        """Panel key -> visual id of each dataset, in the order added.

        A removed visual is absent from its group; a dataset with no
        visual left is absent.
        """
        return [dict(group) for group in self._groups]

    def link_groups(self, group: Mapping[str, UUID]) -> list[list[UUID]]:
        """The visuals of *group* split by what carries one tuple now.

        Parameters
        ----------
        group : Mapping[str, UUID]
            Panel key -> visual id: one of :attr:`groups`.

        Returns
        -------
        list[list[UUID]]
            First the linked visuals together, then each unlinked visual
            alone, in panel order.  Every visual of *group* is in exactly
            one list: ``[[xy, xz, yz, vol]]`` in ``"all"``, ``[[xy, xz,
            yz], [vol]]`` in ``"2d"`` and four lists of one in ``None``.
        """
        linked = _LINKED_KEYS[self._mode]
        out = [[group[key] for key in keys if key in group] for keys in linked]
        together = {key for keys in linked for key in keys}
        out.extend(
            [group[key]] for key in _PANEL_KEYS if key in group and key not in together
        )
        return [ids for ids in out if ids]

    def add_group(
        self, visuals_by_panel: Mapping[str, object], *, require_agreement: bool = False
    ) -> None:
        """Link the panel visuals of one dataset.

        Parameters
        ----------
        visuals_by_panel : Mapping[str, object]
            Panel key (``"xy"``, ``"xz"``, ``"yz"``, ``"vol"``) -> the
            panel's visual model or its id.  Panels may be missing.
        require_agreement : bool
            What to do when visuals this mode links carry different
            planes.  ``False`` (default) makes them agree, as a change to
            a more linked mode does.  ``True`` raises and links nothing.

        Raises
        ------
        ValueError
            With *require_agreement*, if visuals this mode links carry
            different planes.
        """
        group = {
            key: getattr(visuals_by_panel[key], "id", visuals_by_panel[key])
            for key in _PANEL_KEYS
            if key in visuals_by_panel
        }
        controller = self._controller
        if require_agreement:
            for ids in self.link_groups(group):
                first = controller.get_visual_model(ids[0]).clipping_planes
                if any(
                    controller.get_visual_model(other).clipping_planes != first
                    for other in ids[1:]
                ):
                    raise ValueError(
                        "The panel visuals do not carry the same clipping "
                        f"planes, so they cannot be linked in mode {self._mode!r}."
                    )
        self._groups.append(group)
        for visual_id in group.values():
            owner_id = self._scope_ids[visual_id] = uuid4()
            controller.on_clipping_planes_changed(
                visual_id, self._on_planes, owner_id=owner_id
            )
            controller.on_plane_interaction(
                visual_id, self._on_interaction, owner_id=owner_id
            )
            controller.on_visual_removed(visual_id, self._on_removed, owner_id=owner_id)
        self._reconcile(group)
        self.groups_changed.emit()

    def close(self) -> None:
        """Stop linking: close forwarded scopes and drop the subscriptions."""
        for origin in list(self._forwarded):
            self._end_forwarded(origin)
        for owner_id in self._scope_ids.values():
            self._controller.unsubscribe_owner(owner_id)
        self._groups.clear()
        self._scope_ids.clear()

    # ------------------------------------------------------------------
    # Mirroring
    # ------------------------------------------------------------------

    def _linked_to(self, visual_id: UUID) -> list[UUID]:
        """The other visuals that carry *visual_id*'s tuple in this mode."""
        for group in self._groups:
            for ids in self.link_groups(group):
                if visual_id in ids:
                    return [other for other in ids if other != visual_id]
        return []

    def _copy(self, planes, targets: list[UUID], source_id: UUID) -> None:
        """Write *planes* to the *targets* that differ.

        There is no "mirroring in progress" flag.  The event of a visual
        written here comes back to :meth:`_on_planes` and finds every tuple
        of its link group equal, or written by the time this loop reaches
        it; that check is the only echo guard.  A flag would also drop the
        events of another dataset that user code changes from a handler.
        """
        controller = self._controller
        for other in targets:
            if controller.get_visual_model(other).clipping_planes != planes:
                controller.set_clipping_planes(other, planes, source_id=source_id)

    def _reconcile(self, group: Mapping[str, UUID]) -> None:
        """Make every link group agree: the 3D view wins, else the first."""
        controller = self._controller
        vol = group.get("vol")
        for ids in self.link_groups(group):
            if len(ids) < 2:
                continue
            winner = vol if vol in ids else ids[0]
            planes = controller.get_visual_model(winner).clipping_planes
            self._copy(planes, [i for i in ids if i != winner], self._id)

    def _on_planes(self, event: ClippingPlanesChangedEvent) -> None:
        current = self._controller.get_visual_model(event.visual_id).clipping_planes
        if current != event.clipping_planes:
            # A subscriber called before this one changed the visual again;
            # that change's own event carries the tuple to copy.
            return
        targets = self._linked_to(event.visual_id)
        if targets:
            self._copy(event.clipping_planes, targets, event.source_id)

    # ------------------------------------------------------------------
    # Drag forwarding
    # ------------------------------------------------------------------

    def _on_interaction(self, event: PlaneInteractionEvent) -> None:
        """Hold a scope on the linked visuals for as long as one is dragged."""
        origin = event.visual_id
        if event.phase == "end":
            # Whatever ended it, and whoever: only an origin has an entry.
            self._end_forwarded(origin)
            return
        if origin in self._forwarded:
            return
        # Echo guard: a start on a visual this holds a forwarded scope on
        # was caused by a mirrored change.  Per forward, not one flag, so
        # a drag user code starts on another dataset is still forwarded.
        if any(origin in targets for targets in self._forwarded.values()):
            return
        others = self._linked_to(origin)
        if not others:
            return
        self._forwarded[origin] = others
        for other in others:
            self._controller.begin_plane_interaction(
                other, source_id=self._scope_ids[origin]
            )

    def _end_forwarded(self, origin: UUID) -> None:
        source_id = self._scope_ids.get(origin)
        for other in self._forwarded.pop(origin, ()):
            self._controller.end_plane_interaction(other, source_id=source_id)

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def _on_removed(self, event: VisualRemovedEvent) -> None:
        """Forget a removed visual; the rest of its group stays linked."""
        removed = event.visual_id
        if removed not in self._scope_ids:
            return
        # Scopes forwarded from it are closed; one forwarded to it went
        # with the visual.
        self._end_forwarded(removed)
        for others in self._forwarded.values():
            if removed in others:
                others.remove(removed)
        self._controller.unsubscribe_owner(self._scope_ids.pop(removed))
        for group in self._groups:
            for key, visual_id in list(group.items()):
                if visual_id == removed:
                    del group[key]
        self._groups = [group for group in self._groups if group]
        self.groups_changed.emit()
