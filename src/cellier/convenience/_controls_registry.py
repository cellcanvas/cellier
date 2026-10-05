"""Which visuals have controls configured, kept current as visuals come and go.

Both convenience viewers record the ``controls=`` passed to their ``add_*``
methods here, and the layout docks read it back.  A dock is built once but has
to follow the viewer (``plans/multi_visual_controls.md``), so the registry says
when it changed rather than leaving each dock to poll.

Why the viewer and not the controller's bus: ``VisualAddedEvent`` fires inside
the controller *before* ``add_*`` returns and records the config, so a dock
listening for it would see the visual without its controls.  Removal is the
other way round -- only the controller knows a visual went away -- so the
registry listens for ``VisualRemovedEvent`` and prunes itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from psygnal import Signal

from cellier.events import VisualRemovedEvent

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from cellier.controller import CellierController
    from cellier.convenience.gui._controls_config import BaseControlsConfig


class ControlsRegistryMixin:
    """The ``controls=`` record shared by ``Viewer`` and ``OrthoViewer``.

    ``_controls_configs`` maps a representative visual id to its config, in
    registration order.  ``_visual_groups`` maps that id to every visual the
    controls drive: the id alone on a ``Viewer``; on an ``OrthoViewer``, the
    three 2D panel siblings in one group and the 3D panel's visual in another.
    ``_controls_labels`` holds an explicit dock label for a group that was
    given one; a group without one is named from its visuals.
    ``_controls_changed`` is emitted after every change.

    A flag of a config says which visuals get controls; a dock node of the
    layout says where they go.  ``_rendered_dock_nodes`` holds the names of
    the dock nodes of the layout the viewer was last rendered with (``None``
    until it is rendered; :meth:`_record_rendered_layout`), and a flag with
    no node for it is an error: when
    the layout is rendered (:meth:`_check_rendered_layout`) and when a
    visual is added afterwards (:meth:`_check_controls_docks`).  A node with
    no flag is not an error: the dock shows its placeholder.
    """

    _controls_changed = Signal()

    #: The dock node that shows clipping planes controls: the appearance
    #: dock on a ``Viewer``; an ``OrthoViewer`` overrides it.
    _CLIPPING_DOCK_NODE = "AppearanceControls"
    #: Dock node names of the last rendered layout; ``None`` if not rendered.
    _rendered_dock_nodes: frozenset[str] | None = None

    _controller: CellierController
    _controls_configs: dict[UUID, BaseControlsConfig]
    _visual_groups: dict[UUID, list[UUID]]
    _controls_labels: dict[UUID, str]

    def _init_controls_registry(self) -> None:
        """Create the empty record and start pruning removed visuals.

        Needs ``self._controller``.  The subscription is weak so a dropped
        viewer is not kept alive by a controller that outlives it.
        """
        self._controls_configs = {}
        self._visual_groups = {}
        self._controls_labels = {}
        self._controller._outgoing_events.subscribe(
            VisualRemovedEvent, self._forget_removed_visual, weak=True
        )

    def _store_controls(
        self,
        visual_ids: Sequence[UUID],
        controls: BaseControlsConfig | None,
        label: str | None = None,
    ) -> None:
        """Record *controls* for the visual group *visual_ids*.

        The first id is the representative: it keys the config, and its
        visual is the one the controls are seeded from.  *label*, when given,
        is what the dock's selector calls the group.
        """
        if controls is None or not visual_ids:
            return
        rep_id = visual_ids[0]
        self._controls_configs[rep_id] = controls
        self._visual_groups[rep_id] = list(visual_ids)
        if label is not None:
            self._controls_labels[rep_id] = label
        self._controls_changed.emit()

    def _check_controls_docks(
        self,
        controls: BaseControlsConfig | None,
        name: str,
        nodes: frozenset[str] | None = None,
    ) -> None:
        """Refuse a controls flag that the rendered layout has no dock node for.

        Called by every ``add_*`` before anything is added.  A viewer that
        has not been rendered with a layout is not checked: an offscreen
        capture or a hand-built Qt layout records flags and builds no dock.

        Parameters
        ----------
        controls : BaseControlsConfig or None
            The config passed as ``controls=``.
        name : str
            What the visual is called, for the message.
        nodes : frozenset[str] or None
            Dock node names to check against.  Defaults to those of the
            layout last rendered.

        Raises
        ------
        ValueError
            Naming the visual, the flag and the dock node to add.
        """
        nodes = self._rendered_dock_nodes if nodes is None else nodes
        if controls is None or nodes is None:
            return
        from cellier.convenience.layout._shared import missing_dock_node

        problem = missing_dock_node(
            controls, nodes, clipping_node=self._CLIPPING_DOCK_NODE
        )
        if problem is not None:
            raise ValueError(f"{name!r} was added with {problem}")

    def _check_rendered_layout(self, nodes: frozenset[str]) -> None:
        """Check every recorded config against a layout about to be rendered.

        Called by the layout walk before it builds anything, with the names
        of the layout's dock nodes.  It records nothing: the walk calls
        :meth:`_record_rendered_layout` once the layout is built.

        Raises
        ------
        ValueError
            For the first visual with a flag the layout has no dock node for.
        """
        from cellier.convenience.layout._shared import _group_name

        for rep_id, config in self._controls_configs.items():
            try:
                visual = self._controller.get_visual_model(rep_id)
            except KeyError:
                continue
            name = _group_name(
                self._controller, visual, self._visual_groups.get(rep_id, [rep_id])
            )
            self._check_controls_docks(config, name, nodes)

    def _record_rendered_layout(self, nodes: frozenset[str]) -> None:
        """Keep the dock node names of a layout that was just rendered.

        Called by the layout walk after the whole layout is built, so a
        render that raises leaves the viewer as it was.  The names are what
        the ``add_*`` calls that follow are checked against.
        """
        self._rendered_dock_nodes = frozenset(nodes)

    def _forget_removed_visual(self, event: VisualRemovedEvent) -> None:
        """Drop a removed visual from the record, emitting if anything changed.

        Removing one sibling of a group keeps the group.  Removing the
        representative hands the config to the next sibling *in place*, so the
        entry keeps its position in registration order.
        """
        removed = event.visual_id
        rep_id = next(
            (rep for rep, ids in self._visual_groups.items() if removed in ids),
            None,
        )
        if rep_id is None:
            return

        remaining = [i for i in self._visual_groups[rep_id] if i != removed]
        if not remaining:
            self._controls_configs.pop(rep_id, None)
            self._visual_groups.pop(rep_id)
            self._controls_labels.pop(rep_id, None)
        elif rep_id == removed:
            new_rep = remaining[0]
            self._controls_configs = {
                (new_rep if key == rep_id else key): config
                for key, config in self._controls_configs.items()
            }
            self._visual_groups = {
                (new_rep if key == rep_id else key): (
                    remaining if key == rep_id else ids
                )
                for key, ids in self._visual_groups.items()
            }
            if rep_id in self._controls_labels:
                self._controls_labels[new_rep] = self._controls_labels.pop(rep_id)
        else:
            self._visual_groups[rep_id] = remaining
        self._controls_changed.emit()
