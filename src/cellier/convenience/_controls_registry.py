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
    controls drive: the id alone on a ``Viewer``, the four panel siblings on an
    ``OrthoViewer``.  ``_controls_changed`` is emitted after every change.
    """

    _controls_changed = Signal()

    _controller: CellierController
    _controls_configs: dict[UUID, BaseControlsConfig]
    _visual_groups: dict[UUID, list[UUID]]

    def _init_controls_registry(self) -> None:
        """Create the empty record and start pruning removed visuals.

        Needs ``self._controller``.  The subscription is weak so a dropped
        viewer is not kept alive by a controller that outlives it.
        """
        self._controls_configs = {}
        self._visual_groups = {}
        self._controller._outgoing_events.subscribe(
            VisualRemovedEvent, self._forget_removed_visual, weak=True
        )

    def _store_controls(
        self, visual_ids: Sequence[UUID], controls: BaseControlsConfig | None
    ) -> None:
        """Record *controls* for the visual group *visual_ids*.

        The first id is the representative: it keys the config, and its
        visual is the one the controls are seeded from.
        """
        if controls is None or not visual_ids:
            return
        rep_id = visual_ids[0]
        self._controls_configs[rep_id] = controls
        self._visual_groups[rep_id] = list(visual_ids)
        self._controls_changed.emit()

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
        else:
            self._visual_groups[rep_id] = remaining
        self._controls_changed.emit()
