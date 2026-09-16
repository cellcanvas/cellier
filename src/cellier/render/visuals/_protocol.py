"""Protocol shared by all GFX render-layer visuals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from uuid import UUID

    import pygfx as gfx

    from cellier.transform import AffineTransform
    from cellier.visuals._base_visual import BaseVisual


class GFXVisual(Protocol):
    """Structural protocol for all GFX render-layer visuals.

    All visual types expose this interface so the controller can drive
    node construction and geometry rebuilds without knowing the concrete
    type or keeping a reference to the model.
    """

    visual_model_id: UUID

    def has_node(self, mode: str) -> bool:
        """Return True if the node for *mode* has already been built."""
        ...

    def build_node(
        self,
        mode: str,
        visual_model: BaseVisual,
        displayed_axes: tuple[int, ...],
        level_shapes: list[tuple[int, ...]],
        level_transforms: list[AffineTransform],
    ) -> gfx.WorldObject | None:
        """Build the node for *mode* for the first time.

        Reads appearance and config from *visual_model* at call time.
        Must NOT store *visual_model* beyond this call.
        """
        ...

    def rebuild_node_geometry(
        self,
        mode: str,
        displayed_axes: tuple[int, ...],
        level_shapes: list[tuple[int, ...]],
        level_transforms: list[AffineTransform],
    ) -> gfx.WorldObject | None:
        """Rebuild geometry on an already-built node after a dims change."""
        ...

    def get_node(self, mode: str) -> gfx.WorldObject | None:
        """Return the already-built node for *mode*, or None if not built."""
        ...
