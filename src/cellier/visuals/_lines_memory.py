from __future__ import annotations

from typing import Literal

from pydantic import Field

from cellier.visuals._base_visual import BaseAppearance, BaseVisual


class LinesMemoryAppearance(BaseAppearance):
    """Appearance model for a lines visual.

    Parameters
    ----------
    color : tuple[float, float, float, float]
        RGBA colour of every vertex when ``color_mode`` is ``"uniform"``.
        Not a fallback: with ``color_mode="vertex"`` it is unused, and a
        store with no per-vertex colours raises ``ValueError`` when the
        lines are drawn.
    thickness : float
        Line thickness, interpreted in ``thickness_space`` units.
    thickness_space : str
        Coordinate space for thickness.  ``"screen"`` (logical pixels)
        or ``"world"`` (world-space units).  Default ``"screen"``.
    color_mode : str
        ``"uniform"`` — use ``color`` for every vertex.
        ``"vertex"`` — use per-vertex colors from the store; the store
        must carry them.
    opacity : float
        Master opacity multiplier in [0, 1].
    """

    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    thickness: float = 2.0
    thickness_space: Literal["screen", "world"] = "screen"
    color_mode: Literal["uniform", "vertex"] = "uniform"


class LinesVisual(BaseVisual):
    """Model-layer visual for a line-segment collection backed by LinesMemoryStore.

    Parameters
    ----------
    appearance : LinesMemoryAppearance
        Initial appearance.
    requires_camera_reslice : bool
        Always False — lines do not depend on camera position. Frozen.
    """

    visual_type: Literal["lines_memory"] = "lines_memory"
    appearance: LinesMemoryAppearance = Field(default_factory=LinesMemoryAppearance)
    requires_camera_reslice: bool = Field(default=False, frozen=True)
