from __future__ import annotations

from typing import Literal

from pydantic import Field

from cellier.visuals._base_visual import BaseAppearance, BaseVisual


class PointsMarkerAppearance(BaseAppearance):
    """Appearance model for a points visual.

    Parameters
    ----------
    color : tuple[float, float, float, float]
        RGBA colour of every point when ``color_mode`` is ``"uniform"``.
        Not a fallback: with ``color_mode="vertex"`` it is unused, and a
        store with no per-point colours raises ``ValueError`` when the
        points are drawn.
    size : float
        Uniform point size, interpreted in ``size_space`` units.
    size_space : str
        Coordinate space for size.  ``"screen"`` (pixels) or
        ``"world"`` (world-space units).  Default ``"screen"``.
    color_mode : str
        Declares **where RGB comes from**.  ``"uniform"`` — use ``color``
        for every point.  ``"vertex"`` — use per-point colors from the
        store.  A caller declaration, never inferred from the data and
        never overwritten at commit time.
    size_mode : str
        Declares **where the size comes from**, on the same terms as
        ``color_mode``.  ``"uniform"`` — use ``size`` for every point.
        ``"vertex"`` — use per-point sizes from the store.  Declaring
        ``"vertex"`` with no sizes in the store raises at commit rather
        than falling back.
    opacity : float
        Master opacity multiplier in [0, 1].
    """

    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    size: float = 5.0
    size_space: Literal["screen", "world"] = "screen"
    color_mode: Literal["uniform", "vertex"] = "uniform"
    size_mode: Literal["uniform", "vertex"] = "uniform"


class PointsVisual(BaseVisual):
    """Model-layer visual for a point cloud backed by PointsMemoryStore.

    Parameters
    ----------
    appearance : PointsMarkerAppearance
        Initial appearance.
    requires_camera_reslice : bool
        Always False — points do not depend on camera position. Frozen.
    """

    visual_type: Literal["points_memory"] = "points_memory"
    appearance: PointsMarkerAppearance = Field(default_factory=PointsMarkerAppearance)
    requires_camera_reslice: bool = Field(default=False, frozen=True)
