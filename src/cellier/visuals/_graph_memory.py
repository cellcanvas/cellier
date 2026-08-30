# src/cellier/visuals/_graph_memory.py
from __future__ import annotations

from typing import Literal

from psygnal import EventedModel
from pydantic import Field

from cellier.visuals._base_visual import BaseAppearance, BaseVisual


class TrailConfig(EventedModel):
    """Asymmetric window on one sliced axis.

    Extends the slab on a single axis to ``[index - before, index + after]``,
    optionally fading elements by their distance from the current index.
    Nothing here presumes the axis is time: it works identically for a
    z-depth window or a channel sweep.

    The axis is the **dict key** on the visual, not a field here, so a
    config can never disagree with where it is filed.

    Parameters
    ----------
    before : float
        Extent of the window below the current slice index, in data units.
        Default 0.5, which is the symmetric slab points and lines use.
    after : float
        Extent above the current slice index.  Default 0.5.
    fade : bool
        If True, elements fade with distance from the current index.  The
        fade rides its own per-element alpha buffer, so it composes with
        either ``color_mode`` and never overwrites it (D19/D20).
    fade_before : float | None
        Falloff distance below the index.  ``None`` uses ``before`` -- the
        window width is the only sensible falloff when none is given.
    fade_after : float | None
        Falloff distance above the index.  ``None`` uses ``after``.
    min_alpha : float
        Floor applied to in-window elements, so a trail need not vanish
        before the window ends.  Out-of-window elements are forced to
        exactly 0.0 regardless, so a dangling edge (D13) fades to nothing
        at the boundary rather than being cut square.
    """

    before: float = 0.5
    after: float = 0.5
    fade: bool = False
    fade_before: float | None = None
    fade_after: float | None = None
    min_alpha: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def resolved_fade_before(self) -> float:
        """``fade_before``, falling back to ``before``."""
        return self.before if self.fade_before is None else self.fade_before

    @property
    def resolved_fade_after(self) -> float:
        """``fade_after``, falling back to ``after``."""
        return self.after if self.fade_after is None else self.fade_after


class GraphAppearance(BaseAppearance):
    """Appearance model for a graph visual.

    **Flat, with ``node_`` / ``edge_`` prefixes, on purpose.**  psygnal does
    not propagate nested ``EventedModel`` field changes to the parent's
    event group, and ``Controller._wire_appearance`` subscribes only to
    ``visual.appearance.events`` -- so composing this from
    ``PointsMarkerAppearance`` and ``LinesMemoryAppearance`` would silently
    drop every node and edge appearance change (D9).  The cost is a dozen
    duplicated field declarations; the benefit is that the appearance event
    bridge, the GUI control generation and serialization all work unchanged.

    ``BaseAppearance`` contributes ``visible``, ``opacity``,
    ``render_order`` and the depth/transparency fields.  ``visible`` and
    ``render_order`` apply to the group; the rest are pushed to both child
    materials, with ``node_depth_compare`` / ``edge_depth_compare``
    overriding ``depth_compare`` per child.

    Parameters
    ----------
    node_color : tuple[float, float, float, float]
        RGBA uniform node colour, used when ``node_color_mode`` is
        ``"uniform"``.
    node_size : float
        Uniform node size, in ``node_size_space`` units.
    node_size_space : str
        ``"screen"`` (pixels) or ``"world"``.
    node_color_mode : str
        Declares **where node RGB comes from** (D20): ``"uniform"`` is this
        model's ``node_color``, ``"vertex"`` is the store's per-node
        colours.  A caller declaration, never inferred from the data and
        never overwritten at commit time.
    node_size_mode : str
        Declares **where the node size comes from**, on the same terms as
        ``node_color_mode``.  ``"uniform"`` is this model's ``node_size``,
        ``"vertex"`` is the store's per-node sizes.  Declaring ``"vertex"``
        with no sizes in the store raises at commit rather than falling
        back.
    node_visible : bool
        Show the node sub-visual.  Nests under ``visible``.
    node_pick_write : bool
        Write nodes into the pick buffer.
    node_depth_compare : str
        Defaults to ``"<="``, not ``BaseAppearance``'s ``"<"``, so a node
        coplanar with an edge wins the tie and reads on top (D22).  This is
        the mechanism -- ``render_order`` was measured and does nothing for
        coplanar children.  User-overridable like any other field.
    edge_color : tuple[float, float, float, float]
        RGBA uniform edge colour.  A dimmer grey than the node default so
        nodes read on top without any depth trickery.
    edge_thickness : float
        Edge thickness, in ``edge_thickness_space`` units.
    edge_thickness_space : str
        ``"screen"`` (logical pixels) or ``"world"``.
    edge_color_mode : str
        As ``node_color_mode``, for edges.
    edge_visible : bool
        Show the edge sub-visual.
    edge_pick_write : bool
        Write edges into the pick buffer.
    edge_depth_compare : str
        Standard ``"<"``; see ``node_depth_compare`` for why they differ.
    """

    node_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    node_size: float = 5.0
    node_size_space: Literal["screen", "world"] = "screen"
    node_color_mode: Literal["uniform", "vertex"] = "uniform"
    node_size_mode: Literal["uniform", "vertex"] = "uniform"
    node_visible: bool = True
    node_pick_write: bool = True
    node_depth_compare: Literal["<", "<=", "==", "!=", ">=", ">"] = "<="

    edge_color: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    edge_thickness: float = 2.0
    edge_thickness_space: Literal["screen", "world"] = "screen"
    edge_color_mode: Literal["uniform", "vertex"] = "uniform"
    edge_visible: bool = True
    edge_pick_write: bool = True
    edge_depth_compare: Literal["<", "<=", "==", "!=", ">=", ">"] = "<"


class GraphVisual(BaseVisual):
    """Model-layer visual for a spatial graph backed by ``GraphMemoryStore``.

    Rendered as a pygfx compound: one group holding a points node and a
    line-segment node (D1).

    ``trail`` lives on the visual rather than on the appearance because it
    selects *which data is fetched*, so every change to it must trigger a
    reslice.  Keys are **axis indices**; axis names were considered and
    rejected to keep the visual decoupled from the store's metadata and
    consistent with the rest of the dims API.

    An out-of-range trail axis raises ``ValueError``; a valid-but-currently-
    displayed one warns once per (visual, axis).  Both live in the
    controller, because the visual holds only ``data_store_id`` while
    ``ndim`` is on the store (D21).

    Parameters
    ----------
    appearance : GraphAppearance
        Initial appearance.
    trail : dict[int, TrailConfig]
        Axis index -> window configuration.  Not restricted to one axis;
        alphas multiply across axes.
    requires_camera_reslice : bool
        Always False -- graphs do not depend on camera position. Frozen.
    """

    visual_type: Literal["graph_memory"] = "graph_memory"
    appearance: GraphAppearance = Field(default_factory=GraphAppearance)
    trail: dict[int, TrailConfig] = Field(default_factory=dict)
    requires_camera_reslice: bool = Field(default=False, frozen=True)
