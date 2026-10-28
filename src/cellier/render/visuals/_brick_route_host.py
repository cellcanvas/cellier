"""The in-memory visuals' brick route: delegation to the multiscale slot code.

Oblique 2D design v3, 5.4 (I-iii+).  A tilted 2D view of an in-memory image
or labels visual is drawn by the multiscale brick route.  The in-memory
render visual owns, from its first such view on, an *inner* multiscale
render visual (``GFXMultiscaleImageVisual`` or ``GFXMultiscaleLabelVisual``)
built over an :class:`~cellier.data._in_memory_bricks.InMemoryBrickSource`,
in the 3D mode only.  While the route is taken the in-memory visual hands
the scene the inner visual's slice node, plans through it on the chunk
scheduler, and forwards its events to it; aligned 2D views and 3D views
keep today's whole-slice path (I4: one implementation of each piece).

:class:`BrickRouteHost` is that delegation, mixed in ahead of the in-memory
visual's own class.  The inner visual reads the in-memory model through
:class:`_RoutedModel`, which adds the two-level pyramid's ``level_transforms``
and a render config, and gives the multiscale-only appearance fields their
multiscale defaults.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np

from cellier.render.scheduling import ChunkClass, PlanMode

if TYPE_CHECKING:
    from cellier.data._in_memory_bricks import InMemoryBrickSource
    from cellier.render.scheduling import DesiredSet

#: Appearance fields the multiscale visuals read that an in-memory
#: appearance does not have, with the multiscale models' defaults.
_MULTISCALE_DEFAULTS: dict[str, Any] = {
    "attenuation": 1.0,
    "ray_steps_per_voxel": 1.0,
    "frustum_cull": True,
    "coarsest_while_moving_3d": False,
}


class _RoutedAppearance:
    """An in-memory appearance as the multiscale visuals read one."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def __getattr__(self, name: str) -> Any:
        appearance = self._model.appearance
        try:
            return getattr(appearance, name)
        except AttributeError:
            if name in _MULTISCALE_DEFAULTS:
                return _MULTISCALE_DEFAULTS[name]
            raise


@dataclasses.dataclass(frozen=True)
class _RoutedRenderConfig:
    """The in-memory render config, with the fields the 3D-only route ignores."""

    block_size: int
    gpu_budget_bytes: int
    loading: Any
    gpu_budget_bytes_2d: int = 1
    paint_max_tiles: int = 1


class _RoutedModel:
    """An in-memory visual model as the multiscale render visuals read one.

    Reads through to *model*, so appearance changes are seen live; adds the
    brick source's ``level_transforms`` and a render config.
    """

    def __init__(self, model: Any, source: InMemoryBrickSource) -> None:
        self._model = model
        self.level_transforms = list(source.level_transforms)
        config = model.render_config
        self.render_config = _RoutedRenderConfig(
            block_size=config.block_size,
            gpu_budget_bytes=config.gpu_budget_bytes,
            loading=config.loading,
        )
        self.appearance = _RoutedAppearance(model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def _without_backstop(desired: DesiredSet) -> DesiredSet:
    """*desired* with its backstop keys dropped (target-only planning, U8)."""
    keep = desired.cls != np.uint8(ChunkClass.BACKSTOP)
    return dataclasses.replace(
        desired,
        keys=desired.keys[keep],
        cls=desired.cls[keep],
        slice_ids=desired.slice_ids[keep],
        n_truncated_backstop=0,
    )


class BrickRouteHost:
    """Delegates a tilted 2D view of an in-memory visual to the brick route.

    Mixed in ahead of ``GFXImageMemoryVisual`` and ``GFXLabelMemoryVisual``.
    The host class names the inner multiscale visual's class
    (:meth:`_build_inner`).  The controller hands it its brick source
    (:meth:`set_brick_source`) when the visual is added, and its route
    before its render spaces (:meth:`set_slice_route`, design v3 C3).
    """

    #: Read by ``is_chunked_visual``: on the route, the chunk scheduler
    #: loads this visual (its :attr:`chunked_now`).
    routes_chunked: bool = True

    _brick_source: InMemoryBrickSource | None = None
    _brick_inner: Any = None
    _route_axes: tuple[int, int, int] | None = None

    # ── Route state ─────────────────────────────────────────────────────

    @property
    def chunked_now(self) -> bool:
        """Whether the chunk scheduler loads this visual now (on the route)."""
        return self._route_axes is not None and self._brick_inner is not None

    @property
    def brick_source(self) -> InMemoryBrickSource | None:
        """The two-level pyramid the route reads, or ``None`` before add."""
        return self._brick_source

    @property
    def brick_inner(self) -> Any:
        """The inner multiscale visual, or ``None`` before the route is taken."""
        return self._brick_inner

    def set_brick_source(self, source: InMemoryBrickSource | None) -> None:
        """Adopt a (new) brick source; an existing inner visual is dropped."""
        if self._brick_source is not None and self._brick_source is not source:
            self._brick_source.close()
        self._brick_source = source
        self._drop_inner()

    def _drop_inner(self) -> None:
        if self._brick_inner is not None:
            self._brick_inner.close()
        self._brick_inner = None

    def _ensure_inner(self, axes: tuple[int, int, int]) -> Any:
        if self._brick_inner is None:
            if self._brick_source is None:
                raise RuntimeError(
                    "This in-memory visual has no brick source: the controller "
                    "gives it one when it is added."
                )
            self._brick_inner = self._build_inner(
                _RoutedModel(self._visual_model, self._brick_source), axes
            )
            spaces = getattr(self, "_route_spaces", None)
            if spaces is not None:
                self._brick_inner.set_render_spaces(spaces)
            if self.clipping_planes:
                self._brick_inner.set_clipping_planes(self.clipping_planes)
            slab = getattr(self, "_route_slab", None)
            setter = getattr(self._brick_inner, "set_slab", None)
            if slab is not None and setter is not None:
                setter(slab)
        return self._brick_inner

    def _build_inner(self, model: _RoutedModel, axes: tuple[int, int, int]) -> Any:
        raise NotImplementedError

    def set_slice_route(self, axes: tuple[int, int, int] | None) -> bool:
        """Put the 2D views of this visual on the brick route, or take them off.

        Returns
        -------
        bool
            Whether the route changed (the caller then swaps the node in).
        """
        axes = None if axes is None else tuple(int(axis) for axis in axes)
        if axes == self._route_axes:
            return False
        if axes is not None:
            self._ensure_inner(axes)
        self._route_axes = axes
        if self._brick_inner is not None:
            self._brick_inner.set_slice_route(axes)
            if axes is None:
                # The scheduler forgets the atlas when the route is left
                # (the controller); what it drew is cleared with it.
                self._reset_inner_residency()
        return True

    def _reset_inner_residency(self) -> None:
        inner = self._brick_inner
        for slot in getattr(inner, "slots", None) or (inner,):
            cache = getattr(slot, "_block_cache_3d", None)
            if cache is not None:
                cache.clear()
            slot._residency_3d = None

    def prepare_slice_route(self, axes: tuple[int, int, int]) -> None:
        """Build the inner visual and put its atlas on the GPU now (U12)."""
        self._ensure_inner(tuple(int(axis) for axis in axes)).prepare_slice_route(axes)

    @property
    def slice_axes(self) -> tuple[int, int, int] | None:
        """The slice node's world axes on the brick route, else ``None``."""
        return self._route_axes

    @property
    def slice_node(self) -> Any:
        """The node the scene draws on the brick route."""
        if self._brick_inner is None:
            return None
        return self._brick_inner.slice_node

    def set_slab(self, slab) -> None:
        """Hand the slab projection to the inner visual (images only)."""
        self._route_slab = slab
        setter = getattr(self._brick_inner, "set_slab", None)
        if setter is not None:
            setter(slab)

    @property
    def slab(self):
        """The slab the slice node projects, or ``None``."""
        return getattr(self, "_route_slab", None)

    def set_slice_footprint(self, polygon: np.ndarray) -> None:
        """Hand the footprint to the inner visual (bounds, fit, outline)."""
        if self._brick_inner is not None:
            self._brick_inner.set_slice_footprint(polygon)

    def _routed(self, displayed_axes: tuple[int, ...] | None = None) -> bool:
        if not self.chunked_now:
            return False
        return displayed_axes is None or len(displayed_axes) == 2

    # ── Nodes and spaces ────────────────────────────────────────────────

    def get_node_for_dims(self, displayed_axes: tuple[int, ...]):
        if self._routed(displayed_axes):
            return self._brick_inner.get_node_for_dims(tuple(displayed_axes))
        return super().get_node_for_dims(displayed_axes)

    def build_node(self, mode, visual_model, displayed_axes, level_shapes, transforms):
        if self._routed(displayed_axes):
            return self.get_node_for_dims(displayed_axes)
        return super().build_node(
            mode, visual_model, displayed_axes, level_shapes, transforms
        )

    def rebuild_node_geometry(self, mode, displayed_axes, level_shapes, transforms):
        if self._routed(displayed_axes):
            return self.get_node_for_dims(displayed_axes)
        return super().rebuild_node_geometry(
            mode, displayed_axes, level_shapes, transforms
        )

    def set_render_spaces(self, spaces) -> None:
        """On the route the spaces are the slice node's: the inner visual's."""
        if self._route_axes is not None:
            self._route_spaces = spaces
            if self._brick_inner is not None:
                self._brick_inner.set_render_spaces(spaces)
            return
        super().set_render_spaces(spaces)

    # ── Planning (the chunk scheduler's contract) ────────────────────────

    def plan(self, request, config, mode: PlanMode = PlanMode.FULL) -> list:
        """The inner visual's plan, reading from the brick source.

        Until the coarse level is built the plan is target-only at the
        array's own level: no backstop, and nothing that would read the
        coarse level (U8).
        """
        inner = self._brick_inner
        source = self._brick_source
        if inner is None or source is None or self._route_axes is None:
            return []
        built = source.built
        if not built:
            mode = PlanMode.FULL
            config = dataclasses.replace(config, force_level=1)
        desired = inner.plan(request, config, mode)
        if not built:
            desired = [_without_backstop(ds) for ds in desired]
        return [dataclasses.replace(ds, store=source) for ds in desired]

    def residencies(self) -> dict:
        """The inner visual's atlases while the route is taken."""
        if not self.chunked_now:
            return {}
        return self._brick_inner.residencies()

    # ── Picks ───────────────────────────────────────────────────────────

    def _inner_hit(self, hit_object) -> bool:
        node = self.slice_node
        candidate = hit_object
        while candidate is not None:
            if candidate is node:
                return True
            candidate = candidate.parent
        return False

    def pick_data_coordinate(self, hit_object, pick_info: dict):
        if self.chunked_now and self._inner_hit(hit_object):
            return self._brick_inner.pick_data_coordinate(hit_object, pick_info)
        return super().pick_data_coordinate(hit_object, pick_info)

    def pick_collapsed_indices(self):
        if self.chunked_now:
            return self._brick_inner.pick_collapsed_indices()
        return super().pick_collapsed_indices()

    def pick_channel_index(self, hit_object):
        if self.chunked_now and self._inner_hit(hit_object):
            inner = getattr(self._brick_inner, "pick_channel_index", None)
            return None if inner is None else inner(hit_object)
        own = getattr(super(), "pick_channel_index", None)
        return None if own is None else own(hit_object)

    def drawn_channel_indices(self) -> tuple[int, ...]:
        if self.chunked_now:
            inner = getattr(self._brick_inner, "drawn_channel_indices", None)
            if inner is not None:
                return inner()
        own = getattr(super(), "drawn_channel_indices", None)
        return () if own is None else own()

    # ── Events: today's nodes, and the inner visual when it exists ───────

    def _forward(self, name: str, event) -> None:
        handler = getattr(self._brick_inner, name, None)
        if handler is not None:
            handler(event)

    def on_transform_changed(self, event) -> None:
        super().on_transform_changed(event)
        self._forward("on_transform_changed", event)

    def on_appearance_changed(self, event) -> None:
        super().on_appearance_changed(event)
        self._forward("on_appearance_changed", event)

    def on_visibility_changed(self, event) -> None:
        super().on_visibility_changed(event)
        self._forward("on_visibility_changed", event)

    def on_pick_write_changed(self, event) -> None:
        super().on_pick_write_changed(event)
        self._forward("on_pick_write_changed", event)

    def on_aabb_changed(self, event) -> None:
        super().on_aabb_changed(event)
        self._forward("on_aabb_changed", event)

    def set_clipping_planes(self, planes) -> None:
        super().set_clipping_planes(planes)
        if self._brick_inner is not None:
            self._brick_inner.set_clipping_planes(self.clipping_planes)

    @property
    def clipping_planes_affect_request(self) -> bool:
        """On the route clipped bricks are not fetched, as for multiscale."""
        if self.chunked_now:
            return True
        return super().clipping_planes_affect_request

    def on_data_store_contents_changed(self, event) -> None:
        """Recompute the touched coarse blocks before the bricks are re-read."""
        if self._brick_source is not None:
            self._brick_source.contents_changed(getattr(event, "regions", None))
        own = getattr(super(), "on_data_store_contents_changed", None)
        if own is not None:
            own(event)

    def close(self) -> None:
        self._drop_inner()
        if self._brick_source is not None:
            self._brick_source.close()
            self._brick_source = None
        super().close()
