"""SceneManager — owns one pygfx Scene and its registered visuals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygfx as gfx

from cellier.render._scene_config import VisualRenderConfig
from cellier.render._spaces import data_slice_positions, visual_covers_position
from cellier.scene._background import BackgroundAppearance
from cellier.transform import (
    NonAffineTransformError,
    NonInvertibleTransformError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from cellier.data.image import ChunkRequest
    from cellier.render._requests import ReslicingRequest
    from cellier.render.visuals._graph_memory import GFXGraphMemoryVisual
    from cellier.render.visuals._image import GFXMultiscaleImageVisual
    from cellier.render.visuals._image_memory import GFXImageMemoryVisual
    from cellier.render.visuals._lines_memory import GFXLinesMemoryVisual
    from cellier.render.visuals._mesh_memory import GFXMeshMemoryVisual
    from cellier.render.visuals._points_memory import GFXPointsMemoryVisual
    from cellier.render.visuals._scene_overlay import GFXSceneOverlay

    _GFXVisual = (
        GFXMultiscaleImageVisual
        | GFXImageMemoryVisual
        | GFXPointsMemoryVisual
        | GFXLinesMemoryVisual
        | GFXMeshMemoryVisual
        | GFXGraphMemoryVisual
    )


class SceneManager:
    """Owns one pygfx ``gfx.Scene`` and the registry of visuals attached to it.

    Which node (2D or 3D) is active is determined at runtime from the
    ``dims_state`` carried by each ``ReslicingRequest``, not fixed at
    construction time.

    Parameters
    ----------
    scene_id : UUID
        Unique identifier for this scene.
    lighting : str
        ``"none"`` (default) or ``"default"``.
    background : BackgroundAppearance or None
        Background appearance to apply at construction.  ``None`` uses the
        model defaults.
    """

    def __init__(
        self,
        scene_id: UUID,
        lighting: str = "none",
        background: BackgroundAppearance | None = None,
    ) -> None:
        self._scene_id = scene_id
        self._scene = gfx.Scene()
        # The background object and its material are kept so the colors can be
        # rewritten in place at runtime; set_colors writes the material's
        # uniform buffer, so nothing is rebuilt on a change.
        self._background_material = gfx.BackgroundMaterial()
        self._background = gfx.Background(None, self._background_material)
        self._scene.add(self._background)
        self.set_background(
            background if background is not None else BackgroundAppearance()
        )
        self._visuals: dict[UUID, _GFXVisual] = {}
        self._active_nodes: dict[UUID, gfx.WorldObject | None] = {}
        # Scene overlays, keyed by overlay model id.  Their nodes live in the
        # same gfx.Scene as the visuals but are not visuals: they are never
        # sliced and never picked.
        self._overlays: dict[UUID, GFXSceneOverlay] = {}
        # Per-visual store extents, for the out-of-domain check in
        # build_slice_requests.  Absent means "not known", which never skips.
        self._axis_extents: dict[UUID, Sequence[tuple[float, float]] | None] = {}

        self._has_lighting = lighting == "default"
        if self._has_lighting:
            self._scene.add(gfx.AmbientLight(intensity=0.4))
            dir_light = gfx.DirectionalLight(intensity=3.0)
            dir_light.local.position = np.array([500, 500, 1000], dtype=np.float32)
            self._scene.add(dir_light)

    @property
    def background(self) -> gfx.Background:
        """The pygfx background object drawn behind this scene's visuals."""
        return self._background

    def set_background(self, appearance: BackgroundAppearance) -> None:
        """Apply *appearance* to the scene's background object.

        The whole model is applied rather than a single field so the render
        layer never has to dispatch on field names -- ``set_colors`` takes a
        different number of colors per mode, so a per-field push would have
        to reconstruct the others anyway.

        Parameters
        ----------
        appearance : BackgroundAppearance
            The background appearance to apply.
        """
        self._background_material.set_colors(*appearance.to_colors())
        self._background.visible = appearance.visible

    @property
    def has_lighting(self) -> bool:
        """True if this scene was created with lighting enabled."""
        return self._has_lighting

    @property
    def scene_id(self) -> UUID:
        """Unique identifier for this scene."""
        return self._scene_id

    @property
    def scene(self) -> gfx.Scene:
        """The pygfx Scene object. Passed to CanvasView via get_scene_fn."""
        return self._scene

    @property
    def visual_ids(self) -> list[UUID]:
        """IDs of all registered visuals."""
        return list(self._visuals.keys())

    def add_visual(
        self,
        visual: _GFXVisual,
        displayed_axes: tuple[int, ...],
        axis_extents: Sequence[tuple[float, float]] | None = None,
    ) -> None:
        """Register a visual and add its initial node to the scene graph.

        Calls ``visual.get_node_for_dims(displayed_axes)`` to select the
        correct node, then stores it in ``_active_nodes`` and adds it to
        the pygfx scene.

        Parameters
        ----------
        visual : _GFXVisual
            The GFX visual to register.
        displayed_axes : tuple[int, ...]
            Current displayed axes from the scene's dims selection.
        axis_extents : Sequence[tuple[float, float]] or None
            The backing store's per-axis extents in level-0 data
            coordinates, used to decide whether this visual has any data at
            a given slice position.  ``None`` means "not known", and such a
            visual is never skipped.

        Raises
        ------
        ValueError
            If ``get_node_for_dims`` returns ``None``.
        """
        node = visual.get_node_for_dims(displayed_axes)
        if node is None:
            raise ValueError(
                f"Visual {visual.visual_model_id} returned None from "
                f"get_node_for_dims({displayed_axes!r}). "
                "Ensure render_modes includes the required dimensionality."
            )
        self._axis_extents[visual.visual_model_id] = axis_extents
        self._scene.add(node)
        self._visuals[visual.visual_model_id] = visual
        self._active_nodes[visual.visual_model_id] = node

    def set_axis_extents(
        self,
        visual_id: UUID,
        axis_extents: Sequence[tuple[float, float]] | None,
    ) -> None:
        """Replace a visual's store extents after its store's extent changed.

        Parameters
        ----------
        visual_id : UUID
            ID of a registered visual.  An unknown id is ignored.
        axis_extents : Sequence[tuple[float, float]] or None
            The store's new level-0 extents, or ``None`` for "not known".
        """
        if visual_id in self._visuals:
            self._axis_extents[visual_id] = axis_extents

    def get_active_node(self, visual_id: UUID) -> gfx.WorldObject | None:
        """Return the node currently active in the scene for *visual_id*.

        Returns ``None`` if the visual has not yet been registered or if its
        active node is ``None``.

        Parameters
        ----------
        visual_id : UUID
            ID of the visual to query.
        """
        return self._active_nodes.get(visual_id)

    def swap_node(self, visual_id: UUID, new_node: gfx.WorldObject | None) -> None:
        """Replace the active scene-graph node for *visual_id*.

        Removes the previously active node and adds *new_node*.  If
        ``old_node is new_node`` (single-node visuals such as mesh and
        lines), the scene graph is not touched — the node stays in the
        scene and the subsequent reslice updates its content.

        Parameters
        ----------
        visual_id : UUID
            ID of the visual whose node is being swapped.
        new_node : gfx.WorldObject or None
            The node that should be active after this call.
        """
        old_node = self._active_nodes.get(visual_id)

        if old_node is new_node:
            # Single-node visual or no change — leave scene graph alone.
            return

        if old_node is not None:
            self._scene.remove(old_node)

        if new_node is not None:
            self._scene.add(new_node)

        self._active_nodes[visual_id] = new_node

    def add_overlay(self, overlay_id: UUID, overlay: GFXSceneOverlay) -> None:
        """Register a scene overlay and add its node to the scene graph.

        Parameters
        ----------
        overlay_id : UUID
            ID of the overlay's model.
        overlay : GFXSceneOverlay
            The render-layer overlay.
        """
        self._overlays[overlay_id] = overlay
        self._scene.add(overlay.node)

    def remove_overlay(self, overlay_id: UUID) -> None:
        """Unregister a scene overlay and remove its node from the scene graph.

        Parameters
        ----------
        overlay_id : UUID
            ID of the overlay's model.  An unknown id is ignored.
        """
        overlay = self._overlays.pop(overlay_id, None)
        if overlay is not None:
            self._scene.remove(overlay.node)

    def get_overlay(self, overlay_id: UUID) -> GFXSceneOverlay | None:
        """Return the render-layer scene overlay for *overlay_id*, if any."""
        return self._overlays.get(overlay_id)

    def remove_visual(self, visual_id: UUID) -> None:
        """Unregister a visual and remove its node from the scene graph.

        Uses ``_active_nodes`` to identify which node is currently in the
        scene and removes only that one.  Dropping the visual from
        ``_visuals`` releases references to both ``node_3d`` and ``node_2d``
        so GC can collect both nodes' GPU resources (pygfx has no explicit
        destroy API).

        Parameters
        ----------
        visual_id : UUID
            ID of the visual to remove.
        """
        self._axis_extents.pop(visual_id, None)
        active_node = self._active_nodes.pop(visual_id, None)
        if active_node is not None:
            self._scene.remove(active_node)
        visual = self._visuals.pop(visual_id)
        # Explicit release for visuals that hold slots, caches or model
        # connections (unified image design 3.8), rather than trusting GC.
        close = getattr(visual, "close", None)
        if close is not None:
            close()

    def close(self) -> None:
        """Remove every visual and overlay, releasing their GPU resources.

        Something may still reference this scene manager after its scene is
        gone -- a closed controller kept alive, a pending slice task -- so the
        visuals are released here rather than left for when the manager dies.
        Safe to call more than once.
        """
        for visual_id in list(self._visuals):
            self.remove_visual(visual_id)
        for overlay_id in list(self._overlays):
            self.remove_overlay(overlay_id)

    def get_visual_id_for_node(self, node: gfx.WorldObject) -> UUID | None:
        """Return the visual_id whose active scene-graph node is *node*.

        The pick buffer returns leaf nodes (e.g. ``gfx.Image`` inside a
        ``gfx.Group``), while ``_active_nodes`` stores the top-level group.
        This method walks up the parent chain of *node* until it finds a
        registered active node, then returns its visual_id.  Returns None
        if no ancestor belongs to any registered visual.

        Parameters
        ----------
        node : gfx.WorldObject
            The pygfx object returned by the pick buffer.
        """
        candidate = node
        while candidate is not None:
            for visual_id, active_node in self._active_nodes.items():
                if active_node is candidate:
                    return visual_id
            candidate = candidate.parent
        return None

    def get_visual(self, visual_id: UUID) -> _GFXVisual:
        """Return the registered visual for ``visual_id``.

        Parameters
        ----------
        visual_id : UUID
            ID of the visual to retrieve.

        Returns
        -------
        _GFXVisual

        Raises
        ------
        KeyError
            If ``visual_id`` is not registered in this scene.
        """
        return self._visuals[visual_id]

    def build_slice_requests(
        self,
        request: ReslicingRequest,
        visual_configs: dict[UUID, VisualRenderConfig],
    ) -> dict[UUID, list[ChunkRequest]]:
        """Collect ChunkRequests from all (or targeted) registered visuals.

        Dispatches to the 2D or 3D planning path based on scene
        dimensionality.

        Parameters
        ----------
        request : ReslicingRequest
            The reslicing request.
        visual_configs : dict[UUID, VisualRenderConfig]
            Per-visual render configuration.

        Returns
        -------
        dict[UUID, list[ChunkRequest]]
            Mapping of ``visual_model_id`` to that visual's ChunkRequests.
        """
        if len(request.dims_state.selection.displayed_axes) == 2:
            return self._build_slice_requests_2d(request, visual_configs)
        return self._build_slice_requests_3d(request, visual_configs)

    def _has_data_here(self, visual_id: UUID, request: ReslicingRequest) -> bool:
        """Whether a visual has any data at this request's slice positions.

        Compared entirely in the visual's own **data** coordinates: the
        selection is pulled back through its transform by
        ``data_slice_positions`` and checked against its store's extents.

        Only reachable in a **mixed-extent scene** -- a scene's slider range
        is the union of its visuals' extents, so a single-visual scene never
        leaves its own.  A visual whose extents or spaces are unknown is
        never skipped.

        A visual that sets ``decides_empty_slices`` is never skipped either.
        The image visuals apply their own slicing rule (design 3.2) and must
        see the request to hide their data node when the slice misses the
        data; skipped, they would keep drawing the last plane they loaded.
        """
        visual = self._visuals[visual_id]
        if getattr(visual, "decides_empty_slices", False):
            return True
        extents = self._axis_extents.get(visual_id)
        if extents is None:
            return True
        transform = getattr(visual, "_transform", None)
        spaces = getattr(visual, "_spaces", None)
        selection = getattr(request, "selection", None)
        if transform is None or spaces is None or selection is None:
            return True
        try:
            positions = data_slice_positions(selection.region, transform, spaces.world)
        except (ValueError, NonAffineTransformError, NonInvertibleTransformError):
            # The region cannot be pulled back here; leave the decision to
            # the visual's own planner rather than blanking it.
            return True
        return visual_covers_position(extents, positions)

    def _build_slice_requests_3d(
        self,
        request: ReslicingRequest,
        visual_configs: dict[UUID, VisualRenderConfig],
    ) -> dict[UUID, list[ChunkRequest]]:
        """3D planning path using perspective camera and frustum culling."""
        result: dict[UUID, list[ChunkRequest]] = {}
        _, screen_height_px = request.screen_size_px
        for visual_id, visual in self._visuals.items():
            if (
                request.target_visual_ids is not None
                and visual_id not in request.target_visual_ids
            ):
                continue
            if not self._has_data_here(visual_id, request):
                continue

            cfg = visual_configs.get(visual_id, VisualRenderConfig())
            if not cfg.slicing_enabled:
                continue

            frustum_corners_world = (
                request.frustum_corners if cfg.frustum_cull else None
            )

            chunk_requests = visual.build_slice_request(
                camera_pos_world=request.camera_pos,
                frustum_corners_world=frustum_corners_world,
                fov_y_rad=request.fov_y_rad,
                screen_height_px=screen_height_px,
                lod_bias=cfg.lod_bias,
                dims_state=request.dims_state,
                force_level=cfg.force_level,
                selection=request.selection,
            )
            if chunk_requests:
                result[visual_id] = chunk_requests

        return result

    def _build_slice_requests_2d(
        self,
        request: ReslicingRequest,
        visual_configs: dict[UUID, VisualRenderConfig],
    ) -> dict[UUID, list[ChunkRequest]]:
        """2D planning path using orthographic camera and viewport culling."""
        result: dict[UUID, list[ChunkRequest]] = {}

        world_width, world_height = request.world_extent
        viewport_width_px, _viewport_height_px = request.screen_size_px

        # Compute viewport AABB from camera position and world extent.
        cx = float(request.camera_pos[0])
        cy = float(request.camera_pos[1])
        half_w = world_width / 2.0
        half_h = world_height / 2.0
        view_min = np.array([cx - half_w, cy - half_h], dtype=np.float64)
        view_max = np.array([cx + half_w, cy + half_h], dtype=np.float64)

        for visual_id, visual in self._visuals.items():
            if (
                request.target_visual_ids is not None
                and visual_id not in request.target_visual_ids
            ):
                continue
            if not self._has_data_here(visual_id, request):
                continue

            cfg = visual_configs.get(visual_id, VisualRenderConfig())
            if not cfg.slicing_enabled:
                continue

            chunk_requests = visual.build_slice_request_2d(
                camera_pos_world=request.camera_pos,
                viewport_width_px=viewport_width_px,
                world_width=world_width,
                view_min_world=view_min if cfg.frustum_cull else None,
                view_max_world=view_max if cfg.frustum_cull else None,
                dims_state=request.dims_state,
                lod_bias=cfg.lod_bias,
                force_level=cfg.force_level,
                use_culling=cfg.frustum_cull,
                selection=request.selection,
            )
            if chunk_requests:
                result[visual_id] = chunk_requests

        return result
