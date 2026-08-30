"""SceneManager — owns one pygfx Scene and its registered visuals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygfx as gfx

from cellier.render._scene_config import VisualRenderConfig

if TYPE_CHECKING:
    from uuid import UUID

    from cellier.data.image import ChunkRequest
    from cellier.render._requests import ReslicingRequest
    from cellier.render.visuals._graph_memory import GFXGraphMemoryVisual
    from cellier.render.visuals._image import GFXMultiscaleImageVisual
    from cellier.render.visuals._image_memory import GFXImageMemoryVisual
    from cellier.render.visuals._image_memory_multichannel import (
        GFXMultichannelImageMemoryVisual,
    )
    from cellier.render.visuals._image_multiscale_multichannel import (
        GFXMultichannelMultiscaleImageVisual,
    )
    from cellier.render.visuals._lines_memory import GFXLinesMemoryVisual
    from cellier.render.visuals._mesh_memory import GFXMeshMemoryVisual
    from cellier.render.visuals._points_memory import GFXPointsMemoryVisual

    _GFXVisual = (
        GFXMultiscaleImageVisual
        | GFXImageMemoryVisual
        | GFXMultichannelImageMemoryVisual
        | GFXMultichannelMultiscaleImageVisual
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
    """

    def __init__(
        self,
        scene_id: UUID,
        lighting: str = "none",
    ) -> None:
        self._scene_id = scene_id
        self._scene = gfx.Scene()
        self._visuals: dict[UUID, _GFXVisual] = {}
        self._active_nodes: dict[UUID, gfx.WorldObject | None] = {}

        self._has_lighting = lighting == "default"
        if self._has_lighting:
            self._scene.add(gfx.AmbientLight(intensity=0.4))
            dir_light = gfx.DirectionalLight(intensity=3.0)
            dir_light.local.position = np.array([500, 500, 1000], dtype=np.float32)
            self._scene.add(dir_light)

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

    def add_visual(self, visual: _GFXVisual, displayed_axes: tuple[int, ...]) -> None:
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
        self._scene.add(node)
        self._visuals[visual.visual_model_id] = visual
        self._active_nodes[visual.visual_model_id] = node

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
        active_node = self._active_nodes.pop(visual_id, None)
        if active_node is not None:
            self._scene.remove(active_node)
        self._visuals.pop(visual_id)

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

            cfg = visual_configs.get(visual_id, VisualRenderConfig())

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

            cfg = visual_configs.get(visual_id, VisualRenderConfig())

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
            )
            if chunk_requests:
                result[visual_id] = chunk_requests

        return result
