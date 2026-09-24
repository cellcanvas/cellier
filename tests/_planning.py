"""What a 3D multiscale plan would fetch, for tests that inspect requests.

Multiscale visuals no longer build store requests while planning: ``plan()``
returns desired sets of packed keys, and the chunk scheduler builds requests
as it issues reads (``plans/progressive_loading_design_v3.md`` 5.3).  Tests
that assert on the requests a plan implies -- which planes, which levels,
which channels -- ask for all of them here: every wanted key, as into an
empty atlas, in the order the planner wants them (drawn channels
interleaved, as the old per-channel materialisation produced them).
"""

from __future__ import annotations

from itertools import zip_longest
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np

from cellier.render._requests import ReslicingRequest
from cellier.render._scene_config import VisualRenderConfig
from cellier.visuals import ProgressiveLoadingConfig

if TYPE_CHECKING:
    from cellier.data.image import ChunkRequest


def planned_requests_3d(
    visual: Any,
    camera_pos_world: np.ndarray,
    frustum_corners_world: np.ndarray | None,
    fov_y_rad: float,
    screen_height_px: float,
    lod_bias: float = 1.0,
    dims_state: Any = None,
    force_level: int | None = None,
    selection: Any = None,
    backstop: bool = False,
) -> list[ChunkRequest]:
    """Every store request a 3D plan of *visual* wants, in load order.

    *visual* is a ``GFXMultiscaleImageVisual``, a ``GFXMultiscaleLabelVisual``
    or a bare ``_MultiscaleImageSlot``; any other visual's own
    ``build_slice_request`` answers.  The arguments are
    ``build_slice_request``'s, plus *backstop*: whether the plan includes
    the coarse backstop (off by default, so the requests are the target's).
    """
    from cellier.render.scheduling import is_chunked_visual
    from cellier.render.visuals._image import _MultiscaleImageSlot

    if not is_chunked_visual(visual) and not isinstance(visual, _MultiscaleImageSlot):
        # Not loaded by the chunk scheduler: its planner still builds requests.
        return visual.build_slice_request(
            camera_pos_world=camera_pos_world,
            frustum_corners_world=frustum_corners_world,
            fov_y_rad=fov_y_rad,
            screen_height_px=screen_height_px,
            lod_bias=lod_bias,
            dims_state=dims_state,
            force_level=force_level,
            selection=selection,
        )
    if isinstance(visual, _MultiscaleImageSlot):
        visual._begin_region_planning(selection)
        if visual._slice_empty:
            return []
        if dims_state is not None:
            displayed = tuple(dims_state.selection.displayed_axes)
            if displayed != visual._last_displayed_axes:
                visual._update_node_matrix(displayed)
        brick_arr = visual._plan_bricks(
            np.asarray(camera_pos_world),
            frustum_corners_world,
            fov_y_rad,
            screen_height_px,
            lod_bias,
            force_level,
        )
        desired = [visual.desired_set_3d(brick_arr)]
    else:
        request = ReslicingRequest(
            camera_type="perspective",
            camera_pos=np.asarray(camera_pos_world),
            frustum_corners=frustum_corners_world,
            fov_y_rad=fov_y_rad,
            screen_size_px=(screen_height_px, screen_height_px),
            world_extent=(0.0, 0.0),
            dims_state=dims_state,
            selection=selection,
            request_id=uuid4(),
            scene_id=uuid4(),
            canvas_id=uuid4(),
            target_visual_ids=None,
        )
        config = VisualRenderConfig(
            lod_bias=lod_bias,
            force_level=force_level,
            frustum_cull=frustum_corners_world is not None,
            loading=ProgressiveLoadingConfig(backstop=backstop),
        )
        desired = visual.plan(request, config)
    groups = [ds.build_request(ds.keys) for ds in desired if ds is not None]
    return [r for group in zip_longest(*groups) for r in group if r is not None]
