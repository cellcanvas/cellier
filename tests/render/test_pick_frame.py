"""`PickFrame` decodes a pixel exactly as ``renderer.get_pick_info`` does.

`PickFrame` reads the pick target once where ``get_pick_info`` reads a pixel
per call, and uses pygfx's private parts to decode it; this is the check
that the two agree, on every kind of volume the tests pick from.
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx
import pytest
from rendercanvas.offscreen import RenderCanvas as OffscreenRenderCanvas

from tests.render import _clipping as h
from tests.render._pick import PickFrame

#: Not square, and not a multiple of 256 bytes a row.
SIZE = (150, 97)
KINDS = [
    "image_memory_iso",
    "image_multiscale_iso",
    "labels_memory",
    "labels_multiscale",
]


@pytest.mark.parametrize("kind", KINDS)
async def test_a_pick_frame_agrees_with_get_pick_info(
    kind, controller, reslice, tmp_path
):
    scene = controller.add_scene(dim="3d", name=kind)
    h.add_visual(kind, controller, scene.id, h.data_for(kind), tmp_path, kind)
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    controller.fit_camera(scene.id)
    gfx_scene, camera = h.gfx_scene(controller, scene.id)

    canvas = OffscreenRenderCanvas(size=SIZE, pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    renderer.pixel_scale = 1
    canvas.request_draw(lambda: renderer.render(gfx_scene, camera))
    frame = np.asarray(canvas.draw())
    picks = PickFrame(renderer)

    drawn_rows, drawn_cols = np.nonzero(frame[..., 3] > 0)
    assert len(drawn_rows) > 100
    rng = np.random.default_rng(0)
    take = rng.choice(len(drawn_rows), size=20, replace=False)
    pixels = list(zip(drawn_rows[take], drawn_cols[take]))
    # And the corners, where nothing is drawn.
    pixels += [(0, 0), (SIZE[1] - 1, SIZE[0] - 1)]

    hits = 0
    for row, col in pixels:
        pos = (col + 0.5, row + 0.5)
        expected = renderer.get_pick_info(pos)
        expected.pop("rgba")
        assert picks.info(pos) == expected
        hits += expected["world_object"] is not None
    assert hits >= 10
