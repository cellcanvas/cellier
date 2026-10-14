"""A pick on a label's surface names the voxel that was drawn.

The label volume shader reports the surface as the *exact* crossing of the
face between the background cell and the foreground one.  That value sits on a
voxel boundary, and the 14-bit pick encode truncates, so without a correction
the decoded coordinate falls a fraction short and ``floor`` names the
background cell in front of the surface instead of the cell that was hit.

It only shows on the axis whose face the ray entered through, and only on that
axis's *low* face -- a downward bias at an upper boundary stays inside the same
voxel.  So it is invisible from three of six directions, which is why it
survived until someone clicked the edge of a small cube.  Reverting the shader
fix fails exactly the three low-face views here, in every mode that draws the
voxel faces, which is what makes that claim checkable rather than asserted.
``smooth_geometry`` draws no voxel faces; it reports a point in the middle of
a voxel of the label instead (``label_smooth_pick_point``), checked here too.
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx
import pytest

from cellier.render.shaders._label_colormap import build_label_params_buffer
from cellier.render.shaders._label_volume import LabelVolumeMaterial
from cellier.render.visuals._pick import memory_image_data_coordinate

SIZE = 16
CUBE_LO, CUBE_HI = 6, 10  # inclusive, on every axis
LABEL = 3


def _volume() -> np.ndarray:
    data = np.zeros((SIZE, SIZE, SIZE), dtype=np.int32)
    data[CUBE_LO : CUBE_HI + 1, CUBE_LO : CUBE_HI + 1, CUBE_LO : CUBE_HI + 1] = LABEL
    return data


def _node(render_mode: str) -> gfx.Volume:
    """A labels volume node built the way ``GFXLabelMemoryVisual`` builds it."""
    texture = gfx.Texture(_volume(), dim=3, format="1xi4")
    material = LabelVolumeMaterial(
        background_label=0,
        colormap_mode="random",
        salt=0,
        render_mode=render_mode,
        label_params_buffer=build_label_params_buffer(
            background_label=0, salt=0, n_entries=0
        ),
        pick_write=True,
    )
    return gfx.Volume(gfx.Geometry(grid=texture), material)


#: ``(name, view direction, up)``.  One entry per face of the cube, so whichever
#: axis the ray enters along gets its turn.  The ``up`` differs per axis only
#: because ``OrthographicCamera.show_object`` renders nothing for some
#: view/up pairs; any working pair proves the same thing.
_VIEWS = [
    ("from -x", (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ("from +x", (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ("from -y", (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ("from +y", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    ("from -z", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ("from +z", (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
]


@pytest.mark.parametrize(
    "render_mode",
    ["iso_categorical", "flat_categorical", "smooth_iso", "smooth_geometry"],
)
@pytest.mark.parametrize(("name", "view_dir", "up"), _VIEWS)
def test_the_entry_face_floors_into_the_cube(
    offscreen_renderer, render_mode, name, view_dir, up
):
    """Looking straight down an axis, the centre pixel hits the cube's face.

    Whichever face the ray enters through, ``floor`` of the decoded coordinate
    must land in ``[CUBE_LO, CUBE_HI]`` on every axis -- the cube was what was
    drawn, so the cube is what the pick must name.
    """
    from rendercanvas.offscreen import RenderCanvas

    node = _node(render_mode)
    scene = gfx.Scene()
    scene.add(node)

    camera = gfx.OrthographicCamera()
    camera.show_object(scene, view_dir=view_dir, up=up)

    canvas = RenderCanvas(size=(64, 64), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    renderer.pixel_scale = 1
    renderer.ppaa = "none"
    canvas.request_draw(lambda: renderer.render(scene, camera))
    canvas.draw()

    info = renderer.get_pick_info((32, 32))
    assert info.get("world_object") is node, f"{name}: the centre pixel missed"

    coord = memory_image_data_coordinate(info, 3)
    assert coord is not None
    # pygfx (x, y, z) reverses onto data (z, y, x).
    index = [int(np.floor(value)) for value in tuple(coord)[::-1]]
    assert all(CUBE_LO <= value <= CUBE_HI for value in index), (
        f"{name} / {render_mode}: pick floored to {index}, outside the cube "
        f"[{CUBE_LO}, {CUBE_HI}]"
    )
