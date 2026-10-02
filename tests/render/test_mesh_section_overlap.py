"""Which mesh is on top where two sections overlap in a 2D view.

Every mesh's cut lies in the slice plane, so their fragments tie on depth.
A section is drawn with ``depth_compare="<="``: the last drawn wins the tie,
so the order is ``render_order``, then the order the meshes were added, and
an outline is over every fill of the same ``render_order``.  Frames are real
frames of an offscreen canvas: a small sphere inside a large one, cut
through their common centre.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.mesh import MeshMemoryStore
from cellier.render.visuals._mesh import SECTION_DEPTH_COMPARE
from cellier.visuals import MeshFlatAppearance, MeshSectionConfig
from tests._meshes import uv_sphere
from tests.render.test_mesh_section import Rig2D, _load

CENTRE = (16.0, 16.0, 16.0)
RED = (1.0, 0.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)
OUTLINE_ONLY = MeshSectionConfig(fill=False, outline_width=3.0)


@pytest.fixture
def rig():
    rig = Rig2D()
    yield rig
    rig.controller.close()


def _add(rig, name: str, section=None, **appearance):
    radius, color = (10.0, BLUE) if name == "big" else (4.0, RED)
    positions, indices = uv_sphere(radius, CENTRE, n_lat=24, n_lon=48)
    return rig.controller.add_mesh(
        data=MeshMemoryStore(positions=positions, indices=indices, name=name),
        scene_id=rig.scene.id,
        appearance=MeshFlatAppearance(color=color, **appearance),
        name=name,
        section=section,
    )


def _frame(rig) -> np.ndarray:
    return np.asarray(rig.view._canvas.draw())


def _red(frame: np.ndarray) -> int:
    return int(((frame[..., 0] > 150) & (frame[..., 2] < 90)).sum())


def _blue(frame: np.ndarray) -> int:
    return int(((frame[..., 2] > 150) & (frame[..., 0] < 90)).sum())


def _centre(frame: np.ndarray) -> str:
    """The colour at the middle of the canvas, inside both spheres."""
    r, _g, b = frame[frame.shape[0] // 2, frame.shape[1] // 2, :3]
    return "red" if r > 150 and b < 90 else "blue" if b > 150 and r < 90 else "none"


async def test_the_mesh_added_later_is_on_top(rig):
    _add(rig, "big")
    _add(rig, "small")
    await _load(rig)

    assert _centre(_frame(rig)) == "red"


async def test_the_mesh_added_first_is_underneath(rig):
    _add(rig, "small")
    _add(rig, "big")
    await _load(rig)

    frame = _frame(rig)
    assert _centre(frame) == "blue"
    # The small mesh's outline is drawn after every fill, so it still shows.
    assert _red(frame) > 0


@pytest.mark.parametrize("first", ["big", "small"])
async def test_an_outline_is_over_a_fill_in_either_add_order(rig, first):
    for name in (first, "small" if first == "big" else "big"):
        _add(rig, name, section=OUTLINE_ONLY if name == "small" else None)
    await _load(rig)

    frame = _frame(rig)
    assert _red(frame) > 0
    # Only the outline: the big mesh's fill shows inside it.
    assert _centre(frame) == "blue"


@pytest.mark.parametrize("first", ["big", "small"])
async def test_a_higher_render_order_is_on_top_in_either_add_order(rig, first):
    for name in (first, "small" if first == "big" else "big"):
        _add(rig, name, render_order=5 if name == "small" else 0)
    await _load(rig)

    assert _centre(_frame(rig)) == "red"


async def test_changing_render_order_changes_which_is_on_top(rig):
    big = _add(rig, "big")
    _add(rig, "small")
    await _load(rig)
    assert _centre(_frame(rig)) == "red"

    rig.controller.update_appearance_field(big.id, "render_order", 5)

    frame = _frame(rig)
    assert _centre(frame) == "blue"
    # Its fill is now over the small mesh's outline too.
    assert _red(frame) == 0


async def test_depth_compare_on_the_appearance_is_for_the_3d_surface(rig):
    mesh = _add(rig, "big", depth_compare=">")
    gfx = rig.gfx(mesh)
    assert gfx._material_3d.depth_compare == ">"
    assert gfx._material_2d.depth_compare == SECTION_DEPTH_COMPARE == "<="
    assert gfx._material_outline.depth_compare == "<="

    rig.controller.update_appearance_field(mesh.id, "depth_compare", "<")

    assert gfx._material_3d.depth_compare == "<"
    assert gfx._material_2d.depth_compare == "<="
    assert gfx._material_outline.depth_compare == "<="
    # depth_test still switches the section's test off.
    rig.controller.update_appearance_field(mesh.id, "depth_test", False)
    assert gfx._material_2d.depth_test is False
    assert gfx._material_outline.depth_test is False


@pytest.mark.parametrize("first", ["image", "mesh"])
async def test_a_section_is_over_an_image_in_either_add_order(rig, first):
    def image():
        rig.controller.add_image(
            ImageMemoryStore(data=np.full((32, 32, 32), 0.5, dtype=np.float32)),
            rig.scene.id,
        )

    for add in (
        (image, lambda: _add(rig, "big"))
        if first == "image"
        else (
            lambda: _add(rig, "big"),
            image,
        )
    ):
        add()
    await _load(rig)

    frame = _frame(rig)
    assert _centre(frame) == "blue"
    assert _blue(frame) > 1000
