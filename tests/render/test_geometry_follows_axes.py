"""A multiscale visual's geometry follows the displayed axes.

``plans/displayed_axes_required.md`` Phase 1.0.  The geometry of a multiscale
image or labels visual holds the fetch axes and the per-level scale and
translation vectors of the axes it was built for.  It used to be rebuilt only
when the level shapes changed, so a change to a different displayed pair kept
the vectors of the old one.

Each case changes the displayed axes of a scene and compares the geometry
with a scene built directly at the target axes.  The pyramid is anisotropic
(axis 2 is never downsampled), so the pairs ``(0, 1)`` and ``(1, 2)`` have
different scale vectors, and it is not cubic, so they have different shapes
too at every coarse level.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.visuals import MultiscaleImageSingleAppearance
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
)
from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render.conftest import _write_multiscale_zarr

#: The world axes of ``add_scene``'s default coordinate system.
AXES = ("z", "y", "x")

# Axis 2 is never downsampled; axes 0 and 1 are.
SHAPES = [(32, 32, 32), (16, 16, 32), (8, 8, 32)]
SCALES = [(1.0, 1.0, 1.0), (2.0, 2.0, 1.0), (4.0, 4.0, 1.0)]

#: The level drawn by the pixel checks (1-based): the first coarse level,
#: where the pairs ``(0, 1)`` and ``(1, 2)`` have different scale vectors.
FORCED_LEVEL = 2


def _fill(array: np.ndarray) -> None:
    """A block that touches index 0 of every axis and is not symmetric."""
    d, h, w = array.shape
    array[: d // 2, : h // 4, w // 2 :] = 1


def _store(root, dtype: str) -> MultiscaleZarrDataStore:
    _write_multiscale_zarr(
        root,
        levels=[(f"s{i}", shape) for i, shape in enumerate(SHAPES)],
        fill=_fill,
        dtype=dtype,
    )
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(root),
        scale_names=[f"s{i}" for i in range(len(SHAPES))],
        level_scales=SCALES,
        level_translations=[tuple((s - 1.0) / 2.0 for s in scale) for scale in SCALES],
        name="anisotropic",
    )


def _add(controller, kind: str, scene, store, force_level: int | None = None):
    if kind == "image":
        return controller.add_image_multiscale(
            data=store,
            scene_id=scene.id,
            appearance=MultiscaleImageAppearance(
                force_level=force_level, interpolation="nearest"
            ),
            single=MultiscaleImageSingleAppearance(color_map="gray", clim=(0.0, 1.0)),
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=8),
        )
    return controller.add_labels_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(force_level=force_level),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS),
    )


def _gfx(controller, scene, visual):
    return controller._render_manager._scenes[scene.id].get_visual(visual.id)


def _signature(gfx_visual) -> dict:
    geometry = gfx_visual._image_geometry_2d
    assert geometry is not None, "the 2D geometry was not built"
    return {
        "fetch_axes": tuple(geometry.fetch_axes),
        "level_shapes": [tuple(s) for s in geometry.level_shapes],
        "scale": [tuple(map(float, v)) for v in geometry._scale_vecs_data],
        "translation": [tuple(map(float, v)) for v in geometry._translation_vecs_data],
    }


def _scene_at(controller, kind, store, start, name, force_level=None):
    """A scene first shown at *start*, with one visual and one canvas."""
    scene = controller.add_scene(
        displayed_axes=tuple(AXES[axis] for axis in start), name=name
    )
    visual = _add(controller, kind, scene, store, force_level)
    controller.add_canvas(scene_id=scene.id)
    return scene, visual


#: ``(id, the displayed axes in the order the scene shows them)``.  The last
#: entry is the target; the geometry there must match a scene built at it.
CASES = [
    ("2d_01_to_2d_12", [(0, 1), (1, 2)]),
    ("2d_12_to_2d_01", [(1, 2), (0, 1)]),
    ("2d_01_to_3d_to_2d_12", [(0, 1), (0, 1, 2), (1, 2)]),
    ("3d_to_2d_01", [(0, 1, 2), (0, 1)]),
    ("3d_to_2d_12", [(0, 1, 2), (1, 2)]),
]


@pytest.mark.parametrize("kind", ["image", "labels"])
@pytest.mark.parametrize("path", [c[1] for c in CASES], ids=[c[0] for c in CASES])
def test_geometry_matches_a_scene_built_at_the_target(controller, tmp_path, kind, path):
    store = _store(tmp_path, "float32" if kind == "image" else "int32")
    scene, visual = _scene_at(controller, kind, store, path[0], "changed")
    for axes in path[1:]:
        controller.set_displayed_axes(scene.id, axes)

    fresh_scene, fresh_visual = _scene_at(controller, kind, store, path[-1], "fresh")

    assert _signature(_gfx(controller, scene, visual)) == _signature(
        _gfx(controller, fresh_scene, fresh_visual)
    )


def test_the_pairs_differ_on_this_pyramid(controller, tmp_path):
    """The premise: ``(0, 1)`` and ``(1, 2)`` have different geometry here."""
    store = _store(tmp_path, "float32")
    first_scene, first = _scene_at(controller, "image", store, (0, 1), "a")
    second_scene, second = _scene_at(controller, "image", store, (1, 2), "b")
    a = _signature(_gfx(controller, first_scene, first))
    b = _signature(_gfx(controller, second_scene, second))
    assert a["scale"][1] == (2.0, 2.0)
    assert b["scale"][1] == (2.0, 1.0)
    assert a["level_shapes"] != b["level_shapes"]


@pytest.mark.parametrize("kind", ["image", "labels"])
def test_a_rebuilt_node_keeps_its_draw_state(controller, tmp_path, kind):
    """The rebuild makes a new node; it must not reset what the user set."""
    store = _store(tmp_path, "float32" if kind == "image" else "int32")
    scene, visual = _scene_at(controller, kind, store, (0, 1), "changed")
    # The unified image keeps its opacity on the per-channel appearance.
    (visual.single if kind == "image" else visual.appearance).opacity = 0.25
    visual.appearance.render_order = 3
    visual.pick_write = False
    gfx_visual = _gfx(controller, scene, visual)
    before = gfx_visual.node_2d

    controller.set_displayed_axes(scene.id, (1, 2))

    assert gfx_visual.material_2d.opacity == pytest.approx(0.25)
    assert gfx_visual.material_2d.pick_write is False
    if kind == "labels":
        # The unified image keeps one outer group; its slot nodes are rebuilt.
        assert gfx_visual.node_2d is not before
    assert gfx_visual.node_2d.render_order == 3


@pytest.mark.parametrize("kind", ["image", "labels"])
async def test_pixels_match_a_scene_built_at_the_target(
    controller, tmp_path, render_scene, reslice, kind
):
    """A coarse level drawn after a change of pair, against a fresh scene."""
    store = _store(tmp_path, "float32" if kind == "image" else "int32")
    scene, _visual = _scene_at(
        controller, kind, store, (0, 1), "changed", force_level=FORCED_LEVEL
    )
    await reslice(controller, scene.id)
    controller.set_displayed_axes(scene.id, (1, 2))
    await reslice(controller, scene.id)
    changed = render_scene(controller, scene.id)

    fresh_scene, _fresh = _scene_at(
        controller, kind, store, (1, 2), "fresh", force_level=FORCED_LEVEL
    )
    await reslice(controller, fresh_scene.id)
    fresh = render_scene(controller, fresh_scene.id)

    assert fresh[..., 3].any(), "the reference frame is empty"
    np.testing.assert_array_equal(changed, fresh)
