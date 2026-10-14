"""The smooth 3D labels modes, ``smooth_iso`` and ``smooth_geometry``.

Both read ``cellier.label_smooth.wgsl``.  ``smooth_iso`` keeps the voxel-face
surface of ``iso_categorical`` and only shades it with a normal blended across
the voxels; ``smooth_geometry`` draws the surface of each label's trilinear
field instead, so convex corners round off, concave ones fill in, and a hit
can sit in a background voxel.  Every test runs on the in-memory and the
multiscale visual, which find the same surface by different walks (one over
the whole volume, one brick by brick in each brick's own cell space).
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorstore as ts

from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.data.label._label_memory_store import LabelMemoryStore
from cellier.visuals._label_memory import InMemoryLabelsAppearance
from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render._planes import shoot

N = 24
LABEL = 5
KINDS = ["memory", "multiscale"]


def _cube() -> np.ndarray:
    data = np.zeros((N, N, N), dtype=np.int32)
    data[8:16, 8:16, 8:16] = LABEL
    return data


def _sphere() -> np.ndarray:
    z, y, x = np.mgrid[:N, :N, :N]
    data = np.zeros((N, N, N), dtype=np.int32)
    data[(z - 11.5) ** 2 + (y - 11.5) ** 2 + (x - 11.5) ** 2 <= 8.0**2] = LABEL
    return data


def _cross() -> np.ndarray:
    """Three orthogonal bars: concave edges and corners all round the centre."""
    data = np.zeros((N, N, N), dtype=np.int32)
    data[4:20, 10:14, 10:14] = LABEL
    data[10:14, 4:20, 10:14] = LABEL
    data[10:14, 10:14, 4:20] = LABEL
    return data


def _single_voxel() -> np.ndarray:
    data = np.zeros((N, N, N), dtype=np.int32)
    data[12, 12, 12] = LABEL
    return data


def _store(root, data: np.ndarray) -> MultiscaleZarrDataStore:
    """A one-level store, in 8-voxel bricks so surfaces cross brick faces."""
    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(root / "s0")},
        "metadata": {
            "shape": list(data.shape),
            "data_type": "int32",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": list(data.shape)},
            },
        },
        "create": True,
        "delete_existing": True,
    }
    ts.open(spec).result()[...].write(np.ascontiguousarray(data)).result()
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(root),
        scale_names=["s0"],
        level_scales=[(1.0, 1.0, 1.0)],
        level_translations=[(0.0, 0.0, 0.0)],
        name="labels-smooth",
    )


async def _labels(controller, reslice, root, kind, data, render_mode):
    """Add *data* as a labels visual in its own scene; return (scene, visual)."""
    scene = controller.add_scene(displayed_axes=("z", "y", "x"), name="smooth")
    if kind == "memory":
        visual = controller.add_labels(
            data=LabelMemoryStore(data=data, name="labels"),
            scene_id=scene.id,
            appearance=InMemoryLabelsAppearance(render_mode=render_mode),
        )
    else:
        root.mkdir(parents=True, exist_ok=True)
        visual = controller.add_labels_multiscale(
            data=_store(root, data),
            scene_id=scene.id,
            appearance=MultiscaleLabelsAppearance(
                force_level=0, render_mode=render_mode
            ),
            render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=8),
        )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    return scene, visual


async def _shot(controller, reslice, root, kind, data, render_mode, **kwargs):
    scene, visual = await _labels(controller, reslice, root, kind, data, render_mode)
    return shoot(controller, scene, **kwargs), scene, visual


def _luminance(frame: np.ndarray) -> np.ndarray:
    return frame[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def _neighbour_jumps(frame: np.ndarray) -> np.ndarray:
    """|luminance step| between horizontally and vertically adjacent drawn pixels."""
    drawn = frame[..., 3] > 0
    lum = _luminance(frame)
    jumps = []
    for axis in (0, 1):
        a = np.take(lum, np.arange(lum.shape[axis] - 1), axis=axis)
        b = np.take(lum, np.arange(1, lum.shape[axis]), axis=axis)
        both = np.take(drawn, np.arange(drawn.shape[axis] - 1), axis=axis) & np.take(
            drawn, np.arange(1, drawn.shape[axis]), axis=axis
        )
        jumps.append(np.abs(a - b)[both])
    return np.concatenate(jumps)


@pytest.mark.parametrize("kind", KINDS)
async def test_smooth_iso_keeps_the_voxel_faces_and_smooths_their_shading(
    controller, reslice, tmp_path, kind
):
    """Same pixels drawn as ``iso_categorical``; no hard steps in the shading.

    A voxelised sphere shades in ``iso_categorical`` with a hard step at
    every edge between voxel faces.  ``smooth_iso`` draws the very same
    surface -- the same pixels -- with a normal that varies continuously, so
    the largest steps in its shading are far smaller.
    """
    view = {"size": (320, 320), "view_dir": (-0.6, -0.5, -1.0)}
    iso, *_ = await _shot(
        controller,
        reslice,
        tmp_path / "iso",
        kind,
        _sphere(),
        "iso_categorical",
        **view,
    )
    smooth, *_ = await _shot(
        controller, reslice, tmp_path / "smooth", kind, _sphere(), "smooth_iso", **view
    )

    assert iso.drawn.sum() > 5000, "the sphere did not draw"
    assert np.array_equal(iso.drawn, smooth.drawn), (
        "smooth_iso drew a different silhouette from iso_categorical"
    )
    iso_steps = np.percentile(_neighbour_jumps(iso.frame), 99)
    smooth_steps = np.percentile(_neighbour_jumps(smooth.frame), 99)
    assert smooth_steps < 0.5 * iso_steps, (
        f"smooth_iso still shades in steps: 99th percentile {smooth_steps:.1f} "
        f"against {iso_steps:.1f} for iso_categorical"
    )


@pytest.mark.parametrize("kind", KINDS)
async def test_smooth_geometry_rounds_convex_corners(
    controller, reslice, tmp_path, kind
):
    """Corner-on, a rounded cube covers fewer pixels than its voxel faces."""
    view = {"size": (256, 256), "view_dir": (-1.0, -1.0, -1.0)}
    iso, *_ = await _shot(
        controller, reslice, tmp_path / "iso", kind, _cube(), "iso_categorical", **view
    )
    smooth, *_ = await _shot(
        controller,
        reslice,
        tmp_path / "smooth",
        kind,
        _cube(),
        "smooth_geometry",
        **view,
    )

    assert iso.drawn.sum() > 2000, "the cube did not draw"
    # Every smooth pixel is inside the voxel silhouette (a convex shape only
    # loses volume), and the rounding takes off a visible amount.
    assert not (smooth.drawn & ~iso.drawn).any()
    ratio = smooth.drawn.sum() / iso.drawn.sum()
    assert 0.75 < ratio < 0.97, f"smooth/voxel coverage {ratio:.3f}"


@pytest.mark.parametrize("kind", KINDS)
async def test_smooth_geometry_keeps_flat_sides_on_the_voxel_faces(
    controller, reslice, tmp_path, kind
):
    """Face-on, the rounded cube's outline is the voxel cube's.

    The field is exactly 0.5 on a voxel face between a label and background,
    so the flat sides do not move; only the edges round off, and face-on
    they are hidden behind the widest cross-section.  Half a voxel of error
    in where the surface sits would change the area by about an eighth on
    this 8-voxel cube.  The near-orthographic camera keeps the rounded front
    edges from shrinking the outline in perspective.
    """
    view = {"size": (512, 512), "view_dir": (0.0, 0.0, -1.0), "fov": 1.0}
    iso, *_ = await _shot(
        controller, reslice, tmp_path / "iso", kind, _cube(), "iso_categorical", **view
    )
    smooth, *_ = await _shot(
        controller,
        reslice,
        tmp_path / "smooth",
        kind,
        _cube(),
        "smooth_geometry",
        **view,
    )

    assert iso.drawn.sum() > 4000, "the cube did not draw"
    differ = (smooth.drawn ^ iso.drawn).sum() / iso.drawn.sum()
    assert differ < 0.02, f"face-on outlines differ on {differ:.1%} of the cube"


@pytest.mark.parametrize("kind", KINDS)
async def test_smooth_geometry_draws_a_single_voxel(
    controller, reslice, tmp_path, kind
):
    """A voxel's own centre has weight 1, so even a lone voxel draws."""
    iso, *_ = await _shot(
        controller,
        reslice,
        tmp_path / "iso",
        kind,
        _single_voxel(),
        "iso_categorical",
        size=(256, 256),
        view_dir=(-0.6, -0.5, -1.0),
    )
    smooth, *_ = await _shot(
        controller,
        reslice,
        tmp_path / "smooth",
        kind,
        _single_voxel(),
        "smooth_geometry",
        size=(256, 256),
        view_dir=(-0.6, -0.5, -1.0),
    )

    assert iso.drawn.sum() > 0, "the voxel did not draw at all"
    assert 0 < smooth.drawn.sum() < iso.drawn.sum()


@pytest.mark.parametrize("kind", KINDS)
async def test_smooth_geometry_picks_name_the_drawn_label(
    controller, reslice, tmp_path, kind
):
    """Every drawn pixel picks a voxel that carries the label.

    Where the smooth surface fills in a concave corner of the cross it sits
    in a background voxel; the shader reports a point in the nearest voxel
    of the label instead, so the pick still names what was drawn.
    """
    data = _cross()
    shot, scene, visual = await _shot(
        controller,
        reslice,
        tmp_path,
        kind,
        data,
        "smooth_geometry",
        size=(160, 160),
        view_dir=(-1.0, -0.8, -0.6),
    )
    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)

    rows, cols = np.nonzero(shot.drawn)
    assert len(rows) > 1000, "the cross did not draw"
    wrong = []
    for row, col in zip(rows, cols, strict=True):
        info = shot.pick_info(int(row), int(col))
        coord = gfx_visual.pick_data_coordinate(info.get("world_object"), info)
        assert coord is not None, (row, col)
        # pygfx (x, y, z) reverses onto data (z, y, x); [i, i + 1) convention.
        index = tuple(int(np.floor(value)) for value in tuple(coord)[::-1])
        if data[index] != LABEL:
            wrong.append(((row, col), index))
    assert not wrong, f"{len(wrong)} picks outside the label, e.g. {wrong[:5]}"


async def test_smooth_geometry_agrees_between_in_memory_and_multiscale(
    controller, reslice, tmp_path
):
    """The two walks find the same surface.

    The multiscale shader walks each 8-voxel brick in that brick's own cell
    space, through its ghost border; the in-memory one walks the whole
    volume.  A slip in either mapping would move the surface by up to half a
    voxel, which on this cross is far more than the few pixels allowed.
    """
    view = {"size": (256, 256), "view_dir": (-1.0, -0.8, -0.6)}
    memory, *_ = await _shot(
        controller,
        reslice,
        tmp_path / "m",
        "memory",
        _cross(),
        "smooth_geometry",
        **view,
    )
    multiscale, *_ = await _shot(
        controller,
        reslice,
        tmp_path / "ms",
        "multiscale",
        _cross(),
        "smooth_geometry",
        **view,
    )

    assert memory.drawn.sum() > 3000, "the cross did not draw"
    differ = (memory.drawn ^ multiscale.drawn).sum()
    assert differ < 0.005 * memory.drawn.sum(), f"{differ} pixels differ"
    both = memory.drawn & multiscale.drawn
    colour_step = np.abs(
        _luminance(memory.frame)[both] - _luminance(multiscale.frame)[both]
    )
    assert np.percentile(colour_step, 99) < 8.0, "the shading differs"


@pytest.mark.parametrize("kind", KINDS)
async def test_the_smooth_geometry_level_grows_and_shrinks_the_surface(
    controller, reslice, tmp_path, kind
):
    """A lower level grows the label and a higher one shrinks it, live.

    Set through the controller after the visual is drawn, as a GUI slider
    does: the level is a uniform, so the change needs no rebuild.
    """
    scene, visual = await _labels(
        controller, reslice, tmp_path, kind, _cross(), "smooth_geometry"
    )
    view = {"size": (256, 256), "view_dir": (-1.0, -0.8, -0.6)}
    coverage = {}
    for level in (0.3, 0.5, 0.7):
        controller.update_appearance_field(visual.id, "smooth_geometry_level", level)
        coverage[level] = shoot(controller, scene, **view).drawn.sum()

    assert coverage[0.5] > 1000, "the cross did not draw"
    assert coverage[0.3] > coverage[0.5] > coverage[0.7], coverage
