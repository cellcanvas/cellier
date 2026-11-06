"""The MaskedImage's 3D surface (masked image design v2, Phase 4; 6.2, 6.4).

The pair is the synthetic example's (``tests/render/_masked.py``): the mask
is an ellipsoid's distance map and the value its bands of ``BAND_WIDTH``
voxels, so the band a surface shows is known from the threshold, and the
band a projection along the normal shows from the threshold, the thickness
and the placement.  Each band is one primary of a nearest colormap, read as
a pixel's brightest channel (Phong shading scales a colour and adds a grey
highlight).

The extent clamp (MI-P16) is checked here on a mask forced to a coarse level
that covers less than the finest; its control switches the clamp off
through the material, the switch the template variable has for this purpose
only.  The clamp under a camera drag is in ``test_masked_image_drag.py``.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from cellier.convenience.capture import _wait_for_slicer
from cellier.render.shaders._masked_volume_brick import (
    instantiate_by_text,
    value_access_source,
)
from cellier.render.visuals._masked_image import surface_interval
from cellier.visuals import MaskedImageAppearance, MaskedImageSingleAppearance
from tests.render._affine import render
from tests.render._masked import (
    band_colormap,
    band_counts,
    band_pair,
    gfx_visual,
    load,
    masked_config,
    new_viewer,
    pyramid_store,
    store,
    value_to_mask_matrix,
)

SIDE = 64
SIZE = 256


def _band_viewer(tmp_path, kind, value_scale=1, **single):
    """A 3D viewer of the band pair, loaded; ``(viewer, model)``."""
    mask_data, value_data = band_pair(SIDE, value_scale)
    viewer = new_viewer("3d")
    mask = store(kind, tmp_path, "mask", mask_data)
    value = store(kind, tmp_path, "value", value_data)
    # An in-memory value at twice the resolution is 512 bricks of 16.
    value_budget = 48 if kind == "memory" and value_scale == 2 else 16
    model = viewer.controller.add_masked_image(
        mask,
        value,
        viewer.scene.id,
        value_to_mask=value_to_mask_matrix(value_scale),
        single=MaskedImageSingleAppearance(
            color_map=band_colormap(), clim=(0.5, 3.5), **single
        ),
        render_config=masked_config(value_gpu_budget_bytes=value_budget * 1024**2),
    )
    load(viewer, size=(SIZE, SIZE))
    return viewer, model


#: ``(iso, band)``: the surface at ``iso`` lies in ``band``.
SURFACE_BANDS = [(4.0, 1), (12.0, 2)]

#: ``(op, placement, iso, thickness, bands)`` of the projection cases.
PROJECTIONS = [
    # Inward from iso 4 over 8 voxels: distances 4 to 12, bands 1 and 2.
    ("max", "inward", 4.0, 8.0, {2}),
    ("min", "inward", 4.0, 8.0, {1}),
    ("mean", "inward", 4.0, 8.0, {1, 2}),
    # Outward from iso 4 over 3: distances 1 to 4, band 1 only.
    ("max", "outward", 4.0, 3.0, {1}),
    # Outward from iso 13 over 8: distances 5 to 13.
    ("min", "outward", 13.0, 8.0, {1}),
    ("max", "outward", 13.0, 8.0, {2}),
    # Centred on iso 13 over 8: distances 9 to 17, band 2 at least.
    ("min", "centred", 13.0, 8.0, {2}),
]


#: The surface fields each case sets, unnamed ones at their defaults.
SURFACE_FIELDS = (
    "iso_threshold",
    "surface_thickness",
    "surface_projection",
    "surface_placement",
)


def _projection(op, placement, iso, thickness) -> dict:
    return {
        "iso_threshold": iso,
        "surface_thickness": thickness,
        "surface_projection": op,
        "surface_placement": placement,
    }


def _surface_cases(kind: str, value_scale: int) -> list[dict]:
    """The single appearances drawn on one store kind and value grid."""
    cases = [{"iso_threshold": iso} for iso, _band in SURFACE_BANDS]
    if (kind, value_scale) == ("memory", 1):
        cases += [_projection(*case[:4]) for case in PROJECTIONS]
    return cases


def _key(single: dict) -> tuple:
    return tuple(sorted(single.items()))


def _surface_counts(root, kind: str, value_scale: int) -> dict[tuple, np.ndarray]:
    """Band counts of every case of :func:`_surface_cases`, from one viewer.

    Each case sets every surface field (the unnamed ones at their defaults)
    on the live model; ``test_a_projection_follows_the_thickness_when_it_changes``
    is the check that a changed appearance reaches the material.
    """
    defaults = MaskedImageSingleAppearance.model_fields
    viewer, model = _band_viewer(root, kind, value_scale)
    out = {}
    try:
        for single in _surface_cases(kind, value_scale):
            for name in SURFACE_FIELDS:
                setattr(model.single, name, single.get(name, defaults[name].default))
            asyncio.run(_wait_for_slicer(viewer.controller))
            out[_key(single)] = band_counts(render(viewer, SIZE)[0])
    finally:
        viewer.controller.close()
    return out


@pytest.fixture(scope="module")
def surface_counts(tmp_path_factory):
    """``surface_counts(kind, value_scale, **single)``: a frame's band counts.

    The cases differ only in the single appearance, so each store kind and
    value grid is built once and all its cases drawn then (the autouse
    teardown closes every viewer at the end of a test, so none outlives it).
    """
    cache: dict[tuple[str, int], dict[tuple, np.ndarray]] = {}

    def get(kind: str, value_scale: int = 1, **single) -> np.ndarray:
        if (kind, value_scale) not in cache:
            root = tmp_path_factory.mktemp(f"surface_{kind}_{value_scale}")
            cache[kind, value_scale] = _surface_counts(root, kind, value_scale)
        return cache[kind, value_scale][_key(single)]

    return get


# -- the value at the surface (6.2) ----------------------------------------------


@pytest.mark.parametrize("kind", ["memory", "lazy"])
@pytest.mark.parametrize("value_scale", [1, 2])
@pytest.mark.parametrize(("iso", "band"), SURFACE_BANDS)
def test_the_surface_shows_its_band(surface_counts, kind, value_scale, iso, band):
    """Every drawn pixel of the surface at ``iso`` is the band ``iso`` lies in.

    On one grid and on a value at twice the mask's resolution, its own grid.
    """
    counts = surface_counts(kind, value_scale, iso_threshold=iso)
    assert counts.sum() > 500, counts
    assert counts[band - 1] == counts.sum(), counts


# -- the projection along the normal (6.2) ---------------------------------------


@pytest.mark.parametrize(("op", "placement", "iso", "thickness", "bands"), PROJECTIONS)
def test_the_projection_reduces_along_the_inward_normal(
    surface_counts, op, placement, iso, thickness, bands
):
    """The max, min or mean over the placed interval gives the bands it spans.

    The cases pin the sign of the normal (a max inward is deeper, a max
    outward is not) and that the thickness is in world units.
    """
    counts = surface_counts("memory", **_projection(op, placement, iso, thickness))
    assert counts.sum() > 500, counts
    shown = {band + 1 for band in range(3) if counts[band] > 0}
    assert shown == bands, counts


def test_a_projection_follows_the_thickness_when_it_changes(tmp_path):
    """The single appearance reaches the material: a max deepens as T grows."""
    viewer, model = _band_viewer(tmp_path, "lazy", iso_threshold=4.0)
    try:
        assert band_counts(render(viewer, SIZE)[0])[1] == 0
        model.single.surface_thickness = 8.0
        counts = band_counts(render(viewer, SIZE)[0])
        assert counts[1] == counts.sum() > 0, counts
        model.single.surface_projection = "min"
        counts = band_counts(render(viewer, SIZE)[0])
        assert counts[0] == counts.sum() > 0, counts
    finally:
        viewer.controller.close()


@pytest.mark.parametrize(
    ("placement", "expected"),
    [("inward", (0.0, 6.0)), ("centred", (-3.0, 3.0)), ("outward", (-6.0, 0.0))],
)
def test_surface_interval(placement, expected):
    assert surface_interval(6.0, placement) == expected


# -- the value side's copy of the lookup -------------------------------------------


def test_the_value_lookup_is_the_mask_lookup_renamed():
    """No mask binding or accessor survives; the masked clamp is not copied."""
    text = instantiate_by_text(value_access_source())
    code = "\n".join(line.split("//")[0] for line in text.splitlines())
    for name in ("t_cache", "s_cache", "u_vol_params", "u_block_scales"):
        assert f" {name}" not in code and f"({name}" not in code, name
    assert "fn value_sample_atlas(" in code
    assert "mask_extent_clamp" not in code
    assert "$$" not in code


def test_instantiate_by_text_refuses_a_missing_accessor():
    with pytest.raises(ValueError, match="not found"):
        instantiate_by_text("fn get_lod_scale() {}")


# -- the extent clamp (MI-P16, 6.4) -----------------------------------------------


def _strip_viewer(tmp_path, clamp: bool):
    """A mask inside everywhere, 39 voxels on x, forced to its coarsest level.

    Floor halved twice, the coarsest level covers 36 of the 39 voxels on x.
    The value is 1 from x = 37 on, 0 before.
    """
    from cellier.data import ImageMemoryStore

    shape = (24, 24, 39)
    mask = pyramid_store(tmp_path, "mask", np.full(shape, 10.0, np.float32), 3)
    value_data = np.zeros(shape, np.float32)
    value_data[..., 37:] = 1.0
    value = ImageMemoryStore(data=value_data, name="value")
    viewer = new_viewer("3d")
    import cmap

    model = viewer.controller.add_masked_image(
        mask,
        value,
        viewer.scene.id,
        appearance=MaskedImageAppearance(force_level=3),
        single=MaskedImageSingleAppearance(
            color_map=cmap.Colormap(["red", "lime"], interpolation="nearest"),
            clim=(0.0, 1.0),
            iso_threshold=5.0,
        ),
        render_config=masked_config(),
    )
    gfx_visual(viewer, model.id)._slots[0].material_3d.extent_clamp = clamp
    load(viewer, size=(SIZE, SIZE))
    view = viewer.controller._render_manager._canvases[
        viewer.controller.get_canvas_ids(viewer.scene.id)[0]
    ]
    view.camera.show_object(
        view._get_scene_fn(viewer.scene.id), view_dir=(-1, 0, 0), up=(0, 0, 1)
    )
    return viewer


@pytest.mark.parametrize("clamp", [True, False])
def test_a_read_past_a_coarse_level_end_repeats_its_last_voxel(tmp_path, clamp):
    """Seen from +x, the face lies in the missing strip of the coarsest level.

    With the clamp the strip repeats the level's last voxel, so the surface
    is the box's face at x = 38.5, where the value is 1 (lime).  Without it
    the strip reads the brick's zero padding, the surface moves in to about
    x = 35.5, where the value is 0 (red): the control.
    """
    viewer = _strip_viewer(tmp_path, clamp)
    try:
        frame = render(viewer, SIZE)[0]
        rgb = frame[..., :3].astype(np.int64)
        drawn = rgb.max(axis=-1) > 8
        lime = int((np.argmax(rgb, axis=-1)[drawn] == 1).sum())
        if clamp:
            assert drawn.sum() > 10_000
            assert lime == drawn.sum()
        else:
            assert lime < 0.01 * drawn.sum()
    finally:
        viewer.controller.close()
