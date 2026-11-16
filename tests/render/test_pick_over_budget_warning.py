"""A volume too large for a pick to name every voxel gives one warning.

The pick position has 44 bits (`cellier.render._pick_layout`).  A volume
whose index bits add up to more, or with an axis longer than ``2 ** 24``
voxels, resolves to steps of several voxels on some axes, and the render layer
says so with a ``UserWarning``, once per visual and 3D shape, wherever the
layout is written.

The visual tests lower the budget so a small test volume is over it: a volume
over the real 44 bits is far larger than a test can load.
"""

from __future__ import annotations

import warnings
from uuid import uuid4

import numpy as np
import pytest

from cellier.convenience import Viewer
from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.render import _pick_layout
from cellier.render._pick_layout import (
    over_budget_message,
    pick_bit_layout,
    warn_if_pick_over_budget,
)
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    MultiscaleImageAppearance,
    MultiscaleImageChannelAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render._affine import P0, bind, block_volume, load, multiscale_store
from tests.render.conftest import _write_multiscale_zarr
from tests.render.test_labels_pick_face import _pick_every_view, _viewer
from tests.render.test_pick_layout_after_resize import _store as _series_store

DISPLAYED = ("z", "y", "x")

# -- the guard ------------------------------------------------------------------


def test_a_volume_inside_the_budget_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        layout = warn_if_pick_over_budget((2048, 10000, 40000), uuid4(), "lightsheet")
    assert layout == pick_bit_layout((2048, 10000, 40000))


def test_the_warning_is_given_once_per_visual_and_shape():
    visual_id = uuid4()
    with pytest.warns(UserWarning) as record:
        warn_if_pick_over_budget((32640, 32640, 32640), visual_id, "cells")
    assert [str(w.message) for w in record] == [
        "3D picks on visual 'cells' resolve to 2 voxels along x: its 32640 x "
        "32640 x 32640 voxels (x, y, z) need 45 bits to address and the pick "
        "buffer has 44.  A pick's data_coordinate and value can be off by up "
        "to 1 voxel along x."
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # The same visual and shape again: another slot, or a view that
        # came back to a shape already warned about.
        warn_if_pick_over_budget((32640, 32640, 32640), visual_id, "cells")
        warn_if_pick_over_budget(np.array([32640.0] * 3), visual_id, "cells")
    with pytest.warns(UserWarning, match="32641 x 32640"):
        warn_if_pick_over_budget((32641, 32640, 32640), visual_id, "cells")
    with pytest.warns(UserWarning, match="'other'"):
        warn_if_pick_over_budget((32640, 32640, 32640), uuid4(), "other")


@pytest.mark.parametrize(
    ("size", "text"),
    [
        (
            (2**24 + 1, 1, 1),
            "3D picks on visual 'v' resolve to 2 voxels along x: its 16777217 "
            "voxels along x need 25 bits to address and one axis can have at "
            "most 24.  A pick's data_coordinate and value can be off by up to "
            "1 voxel along x.",
        ),
        (
            (2**20, 2**20, 2**20),
            "3D picks on visual 'v' resolve to 64 voxels along x, 32 voxels "
            "along y and 32 voxels along z: its 1048576 x 1048576 x 1048576 "
            "voxels (x, y, z) need 60 bits to address and the pick buffer has "
            "44.  A pick's data_coordinate and value can be off by up to 63 "
            "voxels along x, 31 voxels along y and 31 voxels along z.",
        ),
    ],
)
def test_the_message_names_every_coarse_axis(size, text):
    assert over_budget_message(pick_bit_layout(size), "v") == text


# -- the visuals ----------------------------------------------------------------


@pytest.fixture
def small_budget(monkeypatch):
    """15 position bits: a 40 x 32 x 24 volume needs 16."""
    monkeypatch.setattr(_pick_layout, "PICK_POSITION_BITS", 15)


def _over_budget_warnings(record) -> list[str]:
    return [str(w.message) for w in record if "3D picks on visual" in str(w.message)]


@pytest.mark.parametrize("family", ["image", "labels"])
def test_adding_an_over_budget_volume_warns_once(tmp_path, small_budget, family):
    viewer = _viewer(("z", "y", "x"))
    try:
        data = np.zeros((24, 32, 40), np.int32 if family == "labels" else np.float32)
        store = multiscale_store(tmp_path, "big", data)
        transform = bind(viewer, store, np.eye(4))
        with pytest.warns(UserWarning) as record:
            if family == "labels":
                viewer.add_labels_multiscale(
                    store,
                    name="big",
                    render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS),
                    transform=transform,
                )
            else:
                viewer.add_image_multiscale(
                    store,
                    name="big",
                    render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS),
                    transform=transform,
                )
            load(viewer)
        assert _over_budget_warnings(record) == [
            "3D picks on visual 'big' resolve to 2 voxels along x: its 40 x 32 x "
            "24 voxels (x, y, z) need 16 bits to address and the pick buffer "
            "has 15.  A pick's data_coordinate and value can be off by up to 1 "
            "voxel along x."
        ]
    finally:
        viewer.controller.close()


def test_a_multichannel_image_warns_once_for_all_its_slots(tmp_path, small_budget):
    _write_multiscale_zarr(
        tmp_path,
        levels=[("s0", (3, 24, 32, 40)), ("s1", (3, 12, 16, 20))],
        fill=lambda arr: arr.fill(0.0),
    )
    store = MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(tmp_path),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0, 1.0, 1.0), (1.0, 2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0, 0.0), (0.0, 0.5, 0.5, 0.5)],
    )
    viewer = Viewer(
        [("c", "channel"), *spatial_axes("z", "y", "x")],
        displayed_axes=("z", "y", "x"),
        gui="offscreen",
    )
    try:
        transform = bind(viewer, store, np.eye(5))
        with pytest.warns(UserWarning) as record:
            visual = viewer.add_image_multiscale(
                store,
                name="channels",
                channel_axis=0,
                composite=True,
                channels={i: MultiscaleImageChannelAppearance() for i in range(3)},
                render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS),
                transform=transform,
            )
            load(viewer)
        gfx_visual = viewer.controller._render_manager._scenes[
            viewer.scene.id
        ].get_visual(visual.id)
        assert sum(slot._dataset_size is not None for slot in gfx_visual._slots) >= 3
        assert len(_over_budget_warnings(record)) == 1
        assert "'channels'" in _over_budget_warnings(record)[0]
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("family", ["image", "labels"])
def test_a_shape_seen_before_does_not_warn_again(tmp_path, small_budget, family):
    """Switching the 3D view between shapes warns only on a new one."""
    viewer = Viewer(
        [("t", "time"), *spatial_axes("z", "y", "x")],
        displayed_axes=("z", "y", "x"),
        gui="offscreen",
    )
    try:
        store = _series_store(tmp_path, np.int32 if family == "labels" else np.float32)
        transform = bind(viewer, store, np.eye(5))
        add = (
            viewer.add_labels_multiscale
            if family == "labels"
            else viewer.add_image_multiscale
        )
        config = (
            MultiscaleLabelRenderConfig
            if family == "labels"
            else MultiscaleImageRenderConfig
        )(**SMALL_BUDGETS, block_size=16)
        with pytest.warns(UserWarning) as record:
            add(store, name="series", render_config=config, transform=transform)
            load(viewer)
        # (z, y, x): 40 x 32 x 24 needs 16 bits.
        assert len(_over_budget_warnings(record)) == 1
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # (t, y, x): 40 x 32 x 12 needs 15 bits, inside the budget.
            viewer.scene.dims.selection.displayed_axes = (0, 2, 3)
            load(viewer)
            # Back to (z, y, x): warned about already.
            viewer.scene.dims.selection.displayed_axes = (1, 2, 3)
            load(viewer)
    finally:
        viewer.controller.close()


def test_a_volume_built_on_entering_3d_warns_then(tmp_path, small_budget):
    """A scene that starts in 2D builds its volume, and warns, on the toggle."""
    viewer = _viewer(("y", "x"))
    try:
        store = multiscale_store(tmp_path, "big", np.zeros((24, 32, 40), np.float32))
        transform = bind(viewer, store, np.eye(4))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            viewer.add_image_multiscale(
                store,
                name="big",
                render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS),
                transform=transform,
            )
            load(viewer)
        with pytest.warns(UserWarning, match="visual 'big' resolve to 2 voxels"):
            viewer.scene.dims.selection.displayed_axes = (0, 1, 2)
            load(viewer)
    finally:
        viewer.controller.close()


# -- an over-budget volume still picks within one step ------------------------


@pytest.mark.parametrize("family", ["image", "labels"])
def test_over_budget_picks_land_within_one_step(tmp_path, monkeypatch, family):
    """Each axis's pick is within one step of the block, and exact if 1 voxel.

    With 13 bits for a 40 x 32 x 24 volume (16 needed), the steps are 4, 2 and
    1 voxels on x, y and z.  A pick decodes to the centre of the step that holds
    the voxel hit, so its voxel is at most ``step - 1`` from a block voxel.
    """
    monkeypatch.setattr(_pick_layout, "PICK_POSITION_BITS", 13)
    layout = pick_bit_layout((40, 32, 24))
    assert layout.step == (4.0, 2.0, 1.0)
    viewer = _viewer(DISPLAYED)
    try:
        dtype = np.int32 if family == "labels" else np.float32
        store = multiscale_store(tmp_path, "coarse", block_volume(dtype))
        transform = bind(viewer, store, np.eye(4))
        with pytest.warns(UserWarning, match="4 voxels along x and 2 voxels along y"):
            if family == "labels":
                visual = viewer.add_labels_multiscale(
                    store,
                    appearance=MultiscaleLabelsAppearance(
                        force_level=0, ray_steps_per_voxel=4.0
                    ),
                    render_config=MultiscaleLabelRenderConfig(
                        **SMALL_BUDGETS, block_size=16
                    ),
                    transform=transform,
                )
            else:
                visual = viewer.add_image_multiscale(
                    store,
                    appearance=MultiscaleImageAppearance(force_level=0),
                    render_config=MultiscaleImageRenderConfig(
                        **SMALL_BUDGETS, block_size=16
                    ),
                    single=MultiscaleImageSingleAppearance(
                        render_mode="iso", clim=(0.0, 1.0), iso_threshold=0.5
                    ),
                    transform=transform,
                )
        load(viewer)
        # Data axes (z, y, x) are render axes (z, y, x) reversed.
        slack = np.array(layout.step[::-1]) - 1.0
        low, high = P0 - 1.0, P0 + 1.0
        off_by = []

        def check(_direction, _level, full):
            voxel = np.floor(np.asarray(full))
            distance = np.maximum(np.maximum(low - voxel, voxel - high), 0.0)
            off_by.append(distance)
            assert (distance <= slack).all(), (full, distance)

        n_picks = _pick_every_view(viewer, visual, DISPLAYED, check)
        assert n_picks >= 100
        # The coarse axes are coarse: some picks land outside the block.
        assert np.asarray(off_by)[:, 2].max() >= 1.0
    finally:
        viewer.controller.close()
