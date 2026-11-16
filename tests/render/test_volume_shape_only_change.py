"""A change of the level shapes alone resizes the 3D volume, not only its grids.

``_sync_geometry_3d`` (the image slot and the multiscale labels visual) has two
branches. New fetch axes replace the geometry; the same fetch axes with new
level shapes (a store that grew or shrank) update the geometry in place. Both
must also refresh what is sized from the level-0 shape: the visual's
``_dataset_size`` and ``_norm_size``, the params buffer the shader reads, and
the pick layout of the node built from them. Otherwise a grown store would draw
in the old proxy box and its picks would clamp at the old far face.

No store changes shape today, so the branch is driven directly with a longer
level 0, as a store mutation would.
"""

from __future__ import annotations

import numpy as np
import pytest

import cellier.render.visuals._image as image_module
import cellier.render.visuals._label_multiscale as labels_module
from cellier.render._pick_layout import pick_bit_layout
from tests.render._affine import load
from tests.render.test_pick_large_volume import DISPLAYED, _add, _viewer, _volume

LENGTH = 256


def _render_target(viewer, family, visual):
    gfx_visual = viewer.controller._render_manager._scenes[viewer.scene.id].get_visual(
        visual.id
    )
    return gfx_visual._slots[0] if family == "image_multiscale" else gfx_visual


def _uniform_size(target) -> tuple[int, int, int]:
    data = target._vol_params_buffer.data
    return tuple(int(data[f"dataset_size_{k}"]) for k in "xyz")


@pytest.mark.parametrize("family", ["image_multiscale", "labels_multiscale"])
def test_a_shape_only_change_resizes_the_volume(tmp_path, monkeypatch, family):
    checked = []
    for module in (image_module, labels_module):
        real = module.warn_if_pick_over_budget
        monkeypatch.setattr(
            module,
            "warn_if_pick_over_budget",
            lambda size, *args, _real=real: (
                checked.append(tuple(int(v) for v in size)),
                _real(size, *args),
            )[1],
        )
    viewer = _viewer(DISPLAYED)
    try:
        dtype = np.int32 if family == "labels_multiscale" else np.float32
        data, _ = _volume(LENGTH, dtype)
        visual = _add(viewer, family, data, tmp_path)
        load(viewer)
        target = _render_target(viewer, family, visual)

        before = (LENGTH, 32, 24)
        assert tuple(int(v) for v in target._dataset_size) == before
        assert _uniform_size(target) == before
        old_node = target._inner_node_3d
        old_norm = np.array(target._norm_size)
        fetch_axes = target._volume_geometry.fetch_axes

        # Same displayed (and fetch) axes, x twice as long on every level.
        grown_levels = [(s[0], s[1], s[2] * 2) for s in target._full_level_shapes]
        checked.clear()
        target._sync_geometry_3d(target._last_displayed_axes, grown_levels)

        after = (2 * LENGTH, 32, 24)
        assert target._volume_geometry.fetch_axes == fetch_axes
        assert target._volume_geometry.level_shapes[0] == (24, 32, 2 * LENGTH)
        assert tuple(int(v) for v in target._dataset_size) == after
        assert _uniform_size(target) == after
        # The proxy box (scaled so its longest side is 1) takes the new
        # proportions: x is now twice as long against y and z.
        np.testing.assert_allclose(
            target._norm_size / target._norm_size[0],
            np.asarray(after, dtype=float) / after[0],
        )
        np.testing.assert_allclose(
            target._norm_size[1:] / target._norm_size[0],
            0.5 * old_norm[1:] / old_norm[0],
        )

        # A new node carries the new size and the layout for it.
        new_node = target._inner_node_3d
        assert new_node is not old_node
        assert new_node.pick_layout == pick_bit_layout(target._dataset_size)
        assert new_node.pick_layout.dataset_size == after
        np.testing.assert_allclose(new_node._dataset_size, after)
        np.testing.assert_allclose(new_node._norm_size, target._norm_size)
        # A point past the old far face decodes where it is; the old layout
        # clamps it inside x < LENGTH.
        point = np.array([1.5 * LENGTH + 0.25, 20.25, 7.25])

        def round_trip(layout):
            return layout.decode(np.floor(point * 2.0 ** np.asarray(layout.frac_bits)))

        new_layout = new_node.pick_layout
        np.testing.assert_allclose(
            round_trip(new_layout), point, atol=max(new_layout.step)
        )
        assert round_trip(old_node.pick_layout)[0] < LENGTH

        # The over-budget guard saw the new size.
        assert checked == [after]

        # The same shapes again change nothing.
        target._sync_geometry_3d(target._last_displayed_axes, grown_levels)
        assert target._inner_node_3d is new_node
    finally:
        viewer.controller.close()
