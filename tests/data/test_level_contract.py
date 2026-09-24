"""The level-transform contract (C1-C4) and where it is enforced.

``plans/multiscale_level_transform_v2.md``, "The contract".  The renderer
places pyramid levels that are axis-aligned, coarser with level, offset by
less than one coarse voxel and covering level 0; anything else is rejected
when the store gets its transforms, and again when a store whose transforms
were set directly is added to a scene.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import tensorstore as ts

from cellier.controller import CellierController
from cellier.data._level_contract import (
    level_contract_issues,
    validate_level_transforms,
)
from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.transform import AffineTransform
from tests._v2 import data_system


def _levels(scales, translations):
    """Duck-typed ``level k -> level 0`` transforms (diagonal)."""
    return [
        SimpleNamespace(linear=np.diag(s), translation=np.asarray(t, dtype=float))
        for s, t in zip(scales, translations)
    ]


def _checks(scales, translations, shapes=None):
    issues = level_contract_issues(_levels(scales, translations), shapes)
    return sorted({(i.level, i.axis, i.check.split()[0]) for i in issues})


SHAPES_2 = [(16, 16), (8, 8)]


@pytest.mark.parametrize(
    "translation",
    [(0.5, 0.5), (1.0, 1.0), (0.0, 0.0), (-0.5, 1.5)],
    ids=["block_average", "offset_striding", "plain_striding", "bounds"],
)
def test_supported_pyramid_kinds_pass(translation):
    assert _checks([(1, 1), (2, 2)], [(0, 0), translation], SHAPES_2) == []


def test_non_integer_ratios_pass():
    """98 -> 49 -> 24 (ratios 2 and 4.083), block-average translations."""
    scales = [(1.0,), (2.0,), (98 / 24,)]
    translations = [(0.0,), (0.5,), ((98 / 24 - 1) / 2,)]
    assert _checks(scales, translations, [(98,), (49,), (24,)]) == []


def test_the_lightsheet_metadata_passes():
    """Scale-only (t = 0) with non-integer y ratios, as published (tczyx)."""
    shapes = [(667, 2, 76, 389, 610), (667, 2, 76, 194, 305), (667, 2, 76, 96, 152)]
    scales = [
        (1, 1, 1, 1, 1),
        (1, 1, 1, 0.521340206185567 / 0.26, 2.0),
        (1, 1, 1, 1.0535416666666666 / 0.26, 1.043421052631579 / 0.26),
    ]
    translations = [(0,) * 5] * 3
    assert _checks(scales, translations, shapes) == []


def test_c1_off_diagonal():
    levels = _levels([(1, 1), (2, 2)], [(0, 0), (0.5, 0.5)])
    levels[1].linear = np.array([[2.0, 0.1], [0.0, 2.0]])
    issues = level_contract_issues(levels)
    assert [(i.level, i.axis, i.check[:2]) for i in issues] == [(1, None, "C1")]


def test_c2_finer_than_level_zero_or_than_the_previous_level():
    assert _checks([(1, 1), (0.5, 1)], [(0, 0), (0, 0)]) == [(1, 0, "C2")]
    assert _checks([(1,), (4,), (2,)], [(0,), (0,), (0,)]) == [(2, 0, "C2")]


def test_c3_translation_of_a_voxel_or_more():
    assert _checks([(1,), (2,)], [(0,), (1.6,)]) == [(1, 0, "C3")]
    assert _checks([(1,), (2,)], [(0,), (-0.6,)]) == [(1, 0, "C3")]


def test_c4_cropped_level():
    """Level 1 holds only half of level 0's extent."""
    assert _checks([(1,), (2,)], [(0,), (0.5,)], [(16,), (4,)]) == [(1, 0, "C4")]


def test_trivial_axes_pass():
    """Time and channel axes: scale 1, translation 0, same size."""
    assert (
        _checks(
            [(1, 1, 1), (1, 1, 2)], [(0, 0, 0), (0, 0, 0.5)], [(5, 2, 8), (5, 2, 4)]
        )
        == []
    )


def test_message_names_store_level_axis_value_and_range():
    with pytest.raises(ValueError) as info:
        validate_level_transforms(
            _levels([(1, 1), (2, 2)], [(0, 0), (0.5, 3.0)]), name="pyramid-x"
        )
    message = str(info.value)
    assert "pyramid-x" in message
    assert "level 1, axis 1: C3 translation is 3, allowed [-0.5, 1.5]" in message
    assert "offset by less than one of its own voxels" in message


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------

_SHAPES = {"s0": (8, 8, 8), "s1": (4, 4, 4)}


@pytest.fixture
def pyramid_path(tmp_path) -> str:
    for name, shape in _SHAPES.items():
        ts.open(
            {
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": str(tmp_path / name)},
            },
            create=True,
            dtype=ts.uint8,
            shape=shape,
        ).result()
    return str(tmp_path)


def _store(path, translation, **kwargs) -> MultiscaleZarrDataStore:
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=path,
        scale_names=list(_SHAPES),
        level_scales=[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0), translation],
        name="enforced",
        **kwargs,
    )


def test_a_store_with_its_own_systems_is_checked_at_construction(pyramid_path):
    with pytest.raises(ValueError, match="C3 translation"):
        _store(
            pyramid_path,
            (0.5, 0.5, 4.0),
            data_coordinate_system=data_system(("z", "y", "x")),
        )


def test_a_store_placed_by_a_scene_is_checked_when_added(pyramid_path):
    store = _store(pyramid_path, (0.5, 0.5, 4.0))
    assert store.level_transforms == []
    controller = CellierController()
    try:
        scene = controller.add_scene(dim="3d", name="contract")
        with pytest.raises(ValueError, match="C3 translation"):
            controller.add_image_multiscale(data=store, scene_id=scene.id)
    finally:
        controller.close()


def test_transforms_set_directly_are_checked_when_added(pyramid_path):
    """A store restored with its transforms skips the install; the add checks."""
    store = _store(
        pyramid_path,
        (0.5, 0.5, 0.5),
        data_coordinate_system=data_system(("z", "y", "x")),
    )
    level0, level1 = store.data_coordinate_systems
    store.level_transforms = [
        store.level_transforms[0],
        AffineTransform.from_axis_map(
            level1,
            level0,
            axis_map={a.id: b.id for a, b in zip(level1.axes, level0.axes)},
            scale={a.id: 2.0 for a in level1.axes},
            translation={a.id: 4.0 for a in level1.axes},
            name="level1_to_level0",
        ),
    ]
    controller = CellierController()
    try:
        scene = controller.add_scene(dim="3d", name="contract")
        with pytest.raises(ValueError, match="C3 translation"):
            controller.add_image_multiscale(data=store, scene_id=scene.id)
    finally:
        controller.close()
