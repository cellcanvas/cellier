"""Per-axis extents on data stores.

The extents are the one number a visual's world extent is built from, so the
two conventions they follow -- edges for a grid, a bounding box for geometry
-- are pinned here rather than left to the consumers.
"""

import numpy as np
import pytest

from cellier.data._base_data_store import (
    BaseDataStore,
    geometry_axis_extents,
    gridded_axis_extents,
)
from cellier.data.graph._graph_memory_store import GraphMemoryStore
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.label._label_memory_store import LabelMemoryStore
from cellier.data.lines._lines_memory_store import LinesMemoryStore
from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore

# -- gridded stores: the edge convention ------------------------------------


def test_gridded_extents_are_edges_not_centres():
    """A voxel is centred on its index, so the axis runs half a voxel past it.

    The regression value for the behaviour change this phase makes: the old
    code read the corner *centres* ``(0, size - 1)``.
    """
    assert gridded_axis_extents((8, 16, 24)) == (
        (-0.5, 7.5),
        (-0.5, 15.5),
        (-0.5, 23.5),
    )


def test_image_store_reports_edge_extents():
    store = ImageMemoryStore(data=np.zeros((13, 32, 64), dtype=np.float32))
    assert store.axis_extents == ((-0.5, 12.5), (-0.5, 31.5), (-0.5, 63.5))


def test_label_store_reports_edge_extents():
    store = LabelMemoryStore(data=np.zeros((4, 5), dtype=np.int32))
    assert store.axis_extents == ((-0.5, 3.5), (-0.5, 4.5))


def test_a_size_one_axis_still_has_width():
    """A singleton axis spans one voxel, not zero -- it is a cell, not a point."""
    assert gridded_axis_extents((1,)) == ((-0.5, 0.5),)


# -- geometry stores: the bounding box --------------------------------------


def test_geometry_extents_are_the_bounding_box():
    positions = np.array([[0.0, 1.0, 2.0], [10.0, 5.0, 3.0], [4.0, -2.0, 2.5]])
    assert geometry_axis_extents(positions) == (
        (0.0, 10.0),
        (-2.0, 5.0),
        (2.0, 3.0),
    )


def test_geometry_extents_have_no_half_voxel_padding():
    """A vertex is a point, not a cell, so nothing is added at the ends."""
    positions = np.array([[0.0], [7.0]])
    assert geometry_axis_extents(positions) == ((0.0, 7.0),)


@pytest.mark.parametrize(
    "store",
    [
        PointsMemoryStore(positions=np.array([[0.0, 1.0], [4.0, 9.0]])),
        LinesMemoryStore(positions=np.array([[0.0, 1.0], [4.0, 9.0]])),
        MeshMemoryStore(
            positions=np.array([[0.0, 1.0, 0.0], [4.0, 9.0, 0.0], [1.0, 2.0, 0.0]]),
            indices=np.array([[0, 1, 2]]),
        ),
        GraphMemoryStore(
            positions=np.array([[0.0, 1.0], [4.0, 9.0]]),
            edges=np.array([[0, 1]]),
        ),
    ],
    ids=["points", "lines", "mesh", "graph"],
)
def test_every_geometry_store_answers(store):
    extents = store.axis_extents
    assert extents is not None
    assert extents[0] == (0.0, 4.0)
    assert extents[1] == (1.0, 9.0)


def test_an_empty_geometry_store_has_no_extent():
    """Empty is not zero-width: it must not pull a union to the origin."""
    store = PointsMemoryStore(positions=np.zeros((0, 3), dtype=np.float32))
    assert store.axis_extents is None


def test_non_2d_positions_raise():
    with pytest.raises(ValueError, match="2D"):
        geometry_axis_extents(np.zeros((4,)))


# -- the base contract ------------------------------------------------------


def test_base_store_refuses_to_invent_an_extent():
    """No default: a store that cannot say what it spans must say so."""

    class Storeless(BaseDataStore):
        pass

    with pytest.raises(NotImplementedError, match="axis_extents"):
        _ = Storeless().axis_extents


def test_multiscale_extents_describe_level_zero():
    """Other levels derive through level_transforms; extents are level 0 only."""
    store = ImageMemoryStore(data=np.zeros((16, 16), dtype=np.float32))
    assert store.axis_extents == gridded_axis_extents(store.level_shapes[0])
