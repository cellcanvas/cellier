"""Fan-out tests for the ``cellier.convenience.OrthoViewer`` add_* methods.

``test_ortho_viewer.py`` already covers ``add_image``'s fan-out. These cover the
remaining add methods, each of which registers one shared data store and one
visual per panel named ``f"{name}_{key}"``.
"""

from __future__ import annotations

from cellier.convenience import OrthoViewer
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    MultiscaleImageChannelAppearance,
    MultiscaleImageSingleAppearance,
)
from cellier.visuals._image import MultiscaleImageAppearance
from cellier.visuals._label_memory import InMemoryLabelsAppearance
from cellier.visuals._labels import MultiscaleLabelsAppearance
from cellier.visuals._lines_memory import LinesMemoryAppearance
from cellier.visuals._mesh_memory import MeshFlatAppearance
from cellier.visuals._points_memory import PointsMarkerAppearance

_PANELS = {"xy", "xz", "yz", "vol"}


def _assert_fanned_out(ortho: OrthoViewer, visuals: dict, name: str) -> None:
    assert set(visuals) == _PANELS
    assert {v.name for v in visuals.values()} == {f"{name}_{k}" for k in _PANELS}
    assert len(ortho.controller._model.data.stores) == 1
    for scene in ortho.scenes.values():
        assert len(scene.visuals) == 1


def test_add_labels_fans_out(labels_store):
    ortho = OrthoViewer(spatial_axes("z", "y", "x"))
    visuals = ortho.add_labels(
        labels_store, appearance=InMemoryLabelsAppearance(), name="lbl"
    )
    _assert_fanned_out(ortho, visuals, "lbl")


def test_add_mesh_fans_out(mesh_store):
    ortho = OrthoViewer(spatial_axes("z", "y", "x"))
    visuals = ortho.add_mesh(mesh_store, appearance=MeshFlatAppearance(), name="m")
    _assert_fanned_out(ortho, visuals, "m")


def test_add_points_fans_out(points_store):
    ortho = OrthoViewer(spatial_axes("z", "y", "x"))
    visuals = ortho.add_points(
        points_store, appearance=PointsMarkerAppearance(), name="pts"
    )
    _assert_fanned_out(ortho, visuals, "pts")


def test_add_lines_fans_out(lines_store):
    ortho = OrthoViewer(spatial_axes("z", "y", "x"))
    visuals = ortho.add_lines(
        lines_store, appearance=LinesMemoryAppearance(), name="ln"
    )
    _assert_fanned_out(ortho, visuals, "ln")


def test_add_image_multiscale_fans_out(multiscale_image_store):
    ortho = OrthoViewer(spatial_axes("z", "y", "x"))
    visuals = ortho.add_image_multiscale(
        multiscale_image_store,
        appearance=MultiscaleImageAppearance(),
        name="ms",
        single=MultiscaleImageSingleAppearance(color_map="viridis", render_mode="mip"),
    )
    _assert_fanned_out(ortho, visuals, "ms")


def test_add_labels_multiscale_fans_out(multiscale_labels_store):
    ortho = OrthoViewer(spatial_axes("z", "y", "x"))
    visuals = ortho.add_labels_multiscale(
        multiscale_labels_store,
        appearance=MultiscaleLabelsAppearance(),
        name="mslbl",
    )
    _assert_fanned_out(ortho, visuals, "mslbl")


def test_add_image_multiscale_composite_fans_out(multichannel_multiscale_store):
    ortho = OrthoViewer(
        [("c", "channel"), ("z", "space"), ("y", "space"), ("x", "space")],
        spatial_axes=("z", "y", "x"),
    )
    channels = {
        0: MultiscaleImageChannelAppearance(color_map="red"),
        1: MultiscaleImageChannelAppearance(color_map="green"),
    }
    visuals = ortho.add_image_multiscale(
        multichannel_multiscale_store,
        name="mc",
        channel_axis=0,
        composite=True,
        channels=channels,
    )
    _assert_fanned_out(ortho, visuals, "mc")
    appearances = set()
    for visual in visuals.values():
        assert set(visual.channels) == {0, 1}
        appearances.add(id(visual.channels[0]))
    # Each panel owns its own copy, so a direct edit on one panel stays there.
    assert len(appearances) == 4
