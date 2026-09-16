"""Tests for the ``cellier.convenience.Viewer`` add_* visual methods.

Only ``add_image`` was previously exercised (via the serialization tests). These cover
the remaining add methods, their dict-coercion branches, the UUID data-store branch, the
controls-recording branch, ``add_canvas``, and the ``from_file`` guard.
"""

from __future__ import annotations

import pytest

from cellier.controller import CellierController
from cellier.convenience import Viewer
from cellier.convenience.gui import (
    InMemoryImageControlsConfig,
    MultiscaleImageControlsConfig,
)
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    InMemoryImageSingleAppearance,
    MultiscaleImageChannelAppearance,
    MultiscaleImageSingleAppearance,
)
from cellier.visuals._image import MultiscaleImageAppearance
from cellier.visuals._image_memory import InMemoryImageAppearance
from cellier.visuals._label_memory import InMemoryLabelsAppearance
from cellier.visuals._labels import MultiscaleLabelsAppearance
from cellier.visuals._lines_memory import LinesMemoryAppearance
from cellier.visuals._mesh_memory import MeshFlatAppearance, MeshPhongAppearance
from cellier.visuals._points_memory import PointsMarkerAppearance


def _visual_ids(viewer: Viewer) -> set:
    return {v.id for v in viewer.scene.visuals}


def _n_stores(viewer: Viewer) -> int:
    return len(viewer.controller._model.data.stores)


# ---------------------------------------------------------------------------
# In-memory geometry / labels
# ---------------------------------------------------------------------------


def test_add_labels_registers_visual(labels_store):
    viewer = Viewer(spatial_axes("z", "y", "x"))
    visual = viewer.add_labels(
        labels_store, appearance=InMemoryLabelsAppearance(), name="lbl"
    )
    assert visual.name == "lbl"
    assert visual.id in _visual_ids(viewer)
    assert _n_stores(viewer) == 1


def test_add_labels_defaults_appearance_when_none(labels_store):
    viewer = Viewer(spatial_axes("z", "y", "x"))
    visual = viewer.add_labels(labels_store)
    assert visual.id in _visual_ids(viewer)


def test_add_mesh_flat(mesh_store):
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d")
    visual = viewer.add_mesh(mesh_store, appearance=MeshFlatAppearance(), name="m")
    assert visual.name == "m"
    assert visual.id in _visual_ids(viewer)


def test_add_mesh_phong(mesh_store):
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d")
    # A phong mesh in a scene without lighting warns (renders black otherwise).
    with pytest.warns(UserWarning, match="requires lights"):
        visual = viewer.add_mesh(mesh_store, appearance=MeshPhongAppearance())
    assert visual.id in _visual_ids(viewer)


def test_add_points(points_store):
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d")
    visual = viewer.add_points(
        points_store, appearance=PointsMarkerAppearance(), name="pts"
    )
    assert visual.name == "pts"
    assert visual.id in _visual_ids(viewer)


def test_add_points_defaults_appearance_when_none(points_store):
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d")
    visual = viewer.add_points(points_store)
    assert visual.id in _visual_ids(viewer)


def test_add_lines(lines_store):
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d")
    visual = viewer.add_lines(
        lines_store, appearance=LinesMemoryAppearance(), name="ln"
    )
    assert visual.name == "ln"
    assert visual.id in _visual_ids(viewer)


def test_add_lines_defaults_appearance_when_none(lines_store):
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d")
    visual = viewer.add_lines(lines_store)
    assert visual.id in _visual_ids(viewer)


# ---------------------------------------------------------------------------
# Multiscale
# ---------------------------------------------------------------------------


def test_add_image_multiscale(multiscale_image_store):
    viewer = Viewer(spatial_axes("z", "y", "x"))
    visual = viewer.add_image_multiscale(
        multiscale_image_store,
        appearance=MultiscaleImageAppearance(),
        name="ms",
        single=MultiscaleImageSingleAppearance(color_map="viridis", render_mode="mip"),
    )
    assert visual.name == "ms"
    assert visual.id in _visual_ids(viewer)


def test_add_labels_multiscale(multiscale_labels_store):
    viewer = Viewer(spatial_axes("z", "y", "x"))
    visual = viewer.add_labels_multiscale(
        multiscale_labels_store,
        appearance=MultiscaleLabelsAppearance(),
        name="mslbl",
    )
    assert visual.name == "mslbl"
    assert visual.id in _visual_ids(viewer)


def test_add_image_multiscale_composite(multichannel_multiscale_store):
    viewer = Viewer([("c", "channel"), ("z", "space"), ("y", "space"), ("x", "space")])
    channels = {
        0: MultiscaleImageChannelAppearance(color_map="red"),
        1: MultiscaleImageChannelAppearance(color_map="green"),
    }
    visual = viewer.add_image_multiscale(
        multichannel_multiscale_store,
        name="mc",
        channel_axis=0,
        composite=True,
        channels=channels,
    )
    assert visual.name == "mc"
    assert visual.id in _visual_ids(viewer)
    assert visual.composite is True
    assert set(visual.channels) == {0, 1}


def test_add_image_multiscale_composite_records_controls(
    multichannel_multiscale_store,
):
    viewer = Viewer([("c", "channel"), ("z", "space"), ("y", "space"), ("x", "space")])
    visual = viewer.add_image_multiscale(
        multichannel_multiscale_store,
        controls=MultiscaleImageControlsConfig(appearance=True),
        channel_axis=0,
        composite=True,
        channels={0: MultiscaleImageChannelAppearance(color_map="red")},
    )
    assert visual.id in viewer._controls_configs


# ---------------------------------------------------------------------------
# UUID data-store branch, controls recording, add_canvas, from_file guard
# ---------------------------------------------------------------------------


def test_add_image_accepts_registered_store_uuid(image_store):
    viewer = Viewer(spatial_axes("z", "y", "x"))
    viewer.controller.add_data_store(image_store)
    visual = viewer.add_image(
        image_store.id,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )
    assert visual.id in _visual_ids(viewer)
    assert _n_stores(viewer) == 1


def test_on_ready_registers_callback(image_store):
    viewer = Viewer(spatial_axes("z", "y", "x"))
    called = []
    viewer.on_ready(lambda: called.append(True))
    assert viewer._ready_callbacks and viewer._ready_callbacks[-1] is not None


def test_add_image_records_controls_config(image_store):
    viewer = Viewer(spatial_axes("z", "y", "x"))
    visual = viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(),
        controls=InMemoryImageControlsConfig(appearance=["color_map", "clim"]),
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )
    assert visual.id in viewer._controls_configs


def test_add_image_multiscale_records_controls_config(multiscale_image_store):
    viewer = Viewer(spatial_axes("z", "y", "x"))
    visual = viewer.add_image_multiscale(
        multiscale_image_store,
        appearance=MultiscaleImageAppearance(),
        controls=MultiscaleImageControlsConfig(appearance=["color_map"]),
        single=MultiscaleImageSingleAppearance(color_map="viridis", render_mode="mip"),
    )
    assert visual.id in viewer._controls_configs


def test_add_canvas_returns_widget_anywidget(image_store):
    viewer = Viewer(spatial_axes("z", "y", "x"), gui="anywidget")
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )
    widget = viewer.add_canvas()
    assert widget is not None


def test_from_file_rejects_multi_scene_model(tmp_path, image_store):
    """A model with two scenes is not a valid single-scene Viewer file."""
    controller = CellierController()
    from cellier.scene.dims import spatial_axes, world_coordinate_system

    for name in ("a", "b"):
        controller.add_scene(
            name=name,
            dim="2d",
            coordinate_system=world_coordinate_system(
                spatial_axes("z", "y", "x"), name="world"
            ),
        )
    path = tmp_path / "multi.json"
    controller.to_file(path)

    with pytest.raises(ValueError, match="exactly one scene"):
        Viewer.from_file(path)
