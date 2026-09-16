"""Tests for CellierController.add_canvas camera-type selection."""

from __future__ import annotations

import pytest

from cellier.controller import CellierController
from cellier.scene.cameras import OrthographicCamera, PerspectiveCamera
from cellier.scene.dims import (
    AxisAlignedSelection,
    DimsManager,
    spatial_axes,
    world_coordinate_system,
)
from cellier.scene.scene import Scene


def _make_scene(
    displayed_axes: tuple[int, ...],
    render_modes: set[str] | None = None,
) -> Scene:
    """Return a minimal Scene with the given displayed_axes and render_modes."""
    n_axes = max(displayed_axes) + 1
    axis_labels = tuple(f"axis_{i}" for i in range(n_axes))
    cs = world_coordinate_system(spatial_axes(*axis_labels), name="world")
    slice_indices = dict.fromkeys(range(n_axes), 0)
    dims = DimsManager(
        world_coordinate_system=cs,
        selection=AxisAlignedSelection(
            displayed_axes=displayed_axes,
            slice_indices=slice_indices,
        ),
    )
    resolved_modes = render_modes if render_modes is not None else {"2d", "3d"}
    return Scene(name="test_scene", dims=dims, render_modes=resolved_modes)


# ---------------------------------------------------------------------------
# Single-mode canvases — dim inferred from scene
# ---------------------------------------------------------------------------


def test_add_canvas_3d_scene_creates_perspective_camera(qtbot):
    """Inferring dim from a 3D scene stores a PerspectiveCamera in the model."""
    controller = CellierController()
    scene = controller.add_scene_model(_make_scene(displayed_axes=(0, 1, 2)))

    controller.add_canvas(scene_id=scene.id)

    canvas_model = next(iter(scene.canvases.values()))
    assert "3d" in canvas_model.cameras
    assert isinstance(canvas_model.cameras["3d"], PerspectiveCamera)
    assert canvas_model.cameras["3d"].near_clipping_plane == pytest.approx(1.0)
    assert canvas_model.cameras["3d"].far_clipping_plane == pytest.approx(8000.0)


def test_add_canvas_2d_scene_creates_orthographic_camera(qtbot):
    """Inferring dim from a 2D scene stores an OrthographicCamera in the model."""
    controller = CellierController()
    scene = controller.add_scene_model(_make_scene(displayed_axes=(1, 2)))

    controller.add_canvas(scene_id=scene.id)

    canvas_model = next(iter(scene.canvases.values()))
    assert "2d" in canvas_model.cameras
    assert isinstance(canvas_model.cameras["2d"], OrthographicCamera)
    assert canvas_model.cameras["2d"].near_clipping_plane == pytest.approx(-500.0)
    assert canvas_model.cameras["2d"].far_clipping_plane == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Explicit render_modes — single mode
# ---------------------------------------------------------------------------


def test_add_canvas_explicit_render_modes_3d_only(qtbot):
    """render_modes={'3d'} stores only a PerspectiveCamera, no '2d' key."""
    controller = CellierController()
    scene = controller.add_scene_model(_make_scene(displayed_axes=(0, 1, 2)))

    controller.add_canvas(scene_id=scene.id, render_modes={"3d"})

    canvas_model = next(iter(scene.canvases.values()))
    assert set(canvas_model.cameras.keys()) == {"3d"}
    assert isinstance(canvas_model.cameras["3d"], PerspectiveCamera)


def test_add_canvas_explicit_render_modes_2d_only(qtbot):
    """render_modes={'2d'} stores only an OrthographicCamera, no '3d' key."""
    controller = CellierController()
    scene = controller.add_scene_model(_make_scene(displayed_axes=(0, 1, 2)))

    controller.add_canvas(
        scene_id=scene.id,
        render_modes={"2d"},
        initial_dim="2d",
    )

    canvas_model = next(iter(scene.canvases.values()))
    assert set(canvas_model.cameras.keys()) == {"2d"}
    assert isinstance(canvas_model.cameras["2d"], OrthographicCamera)


# ---------------------------------------------------------------------------
# Dual-mode canvas
# ---------------------------------------------------------------------------


def test_add_canvas_both_modes_stores_both_cameras(qtbot):
    """render_modes={'2d','3d'} stores both camera types in the model."""
    controller = CellierController()
    scene = controller.add_scene_model(_make_scene(displayed_axes=(0, 1, 2)))

    controller.add_canvas(scene_id=scene.id, render_modes={"2d", "3d"})

    canvas_model = next(iter(scene.canvases.values()))
    assert set(canvas_model.cameras.keys()) == {"2d", "3d"}
    assert isinstance(canvas_model.cameras["3d"], PerspectiveCamera)
    assert isinstance(canvas_model.cameras["2d"], OrthographicCamera)


def test_add_canvas_both_modes_initial_dim_2d(qtbot):
    """initial_dim='2d' on a dual-mode canvas activates the 2D camera first."""
    controller = CellierController()
    # Scene is 3D but canvas supports both.
    scene = controller.add_scene_model(_make_scene(displayed_axes=(0, 1, 2)))

    controller.add_canvas(
        scene_id=scene.id,
        render_modes={"2d", "3d"},
        initial_dim="2d",
    )

    canvas_model = next(iter(scene.canvases.values()))
    assert "2d" in canvas_model.cameras
    assert "3d" in canvas_model.cameras
    # CanvasView._dim should reflect initial_dim.
    canvas_view = controller._render_manager._canvases[canvas_model.id]
    assert canvas_view._dim == "2d"


# ---------------------------------------------------------------------------
# Custom depth ranges
# ---------------------------------------------------------------------------


def test_add_canvas_custom_depth_range_3d(qtbot):
    """depth_range_3d is stored on the PerspectiveCamera."""
    controller = CellierController()
    scene = controller.add_scene_model(_make_scene(displayed_axes=(0, 1, 2)))

    controller.add_canvas(scene_id=scene.id, depth_range_3d=(0.1, 100.0))

    canvas_model = next(iter(scene.canvases.values()))
    camera = canvas_model.cameras["3d"]
    assert camera.near_clipping_plane == pytest.approx(0.1)
    assert camera.far_clipping_plane == pytest.approx(100.0)


def test_add_canvas_custom_depth_range_2d(qtbot):
    """depth_range_2d is stored on the OrthographicCamera."""
    controller = CellierController()
    scene = controller.add_scene_model(_make_scene(displayed_axes=(0, 1, 2)))

    controller.add_canvas(
        scene_id=scene.id,
        render_modes={"2d", "3d"},
        depth_range_2d=(-1000.0, 1000.0),
    )

    canvas_model = next(iter(scene.canvases.values()))
    camera = canvas_model.cameras["2d"]
    assert camera.near_clipping_plane == pytest.approx(-1000.0)
    assert camera.far_clipping_plane == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_add_canvas_initial_dim_not_in_render_modes_raises(qtbot):
    """Passing initial_dim outside render_modes raises ValueError."""
    controller = CellierController()
    scene = controller.add_scene_model(_make_scene(displayed_axes=(0, 1, 2)))

    with pytest.raises(ValueError, match="initial_dim"):
        controller.add_canvas(
            scene_id=scene.id,
            render_modes={"3d"},
            initial_dim="2d",
        )
