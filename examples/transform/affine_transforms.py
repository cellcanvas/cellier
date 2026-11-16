# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cellier[pyside]",
# ]
# ///

"""An affine transform being applied to tzyx time series.

A tzyx time series simulating an image translating over time.
The point and image should remain aligned over the full time series.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np

from cellier.convenience import Layout, Viewer, axis_values_from_viewer, run
from cellier.convenience.gui import build_canvas_widget
from cellier.data import ImageMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.transform import AffineTransform, Axis, DataCoordinateSystem
from cellier.visuals import InMemoryImageSingleAppearance, PointsMarkerAppearance

#: ``(t, z, y, x)`` shape of the series.
SHAPE = (8, 24, 64, 64)
#: Stage drift in data ``x``, voxels per frame.
DRIFT = 2.0
#: Voxel depth in ``z`` relative to its width.
Z_SCALE = 2.0
#: Rotation of the ``(y, x)`` plane, degrees.
ANGLE = 25.0
#: The blob's centre in the first frame, data ``(z, y, x)``.
BLOB = np.array([12.0, 30.0, 20.0])


def _series() -> tuple[np.ndarray, np.ndarray]:
    """The drifting image and its blob centres, ``(t, z, y, x)``."""
    grid = np.stack(
        np.meshgrid(
            *[np.arange(n, dtype=np.float32) for n in SHAPE[1:]], indexing="ij"
        ),
        axis=-1,
    )
    frames = []
    centres = []
    for t in range(SHAPE[0]):
        centre = BLOB + np.array([0.0, 0.0, DRIFT * t])
        distance = np.linalg.norm(grid - centre, axis=-1)
        blob = np.clip((6.0 - distance) / 2.0, 0.0, 1.0)
        # A dim bar beside the blob drifts with it: the whole field moved.
        bar = np.zeros(SHAPE[1:], dtype=np.float32)
        x0 = int(30 + DRIFT * t)
        bar[8:16, 40:56, x0 : x0 + 4] = 0.4
        frames.append(np.maximum(blob, bar))
        centres.append([float(t), *centre])
    return np.stack(frames).astype(np.float32), np.array(centres, dtype=np.float32)


def _data_system(name: str) -> DataCoordinateSystem:
    """``tzyx`` with a discrete time axis: the slider steps frame by frame."""
    axes = (
        Axis(name="t", axis_type="time", sampling="discrete"),
        Axis(name="z", axis_type="space"),
        Axis(name="y", axis_type="space"),
        Axis(name="x", axis_type="space"),
    )
    return DataCoordinateSystem(name=name, axes=axes, datastore_id=uuid4())


def _matrix() -> np.ndarray:
    """``tzyx -> TZYX``: drift correction, voxel size, rotation (5x5)."""
    theta = np.deg2rad(ANGLE)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    drift = np.eye(4)
    drift[3, 0] = -DRIFT  # x - 2 t
    place = np.eye(4)
    place[1, 1] = Z_SCALE
    place[2:4, 2:4] = rotation
    matrix = np.eye(5)
    matrix[:4, :4] = place @ drift
    return matrix


def build(gui: str = "offscreen") -> Viewer:
    """The viewer: a rotated, drift-corrected image and its tracked blob."""
    viewer = Viewer(
        [("t", "time"), ("z", "space"), ("y", "space"), ("x", "space")],
        displayed_axes=("y", "x"),
        gui=gui,
    )
    world = viewer.scene.dims.world_coordinate_system
    volume, centres = _series()
    matrix = _matrix()

    image = ImageMemoryStore(
        data=volume, name="series", data_coordinate_systems=[_data_system("series")]
    )
    viewer.add_image(
        image,
        single=InMemoryImageSingleAppearance(
            color_map="viridis", clim=(0.0, 1.0), render_mode="mip"
        ),
        name="series",
        transform=AffineTransform.from_matrix(
            matrix, image.data_coordinate_systems[0], world
        ),
    )
    points = PointsMemoryStore(
        positions=centres,
        name="track",
        data_coordinate_systems=[_data_system("track")],
    )
    viewer.add_points(
        points,
        # Drawn after the image and over it: in 2D both sit at depth 0.
        appearance=PointsMarkerAppearance(
            size=12.0,
            color=(1.0, 0.3, 0.3, 1.0),
            render_order=1,
            depth_test=False,
        ),
        name="track",
        transform=AffineTransform.from_matrix(
            matrix, points.data_coordinate_systems[0], world
        ),
    )
    # The blob's plane in z, in the world (z voxels are twice as deep), and
    # a slab one voxel deep so the points near it are selected too.
    viewer.set_slice_positions({1: float(BLOB[0] * Z_SCALE)})
    viewer.controller.update_thickness(viewer.scene.id, {1: Z_SCALE / 2})
    return viewer


if __name__ == "__main__":
    viewer = build(gui="qt")
    canvas_widget = build_canvas_widget(viewer, axis_values_from_viewer(viewer))
    run(viewer, Layout.single(canvas_widget), fit="ready")
