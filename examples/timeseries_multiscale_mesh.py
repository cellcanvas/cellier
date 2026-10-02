"""A time series of moving ellipsoid meshes, with levels of detail.

The same scene as ``timeseries_mesh.py``, but the mesh is stored at two
levels: a fine one and a coarse one.  Cellier keeps both loaded and
draws one of them:

- Drag the ``t`` slider (or ``z`` in 2D): each step loads and draws the
  coarse level.  When the slider rests, the fine level loads and replaces it.
- Click on the slider track or type a value: the coarse level is drawn
  first and the fine level replaces it when it has loaded.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np

from cellier.convenience import (
    AppearanceControls,
    ContinuousAxisValues,
    DiscreteAxisValues,
    Layout,
    Viewer,
    run,
)
from cellier.convenience.gui import MeshControlsConfig, build_canvas_widget
from cellier.data.mesh import MeshLevel, MultiscaleMeshStore
from cellier.render import OutlineConfig, RenderManagerConfig
from cellier.scene.dims import spatial_axes
from cellier.transform import Axis, DataCoordinateSystem
from cellier.visuals import GeometryLodConfig, MeshPhongAppearance, MeshSectionConfig

# ---------------------------------------------------------------------------
# Sizes
# ---------------------------------------------------------------------------

N_CELLS = 16
N_TIMEPOINTS = 20
#: Latitude bands of each ellipsoid at each level, finest first.  An ellipsoid
#: with ``n`` bands has about ``4 * n**2`` faces.
RESOLUTIONS = (48, 6)
#: The cells move inside a cube of this many world units a side.
EXTENT = 100.0
SEED = 7

# ---------------------------------------------------------------------------
# Data: ellipsoids on a persistent random walk
# ---------------------------------------------------------------------------


def unit_sphere(n_lat: int) -> tuple[np.ndarray, np.ndarray]:
    """A closed unit sphere: ``(positions (V, 3) as (z, y, x), faces (F, 3))``.

    Two pole vertices and ``n_lat - 1`` rings, so every edge is shared by two
    faces.  A closed surface is what lets a 2D section be filled.
    """
    n_lon = 2 * n_lat
    theta = np.linspace(0.0, np.pi, n_lat + 1)[1:-1]
    phi = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    rings = np.stack(
        [
            np.repeat(np.cos(theta), n_lon),
            np.outer(np.sin(theta), np.sin(phi)).ravel(),
            np.outer(np.sin(theta), np.cos(phi)).ravel(),
        ],
        axis=1,
    )
    positions = np.concatenate([[[1.0, 0.0, 0.0]], rings, [[-1.0, 0.0, 0.0]]])
    south = len(positions) - 1
    last_ring = 1 + (n_lat - 2) * n_lon
    faces = []
    for j in range(n_lon):
        k = (j + 1) % n_lon
        faces.append([0, 1 + j, 1 + k])
        faces.append([south, last_ring + k, last_ring + j])
    for i in range(n_lat - 2):
        for j in range(n_lon):
            k = (j + 1) % n_lon
            a, b = 1 + i * n_lon + j, 1 + i * n_lon + k
            faces += [[a, a + n_lon, b], [b, a + n_lon, b + n_lon]]
    return positions, np.array(faces, dtype=np.int32)


def rotation_onto(direction: np.ndarray) -> np.ndarray:
    """A rotation taking the first axis onto *direction* (a unit vector)."""
    first = direction / np.linalg.norm(direction)
    helper = (
        np.array([0.0, 0.0, 1.0]) if abs(first[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    second = np.cross(helper, first)
    second /= np.linalg.norm(second)
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=1)


def migrating_cells(
    n_cells: int, n_timepoints: int, n_lat: int, extent: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Every cell at every timepoint, as one mesh.

    Returns
    -------
    positions : np.ndarray
        ``(n_timepoints * n_cells * V, 4)`` float32, columns ``(t, z, y, x)``;
        ``t`` is the timepoint's index.
    indices : np.ndarray
        ``(n_timepoints * n_cells * F, 3)`` int32.
    """
    rng = np.random.default_rng(seed)
    sphere, faces = unit_sphere(n_lat)

    centres = rng.uniform(0.25 * extent, 0.75 * extent, (n_cells, 3))
    velocities = rng.normal(size=(n_cells, 3))
    velocities /= np.linalg.norm(velocities, axis=1, keepdims=True)
    speeds = rng.uniform(1.0, 3.0, n_cells)
    radii = rng.uniform(6.0, 10.0, n_cells)
    phases = rng.uniform(0.0, 2.0 * np.pi, n_cells)

    all_positions, all_faces, offset = [], [], 0
    for t in range(n_timepoints):
        for cell in range(n_cells):
            # Long along the direction of travel, and breathing over time;
            # the two short axes shrink to keep the volume.
            stretch = 1.5 + 0.4 * np.sin(0.6 * t + phases[cell])
            semi_axes = radii[cell] * np.array([stretch, stretch**-0.5, stretch**-0.5])
            points = (sphere * semi_axes) @ rotation_onto(velocities[cell]).T
            points += centres[cell]
            time = np.full((len(points), 1), float(t))
            all_positions.append(np.concatenate([time, points], axis=1))
            all_faces.append(faces + offset)
            offset += len(points)

        # One step of the walk: mostly the old heading, a little noise, and
        # a turn back toward the middle near the walls.
        velocities += 0.35 * rng.normal(size=velocities.shape)
        outside = (centres < 0.15 * extent) | (centres > 0.85 * extent)
        velocities += np.where(outside, np.sign(0.5 * extent - centres), 0.0)
        velocities /= np.linalg.norm(velocities, axis=1, keepdims=True)
        centres = centres + speeds[:, None] * velocities

    return (
        np.concatenate(all_positions).astype(np.float32),
        np.concatenate(all_faces).astype(np.int32),
    )


# One mesh per level.  The same seed gives the same walk, so every level is
# the same cells at the same places, only with more or fewer faces.
levels = [
    MeshLevel(positions=positions, indices=indices)
    for positions, indices in (
        migrating_cells(N_CELLS, N_TIMEPOINTS, resolution, EXTENT, SEED)
        for resolution in RESOLUTIONS
    )
]

# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

# ``t`` holds timepoint indices, so it is a discrete axis: the slider then
# selects whole timepoints, and the mesh changes frame at the same instant
# an image on the same axis would.
# The levels are listed finest first.  They share one coordinate system.
store = MultiscaleMeshStore(
    levels=levels,
    name="cells",
    data_coordinate_systems=[
        DataCoordinateSystem(
            name="cells",
            axes=(
                Axis(name="t", axis_type="time", sampling="discrete"),
                Axis(name="z", axis_type="space"),
                Axis(name="y", axis_type="space"),
                Axis(name="x", axis_type="space"),
            ),
            datastore_id=uuid4(),
        )
    ],
)

# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

viewer = Viewer(
    [("t", "time"), *spatial_axes("z", "y", "x")],
    dim="3d",
    # Phong shading needs lights.
    lighting="default",
    # The screen-space outline pass, for the "Outline" controls.
    render_config=RenderManagerConfig(outline=OutlineConfig(enabled=True)),
)

viewer.add_multiscale_mesh(
    store,
    MeshPhongAppearance(color=(0.35, 0.65, 0.95, 1.0), side="both"),
    name="cells",
    # How a 2D view draws the mesh: an outline and a fill of the cut.
    section=MeshSectionConfig(mode="cut", outline=True, fill=True, outline_width=2.0),
    # Which coarse level is kept beside the finest (None: the coarsest), and
    # what a slider drag loads: "coarse" loads the coarse level while the
    # slider moves and the finest when it rests.
    lod=GeometryLodConfig(coarse_level=None, dims_drag="coarse"),
    controls=MeshControlsConfig(
        appearance=True,
        section_controls=True,
        outline_controls=True,
        dataset_info=True,
    ),
)

# ---------------------------------------------------------------------------
# Canvas, sliders, layout
# ---------------------------------------------------------------------------

# Slider ranges, by world axis.  Written out because a scene that holds only
# geometry has no image shape to read them from.
axis_values = {
    0: DiscreteAxisValues(values=tuple(float(t) for t in range(N_TIMEPOINTS))),
    **{axis: ContinuousAxisValues(min=0.0, max=EXTENT) for axis in (1, 2, 3)},
}
canvas_widget = build_canvas_widget(viewer, axis_values)

# Start the z slider on the plane that crosses the most cells at t = 0, so
# "Switch to 2D" lands on something.
fine = levels[0].positions
first_frame = fine[fine[:, 0] == 0.0, 1].reshape(N_CELLS, -1)
planes = np.arange(0.0, EXTENT + 1.0)
crossed = (first_frame.min(axis=1)[:, None] < planes) & (
    planes < first_frame.max(axis=1)[:, None]
)
start_z = float(planes[np.argmax(crossed.sum(axis=0))])
viewer.controller.update_slice_indices(viewer.scene.id, {0: 0.0, 1: start_z})

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run(
        viewer,
        Layout(center=canvas_widget, left_dock=AppearanceControls()),
        fit="ready",
    )
