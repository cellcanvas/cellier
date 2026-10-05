"""An example of how 2d cross-sections appear for watertight meshes.

The scene has two meshes:

- a "cell": a closed sphere with a smaller sphere inside it whose faces are
  wound the other way.  The inner sphere is a cavity, so the fill has a hole.
- a "nucleus": a closed sphere inside the cell, drawn as an outline only.

Where two meshes overlap in a 2D view, the one with the higher
render_order is on top, then the one added later; an outline is drawn
over a fill.  Thus, the nucleus's outline shows on the cell's fill.
"""

from __future__ import annotations

import numpy as np

from cellier.convenience import (
    AppearanceControls,
    ContinuousAxisValues,
    Layout,
    Viewer,
    run,
)
from cellier.convenience.gui import MeshControlsConfig, build_canvas_widget
from cellier.data.mesh import MeshMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.visuals import MeshPhongAppearance, MeshSectionConfig

#: The meshes sit inside a cube of this many world units a side.
EXTENT = 100.0

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def sphere(
    radius: float, centre: tuple[float, float, float], n_lat: int = 48
) -> tuple[np.ndarray, np.ndarray]:
    """A closed sphere: ``(positions (V, 3) as (z, y, x), faces (F, 3))``.

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
    positions = radius * positions + np.asarray(centre)
    return positions.astype(np.float32), np.array(faces, dtype=np.int32)


# The cell: an outer surface and a cavity.  The cavity's faces are reversed,
# so they are wound the other way from the outer surface's.  The fill follows
# the winding: inside the outer surface is filled, inside the cavity is not.
outer_positions, outer_faces = sphere(30.0, (50.0, 50.0, 38.0))
cavity_positions, cavity_faces = sphere(9.0, (50.0, 58.0, 50.0))
cell_store = MeshMemoryStore(
    positions=np.concatenate([outer_positions, cavity_positions]),
    indices=np.concatenate([outer_faces, cavity_faces[:, ::-1] + len(outer_positions)]),
    name="cell",
)

nucleus_positions, nucleus_faces = sphere(10.0, (50.0, 40.0, 28.0))
nucleus_store = MeshMemoryStore(
    positions=nucleus_positions, indices=nucleus_faces, name="nucleus"
)

# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

viewer = Viewer(
    spatial_axes("z", "y", "x"),
    dim="2d",
    # Phong shading needs lights (for the 3D view).
    lighting="default",
)

controls = MeshControlsConfig(appearance=True, section_controls=True)

viewer.add_mesh(
    nucleus_store,
    MeshPhongAppearance(color=(0.95, 0.6, 0.2, 1.0), side="both"),
    name="nucleus",
    # Outline only: the cell's fill shows inside it.
    section=MeshSectionConfig(outline=True, fill=False, outline_width=3.0),
    controls=controls,
)
viewer.add_mesh(
    cell_store,
    MeshPhongAppearance(color=(0.35, 0.65, 0.95, 1.0), side="both"),
    name="cell",
    # The default: an outline and a fill of the cut at the slice plane.
    section=MeshSectionConfig(mode="cut", outline=True, fill=True, outline_width=2.0),
    controls=controls,
)

# ---------------------------------------------------------------------------
# Canvas, sliders, layout
# ---------------------------------------------------------------------------

# Slider ranges, by world axis.  Written out because a scene that holds only
# geometry has no image shape to read them from.
axis_values = {axis: ContinuousAxisValues(min=0.0, max=EXTENT) for axis in (0, 1, 2)}
canvas_widget = build_canvas_widget(viewer, axis_values)

# Start on the plane through the middle of the cell: it crosses the cavity
# and the nucleus.
viewer.controller.update_slice_indices(viewer.scene.id, {0: 50.0})

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run(
        viewer,
        Layout(center=canvas_widget, left_dock=AppearanceControls()),
        fit="ready",
    )
