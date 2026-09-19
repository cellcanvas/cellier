r"""A capture target: three spheres in a volume.

Meant for ``cellier.convenience.capture``.

Run it from the repo root (no display needed)::

    .venv/bin/python -m cellier.convenience.capture examples/capture/spheres.py \
        --size 800x600 --out spheres.png

or, with ambient occlusion on (which is what ``--frames converged`` is for --
a single-sample AO frame is visibly noisy)::

    .venv/bin/python -m cellier.convenience.capture examples/capture/spheres.py \
        --size 800x600 --frames converged --out spheres_ao.png

The only contract a target has to meet is a module-level ``build()`` returning
a populated ``Viewer`` or ``OrthoViewer``.  Everything else is up to you.

Two rules, both of which follow from the script running with no display:

1. **Do not call** ``launch`` / ``show`` / ``display``.  They open a window and
   block; the capture script wants the viewer object, not a running app.
2. **Do not add a canvas or reslice.**  The script does both -- it gives every
   scene a canvas (data loading is planned per canvas, so a scene without one
   loads nothing), drives the slicer to quiescence, and fits the cameras
   before capturing.

``gui="offscreen"`` is the natural choice here because nothing is ever
displayed, but it is not required: capture always renders offscreen, so a
target that builds a ``gui="qt"`` viewer is captured identically.
"""

import numpy as np

from cellier.convenience import Viewer
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.visuals._image_memory import InMemoryImageSingleAppearance

#: Volume shape, in (z, y, x).
SHAPE = (64, 96, 96)

#: Sphere centres (z, y, x) and radii, in voxels.  Overlapping and touching
#: spheres give ambient occlusion some creases to darken, which is what makes
#: ``--frames converged`` visibly different from the default single frame.
SPHERES = [
    ((32, 34, 34), 18.0),
    ((32, 62, 52), 14.0),
    ((24, 40, 68), 10.0),
]


def _make_volume() -> np.ndarray:
    """Return a float32 volume holding :data:`SPHERES`, smoothly falling off.

    A soft edge rather than a hard one: an isosurface through a binary volume
    is a staircase, and the point of the example is to look at a render.
    """
    grids = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in SHAPE], indexing="ij")
    volume = np.zeros(SHAPE, dtype=np.float32)
    for centre, radius in SPHERES:
        distance = np.sqrt(sum((g - c) ** 2 for g, c in zip(grids, centre)))
        # 1 at the centre, 0 a couple of voxels outside the nominal radius.
        volume = np.maximum(volume, np.clip((radius - distance) / 3.0, 0.0, 1.0))
    return volume


def build() -> Viewer:
    """Build the viewer ``cellier.convenience.capture`` will capture."""
    viewer = Viewer(spatial_axes("z", "y", "x"), gui="offscreen")
    viewer.add_image(
        data=ImageMemoryStore(data=_make_volume(), name="spheres"),
        single=InMemoryImageSingleAppearance(
            color_map="magma",
            clim=(0.0, 1.0),
            # Isosurface rather than the default "mip": MIP renders the whole
            # volume box opaque, and ambient occlusion needs surfaces before it
            # has anything to darken.
            render_mode="iso",
            iso_threshold=0.5,
        ),
    )
    # What makes --frames converged worth passing: AO takes one noisy sample
    # per frame, and the temporal accumulator is what averages the per-frame
    # kernel rotation away.  Without AO the image is identical on the first
    # frame and "converged" just draws 44 of them for the same picture.
    viewer.ambient_occlusion_enabled = True
    # Without this the scene displays two axes -- a single z slice -- and the
    # picture is a flat cross-section rather than a volume.
    viewer.set_displayed_dimensions(("z", "y", "x"))
    return viewer
