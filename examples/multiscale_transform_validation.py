"""Check that multiscale images and labels stay aligned across pyramid levels.

This example shows multiscale image and labels visuals that are overlaid.
Both contain several spheres. The viewer has a spinbox to change the current
scale level being viewed. As the level is changed, the level of detail should
change, but the spheres should remain centered and should not appear to translate.

The images and labels are stored in a temporary directory that is deleted
when the viewer is closed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import zarr
from qtpy import QtWidgets
from skimage.measure import label

from cellier.convenience import Viewer, axis_values_from_viewer, launch, spatial_axes
from cellier.convenience.gui import build_canvas_widget
from cellier.data import OMEZarrImageDataStore, OMEZarrLabelDataStore
from cellier.transform import AffineTransform
from cellier.visuals import (
    MultiscaleImageAppearance,
    MultiscaleImageSingleAppearance,
    MultiscaleLabelsAppearance,
)

# Dataset parameters
AXIS_NAMES = ("z", "y", "x")

#: Level-0 shape, (z, y, x).
SHAPE = (40, 160, 240)

#: Level-0 voxel size in micrometers, (z, y, x).
VOXEL_SIZE = (2.0, 0.6, 0.4)

#: Cumulative downscale factor of each level relative to level 0, (z, y, x).
LEVEL_FACTORS = (
    (1, 1, 1),
    (1, 2, 2),
    (2, 4, 2),
    (2, 8, 4),
)

#: Sphere centres in level-0 voxels (z, y, x), and radii in micrometers.
SPHERES = (
    ((10, 40, 60), 12.0),
    ((20, 80, 120), 18.0),
    ((28, 120, 180), 10.0),
    ((12, 120, 70), 8.0),
    ((30, 36, 190), 14.0),
)

#: Image intensity inside a sphere; outside is 0.
SPHERE_VALUE = 200

#: The isosurface and the labels threshold, halfway between 0 and 200.
THRESHOLD = 100.0

LABELS_NAME = "spheres"


# Helpers to make the data
def _check_centres() -> None:
    """Check that every sphere centre is a symmetry point of every level.

    Along an axis with cumulative factor ``F`` the kept level-0 voxels are
    ``F // 2 + F * i``.  That lattice is symmetric about ``c`` when ``c`` is
    on a sample or halfway between two, i.e. ``(c - F // 2) % F`` is ``0``
    or ``F / 2``.  Then the samples of a sphere centred at ``c`` are
    symmetric about ``c`` and the coarse sphere has the same centre.
    """
    for centre, _ in SPHERES:
        for factors in LEVEL_FACTORS:
            for axis, (c, f) in enumerate(zip(centre, factors)):
                phase = (c - f // 2) % f
                if phase != 0 and 2 * phase != f:
                    raise ValueError(
                        f"Sphere centre {centre} is not symmetric under factor "
                        f"{f} on axis {AXIS_NAMES[axis]!r}: a coarse level "
                        "would move it."
                    )


def _make_image() -> np.ndarray:
    """Level-0 image: spheres (round in micrometers) at ``SPHERE_VALUE``."""
    z, y, x = np.indices(SHAPE, dtype=np.float64)
    image = np.zeros(SHAPE, dtype=np.uint8)
    for (cz, cy, cx), radius in SPHERES:
        distance_sq = (
            ((z - cz) * VOXEL_SIZE[0]) ** 2
            + ((y - cy) * VOXEL_SIZE[1]) ** 2
            + ((x - cx) * VOXEL_SIZE[2]) ** 2
        )
        image[distance_sq <= radius**2] = SPHERE_VALUE
    return image


def _downsample(array: np.ndarray, factors: tuple[int, ...]) -> np.ndarray:
    """Nearest-neighbour downsampling by offset striding."""
    return array[tuple(slice(f // 2, None, f) for f in factors)]


def _level_transforms(factors: tuple[int, ...]) -> list[dict]:
    """A level's OME-Zarr ``coordinateTransformations``: scale, then translation."""
    return [
        {"type": "scale", "scale": [f * s for f, s in zip(factors, VOXEL_SIZE)]},
        {
            "type": "translation",
            "translation": [(f // 2) * s for f, s in zip(factors, VOXEL_SIZE)],
        },
    ]


def _multiscales(name: str) -> list[dict]:
    """The ``multiscales`` block shared by the image and the labels."""
    return [
        {
            "name": name,
            "axes": [
                {"name": axis, "type": "space", "unit": "micrometer"}
                for axis in AXIS_NAMES
            ],
            "datasets": [
                {"path": str(level), "coordinateTransformations": _level_transforms(f)}
                for level, f in enumerate(LEVEL_FACTORS)
            ],
        }
    ]


def _write_pyramid(group: zarr.Group, level0: np.ndarray) -> None:
    """Write one array per level of *level0*'s pyramid into *group*."""
    for level, factors in enumerate(LEVEL_FACTORS):
        data = _downsample(level0, factors)
        array = group.create_array(
            str(level),
            shape=data.shape,
            dtype=data.dtype,
            chunks=tuple(min(n, 64) for n in data.shape),
        )
        array[...] = data


def write_dataset(root_path: Path) -> Path:
    """Write the OME-Zarr image, with its labels inside it, under *root_path*.

    Returns
    -------
    Path
        The image group, ``<root_path>/spheres.ome.zarr``.  The labels are
        its ``labels/spheres`` group.
    """
    _check_centres()
    image = _make_image()
    # The labels pyramid is downsampled from the level-0 labels, not labelled
    # per level, so label ids and positions agree at every level.
    labels = label(image > THRESHOLD).astype(np.int32)

    image_path = root_path / "spheres.ome.zarr"
    root = zarr.open_group(image_path, mode="w", zarr_format=3)
    root.attrs["ome"] = {"version": "0.5", "multiscales": _multiscales("spheres")}
    _write_pyramid(root, image)

    labels_root = root.create_group("labels")
    labels_root.attrs["ome"] = {"version": "0.5", "labels": [LABELS_NAME]}
    labels_group = labels_root.create_group(LABELS_NAME)
    labels_group.attrs["ome"] = {
        "version": "0.5",
        "image-label": {},
        "multiscales": _multiscales(LABELS_NAME),
    }
    _write_pyramid(labels_group, labels)
    return image_path


# Helpers to make the Viewer
def _to_world(store, world) -> AffineTransform:
    """Data -> world: the store's level-0 voxel size and origin, from its metadata."""
    store_cs = store.data_coordinate_systems[0]
    return AffineTransform.from_axis_map(
        store_cs,
        world,
        axis_map={name: name for name in AXIS_NAMES},
        scale=dict(zip(AXIS_NAMES, store.physical_scale)),
        translation=dict(zip(AXIS_NAMES, store.physical_translation)),
        name="to_world",
    )


def build_viewer(image_path: Path, gui: str = "qt"):
    """Build the viewer with the image and labels visuals.

    Parameters
    ----------
    image_path : Path
        The image group written by :func:`write_dataset`.
    gui : str
        The viewer's GUI toolkit.  ``"offscreen"`` builds it headless.

    Returns
    -------
    tuple
        ``(viewer, image_visual, labels_visual)``.
    """
    viewer = Viewer(
        spatial_axes(*AXIS_NAMES), dim="3d", render_modes={"2d", "3d"}, gui=gui
    )
    world = viewer.scene.dims.world_coordinate_system

    image_store = OMEZarrImageDataStore.from_path(image_path.as_uri(), name="image")
    labels_store = OMEZarrLabelDataStore.from_path(
        (image_path / "labels" / LABELS_NAME).as_uri(), name="labels"
    )

    image_visual = viewer.add_image_multiscale(
        image_store,
        appearance=MultiscaleImageAppearance(force_level=1),
        name="image",
        transform=_to_world(image_store, world),
        # "gray" runs black -> white ("grays" is ColorBrewer's white -> black).
        # The clim tops out at the threshold so the isosurface draws white.
        single=MultiscaleImageSingleAppearance(
            color_map="gray",
            clim=(0.0, THRESHOLD),
            render_mode="iso",
            iso_threshold=THRESHOLD,
        ),
    )
    labels_visual = viewer.add_labels_multiscale(
        labels_store,
        appearance=MultiscaleLabelsAppearance(force_level=1),
        name="labels",
        transform=_to_world(labels_store, world),
    )
    return viewer, image_visual, labels_visual


def _level_description(level: int | None) -> str:
    """A level's voxel size and translation, as written to the metadata."""
    if level is None:
        return "level of detail chosen by the camera"
    [scale, translation] = _level_transforms(LEVEL_FACTORS[level])
    fmt = ", ".join
    return (
        f"voxel size (z, y, x): {fmt(f'{v:g}' for v in scale['scale'])} um\n"
        f"translation: {fmt(f'{v:g}' for v in translation['translation'])} um"
    )


def build_window(viewer, image_visual, labels_visual) -> QtWidgets.QWidget:
    """Construct the viewer window.

    The window contains the canvas, visibility checkboxes, and level selector.
    """
    canvas_widget = build_canvas_widget(viewer, axis_values_from_viewer(viewer))

    image_visible = QtWidgets.QCheckBox("image visible")
    image_visible.setChecked(True)
    image_visible.toggled.connect(
        lambda checked: setattr(image_visual.appearance, "visible", checked)
    )
    labels_visible = QtWidgets.QCheckBox("labels visible")
    labels_visible.setChecked(True)
    labels_visible.toggled.connect(
        lambda checked: setattr(labels_visual.appearance, "visible", checked)
    )

    # Shown 0-based, like the zarr level paths; force_level is 1-based.
    # The minimum, -1, is "auto": no forced level.
    level_box = QtWidgets.QSpinBox()
    level_box.setRange(-1, len(LEVEL_FACTORS) - 1)
    level_box.setSpecialValueText("auto")
    level_box.setValue(0)
    level_info = QtWidgets.QLabel(_level_description(0))

    def _set_level(value: int) -> None:
        level = None if value < 0 else value
        force_level = None if level is None else level + 1
        image_visual.appearance.force_level = force_level
        labels_visual.appearance.force_level = force_level
        level_info.setText(_level_description(level))

    level_box.valueChanged.connect(_set_level)

    controls = QtWidgets.QWidget()
    column = QtWidgets.QVBoxLayout(controls)
    column.addWidget(image_visible)
    column.addWidget(labels_visible)
    column.addWidget(QtWidgets.QLabel("scale level"))
    column.addWidget(level_box)
    column.addWidget(level_info)
    column.addStretch(1)
    # The description changes length with the level; size the column for the
    # widest one so the canvas beside it keeps its width.
    widest = max(
        level_info.fontMetrics().boundingRect(line).width()
        for level in (None, *range(len(LEVEL_FACTORS)))
        for line in _level_description(level).splitlines()
    )
    level_info.setMinimumWidth(widest)
    controls.setFixedWidth(controls.sizeHint().width())

    window = QtWidgets.QWidget()
    window.setWindowTitle("multiscale transform validation")
    row = QtWidgets.QHBoxLayout(window)
    row.addWidget(canvas_widget.widget, stretch=1)
    row.addWidget(controls)
    window.resize(1100, 750)
    window.canvas = canvas_widget
    return window


def main() -> None:
    """Write the dataset, build the viewer and launch it."""
    with tempfile.TemporaryDirectory() as tmp:
        image_path = write_dataset(Path(tmp))
        viewer, image_visual, labels_visual = build_viewer(image_path)
        launch(viewer, build_window(viewer, image_visual, labels_visual))


if __name__ == "__main__":
    main()
