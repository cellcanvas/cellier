#!/usr/bin/env python
"""Demo: multiscale OME-Zarr label volume with 2D/3D toggle and salt control.

Usage:
    # Create a synthetic example dataset and open it:
    uv run python scripts/v2/multiscale_labels.py --make-example /tmp/labels_demo.ome.zarr

    # Open an existing OME-Zarr label group:
    uv run python scripts/v2/multiscale_labels.py /path/to/seg.ome.zarr/labels/cells

The positional argument should be the URI to the label group
(either a local path or a remote URI supported by TensorStore).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import PySide6.QtAsyncio as QtAsyncio
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellier.controller import CellierController
from cellier.scene.dims import CoordinateSystem
from cellier.visuals import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)

# World geometry for the synthetic example
_SCALE_ZYX = (4.0, 1.0, 1.0)  # µm per voxel at finest level
_WORLD_SIZE = (800, 800, 800)  # µm
_DS_ZYX = (1, 2, 2)  # anisotropic downscale factor per level
_N_LEVELS = 4
_VOXEL_SIZE_ZYX = tuple(w / s for w, s in zip(_WORLD_SIZE, _SCALE_ZYX))
# → (200, 800, 800) voxels at level 0


def _ensure_uri_scheme(path_or_uri: str) -> str:
    """Convert a local path to a file:// URI if needed.

    If the input already has a scheme (like file://, s3://, etc.),
    return it as-is. Otherwise, convert local paths to file:// URIs
    with absolute paths.
    """
    if "://" in path_or_uri:
        # Already has a scheme
        return path_or_uri
    else:
        # Local path: convert to file:// URI with absolute path
        abs_path = Path(path_or_uri).absolute()
        return f"file://{abs_path}"


def make_example_dataset(output_path: str) -> str:
    """Write a synthetic anisotropic OME-Zarr v0.5 labels dataset.

    Returns the URI of the label group (labels/cells) suitable for
    passing back to ``main()``.
    """
    import yaozarrs.v05 as y5
    import zarr
    from skimage.data import binary_blobs
    from skimage.measure import label as skimage_label

    # ── Generate label data ───────────────────────────────────────────────
    nz = int(_WORLD_SIZE[0] / _SCALE_ZYX[0])  # 200
    ny = int(_WORLD_SIZE[1] / _SCALE_ZYX[1])  # 800
    nx = int(_WORLD_SIZE[2] / _SCALE_ZYX[2])  # 800

    print(f"Generating label data {nz}×{ny}×{nx}...", flush=True)
    blobs_raw = binary_blobs(
        length=max(nz, ny, nx), n_dim=3, volume_fraction=0.04, rng=42
    )
    blobs_raw = blobs_raw[:nz, :ny, :nx]
    data_l0 = skimage_label(blobs_raw).astype(np.int32)

    # ── Build pyramid levels via strided slicing ─────────────────────────
    levels: list[np.ndarray] = [data_l0]
    for i in range(1, _N_LEVELS):
        dz = _DS_ZYX[0] ** i
        dy = _DS_ZYX[1] ** i
        dx = _DS_ZYX[2] ** i
        levels.append(data_l0[::dz, ::dy, ::dx])

    # ── Open zarr groups ─────────────────────────────────────────────────
    root_path = Path(output_path)
    store = zarr.storage.LocalStore(str(root_path))
    root_grp = zarr.open_group(store=store, mode="w", zarr_format=3)
    labels_grp = root_grp.require_group("labels")
    cells_grp = labels_grp.require_group("cells")

    # ── Write arrays ─────────────────────────────────────────────────────
    chunk = (64, 64, 64)
    for i, arr in enumerate(levels):
        cells_grp.create_array(
            str(i),
            data=arr,
            chunks=chunk,
        )
        print(f"  level {i}: shape={arr.shape}", flush=True)

    # ── Build OME metadata ────────────────────────────────────────────────
    axes = [
        y5.SpaceAxis(name="z", unit="micrometer"),
        y5.SpaceAxis(name="y", unit="micrometer"),
        y5.SpaceAxis(name="x", unit="micrometer"),
    ]

    datasets = []
    for i in range(_N_LEVELS):
        scale = [
            _SCALE_ZYX[0] * float(_DS_ZYX[0] ** i),
            _SCALE_ZYX[1] * float(_DS_ZYX[1] ** i),
            _SCALE_ZYX[2] * float(_DS_ZYX[2] ** i),
        ]
        datasets.append(
            y5.Dataset(
                path=str(i),
                coordinateTransformations=[y5.ScaleTransformation(scale=scale)],
            )
        )

    multiscale = y5.Multiscale(axes=axes, datasets=datasets)
    label_image = y5.LabelImage(
        multiscales=[multiscale],
        image_label=y5.ImageLabel(),
    )

    # Write label image metadata to cells group
    cells_grp.attrs["ome"] = label_image.model_dump(by_alias=True, exclude_none=True)

    # Write labels group metadata
    labels_grp.attrs["ome"] = y5.LabelsGroup(labels=["cells"]).model_dump(
        by_alias=True, exclude_none=True
    )

    print(f"Dataset written to {root_path}", flush=True)
    return f"file://{root_path.absolute()}/labels/cells"


def _extract_level0_scale_from_label_group(label_uri: str) -> tuple[float, ...]:
    """Return absolute level-0 voxel scale from a label-group OME metadata block.

    Supports both ``ome.multiscales`` and _legacy top-level ``multiscales``.
    Applies global + dataset-level scale terms, matching datastore parsing.
    """
    import yaozarrs

    group = yaozarrs.open_group(label_uri)
    attrs = group.attrs

    if "ome" in attrs and isinstance(attrs["ome"], dict):
        multiscales = attrs["ome"].get("multiscales")
    else:
        multiscales = attrs.get("multiscales")
    if not multiscales:
        raise ValueError(
            "No multiscales metadata found in label group attributes. "
            "Expected 'ome.multiscales' or 'multiscales'."
        )

    ms = multiscales[0]
    axes = ms.get("axes") or []
    n_axes = len(axes)

    global_scale = [1.0] * n_axes
    for ct in ms.get("coordinateTransformations") or []:
        if ct.get("type") == "scale":
            global_scale = list(ct["scale"])

    datasets = ms.get("datasets") or []
    if not datasets:
        raise ValueError("No datasets found in multiscales metadata.")

    ds0 = datasets[0]
    ds_scale = [1.0] * n_axes
    for ct in ds0.get("coordinateTransformations") or []:
        if ct.get("type") == "scale":
            ds_scale = list(ct["scale"])

    return tuple(g * s for g, s in zip(global_scale, ds_scale))


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


def main(label_uri: str):
    app = QApplication.instance() or QApplication(sys.argv)

    # ── Data store ────────────────────────────────────────────────────────
    from cellier.data import OMEZarrLabelDataStore

    data_store = OMEZarrLabelDataStore.from_path(label_uri)
    ndim = len(data_store.level_shapes[0])

    # ── Controller + scene ───────────────────────────────────────────────
    coord_sys = CoordinateSystem(name="world", axis_labels=("z", "y", "x")[-ndim:])
    controller = CellierController()

    scene = controller.add_scene(
        name="label_scene",
        dim="3d",
        coordinate_system=coord_sys,
        render_modes={"2d", "3d"},
    )

    # ── Transform ─────────────────────────────────────────────────────────
    level0_scale = _extract_level0_scale_from_label_group(label_uri)
    # A transform names the two systems it maps between, so it is built here,
    # where both the store and the scene's world exist.
    transform = controller.data_to_world(scene.id, data_store, scale=level0_scale)

    # ── Appearance + visual ───────────────────────────────────────────────
    appearance = MultiscaleLabelsAppearance(
        colormap_mode="random",
        background_label=0,
        salt=0,
        render_mode="smooth_iso",
        lod_bias=1.0,
        frustum_cull=True,
        force_level=None,
    )
    render_config = MultiscaleLabelRenderConfig(block_size=32)

    visual = controller.add_labels_multiscale(
        data=data_store,
        scene_id=scene.id,
        appearance=appearance,
        name="labels",
        render_config=render_config,
        transform=transform,
    )

    # ── UI ────────────────────────────────────────────────────────────────
    root = QWidget()
    root.setWindowTitle("Multiscale Labels Demo")
    outer_layout = QVBoxLayout(root)
    controller.set_widget_parent(root)

    canvas_widget = controller.add_canvas(scene_id=scene.id)
    canvas_id = next(iter(scene.canvases))
    outer_layout.addWidget(canvas_widget)

    controls = QHBoxLayout()

    _mode = ["3d"]

    toggle_btn = QPushButton("Switch to 2D")
    controls.addWidget(toggle_btn)

    # ── 2D-only: Z slice spinbox ──────────────────────────────────────
    z_depth = data_store.level_shapes[0][0]
    z_label = QLabel("Z slice:")
    z_slice_spin = QSpinBox()
    z_slice_spin.setRange(0, z_depth - 1)
    z_slice_spin.setValue(z_depth // 2)
    z_slice_spin.setVisible(False)  # Hidden initially (start in 3D mode)
    z_label.setVisible(False)
    controls.addWidget(z_label)
    controls.addWidget(z_slice_spin)

    # Track 2D-only widgets for show/hide on toggle
    _widget_2d = [z_label, z_slice_spin]

    salt_label = QLabel("salt:")
    salt_spin = QSpinBox()
    salt_spin.setRange(0, 9999)
    controls.addWidget(salt_label)
    controls.addWidget(salt_spin)

    controls.addStretch()
    outer_layout.addLayout(controls)

    def _fit_camera_for_mode() -> None:
        controller.look_at_visual(visual.id, canvas_id)

    def _apply_dims():
        if _mode[0] == "3d":
            scene.dims.selection.displayed_axes = tuple(range(ndim - 3, ndim))
            scene.dims.selection.slice_indices = {}
        else:
            scene.dims.selection.displayed_axes = (ndim - 2, ndim - 1)
            scene.dims.selection.slice_indices = {ndim - 3: z_slice_spin.value()}
        # Fit camera before requesting data so frustum culling uses a sane view.
        _fit_camera_for_mode()
        controller.reslice_scene(scene.id)

    def _on_z_slice_changed(value: int) -> None:
        """Update dims when Z slice spinbox changes."""
        if _mode[0] == "2d":
            scene.dims.selection.slice_indices = {ndim - 3: value}
            controller.reslice_scene(scene.id)

    def _toggle():
        if _mode[0] == "3d":
            _mode[0] = "2d"
            toggle_btn.setText("Switch to 3D")
            for w in _widget_2d:
                w.setVisible(True)
        else:
            _mode[0] = "3d"
            toggle_btn.setText("Switch to 2D")
            for w in _widget_2d:
                w.setVisible(False)
        _apply_dims()

    toggle_btn.clicked.connect(_toggle)
    z_slice_spin.valueChanged.connect(_on_z_slice_changed)
    salt_spin.valueChanged.connect(lambda v: setattr(visual.appearance, "salt", v))

    # Camera fit on first data delivery
    _camera_fitted: set[str] = set()

    def _on_data_ready(event):
        mode = _mode[0]
        if mode not in _camera_fitted:
            _camera_fitted.add(mode)
            controller.look_at_visual(visual.id, canvas_id)

    controller.on_reslice_completed(visual.id, _on_data_ready, owner_id=controller._id)

    root.resize(960, 720)
    root.show()

    # Start from a camera pose that sees the volume before first 3D reslice.
    _fit_camera_for_mode()
    QTimer.singleShot(0, controller.reslice_all)

    QtAsyncio.run(handle_sigint=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multiscale labels demo viewer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--make-example",
        metavar="PATH",
        help="Create a synthetic OME-Zarr v0.5 labels dataset at PATH then open it.",
    )
    group.add_argument(
        "uri",
        nargs="?",
        help="URI of an existing OME-Zarr label group to open.",
    )
    args = parser.parse_args()

    if args.make_example:
        label_uri = make_example_dataset(args.make_example)
    else:
        if args.uri is None:
            parser.error("Provide a URI or use --make-example PATH.")
        label_uri = _ensure_uri_scheme(args.uri)

    main(label_uri)
