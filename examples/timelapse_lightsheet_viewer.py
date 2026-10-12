# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cellier[graph,pyside,remote]",
# ]
# ///

"""View a timelapse light-sheet organoid dataset in 2D and 3D.

The dataset is movie of a growing organoid with a sgmentation and tracking graph
from the Liberali Lab (https://zenodo.org/records/22078388).

World coordinates are (t, c, z, y, x), with t in seconds and everything
and z/y/x in micrometers.  The scene contains:

- the two-channel deconvolved volume (``deconv.ome.zarr``)
- the nucleus segmentation (deconv.ome.zarr/labels/nucleus)
- the nucleus lineage tree (deconv.ome.zarr/tracks/nucleus.geff)
- a gray bounding box around everything.

To run the example, download the dataset and pass the path of the ``001-full`` folder:

    python examples/timelapse_lightsheet_viewer.py /path/to/001-full
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from uuid import uuid4

import numpy as np
import pyarrow.parquet as pq
from cmap import Colormap

from cellier.convenience import (
    AppearanceControls,
    FieldControls,
    ImageControl,
    Layout,
    LevelOfDetailControl,
    LoadingConfigControl,
    LoadingIndicator,
    OverlayControls,
    TrailControl,
    Viewer,
    VStack,
    axis_values_from_viewer,
    run,
    spatial_axes,
)
from cellier.convenience.gui import build_canvas_widget
from cellier.data import (
    GraphMemoryStore,
    OMEZarrImageDataStore,
    OMEZarrLabelDataStore,
)
from cellier.render import (
    OutlineConfig,
    OutlineLayerConfig,
    RenderManagerConfig,
)
from cellier.scene import BackgroundAppearance
from cellier.transform import (
    AffineTransform,
    Axis,
    AxisCoordinates,
    ByDimensionTransform,
    CoordinateSystem,
    DataCoordinateSystem,
    NonUniformAxisTransform,
)
from cellier.visuals import (
    GraphAppearance,
    MultiscaleImageAppearance,
    MultiscaleImageChannelAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
    SceneBoundingBox,
    SceneBoundingBoxAppearance,
    TrailConfig,
    VisualOutline,
)

# The GEFF axis that holds the frame index.  It is distinct from the ``t`` node
# property, which is a separately computed elapsed-seconds value.
GRAPH_TIME_AXIS = "t_idx"

# How much history the graph's trail shows, in world seconds.
TRAIL_BEFORE = 4800.0

# Outline colors (RGBA), one palette entry per outlined visual.  The labels
# use slot 1 (the first entry) and the tracks use slot 2 (the second).
LABEL_OUTLINE_COLOR = (0.0, 0.0, 0.0, 1.0)
TRACK_OUTLINE_COLOR = (0.0, 0.0, 0.0, 1.0)

# Outline thickness in internal pixels (the renderer halves this on screen).  The
# labels are outlined inward.  The graph is outlined outward, so its thin edges and
# markers are not consumed.
LABEL_OUTLINE_THICKNESS = 2
TRACK_OUTLINE_THICKNESS = 2

# Bounding box line color (RGBA) and width in screen pixels.
BOX_COLOR = (0.5, 0.5, 0.5, 1.0)
BOX_THICKNESS = 1.5

# Composite-mode colormaps by channel index.  Hex strings, not RGB tuples: cmap reads
# a tuple of 0/1 integers as 0-255.
CHANNEL_COLORMAPS = {
    0: Colormap(["#ffffff", "#00ff00"], name="white_green"),
    1: Colormap(["#ffffff", "#0000ff"], name="white_blue"),
}

# Contrast limits of the image, and the range of the contrast sliders.
CLIM = (0.0, 4000.0)
CLIM_RANGE = (0.0, 65535.0)

# Side length of the 3D bricks on thd shader, in voxels.
BLOCK_SIZE = 64

# The narrowest the right (appearance) dock may be, in logical pixels.
RIGHT_DOCK_MIN_WIDTH = 340


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line and check the dataset folder.

    Parameters
    ----------
    argv : list[str] or None
        The arguments, without the program name.  ``sys.argv`` when ``None``.

    Returns
    -------
    argparse.Namespace
        ``dataset_root``, the resolved path of the ``001-full`` folder.
    """
    parser = argparse.ArgumentParser(
        description=(
            "View the Zenodo light-sheet organoid timelapse "
            "(https://zenodo.org/records/22078388)."
        )
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="path of the dataset folder (the one that holds deconv.ome.zarr)",
    )
    args = parser.parse_args(argv)
    args.dataset_root = args.dataset_root.expanduser().resolve()

    paths = _dataset_paths(args.dataset_root)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        parser.error(
            "this does not look like the light-sheet dataset; missing:\n  "
            + "\n  ".join(missing)
            + "\nDownload it from https://zenodo.org/records/22078388."
        )
    return args


def _dataset_paths(dataset_root: Path) -> dict[str, Path]:
    """The paths of the dataset's parts, by name."""
    deconv = dataset_root / "deconv.ome.zarr"
    return {
        "image": deconv,
        "labels": deconv / "labels" / "nucleus",
        "tracks": deconv / "tracks" / "nucleus.geff",
        "timestamps": deconv / "tables" / "timestamps" / "table.parquet",
    }


# ---------------------------------------------------------------------------
# Timestamps -> a shared non-uniform time axis
# ---------------------------------------------------------------------------


def _load_relative_seconds(path: Path) -> tuple[float, ...]:
    """Read the timestamps table and return seconds since the first frame.

    Parameters
    ----------
    path : Path
        Path to the parquet table, which holds ``t_idx`` (frame index) and
        ``posix_timestamp`` (POSIX seconds).

    Returns
    -------
    tuple[float, ...]
        One entry per frame in frame order, starting at 0.0.
    """
    table = pq.read_table(path)
    t_idx = table.column("t_idx").to_numpy()
    posix = table.column("posix_timestamp").to_numpy()
    order = np.argsort(t_idx)
    t_idx = t_idx[order]
    posix = posix[order]
    if not np.array_equal(t_idx, np.arange(t_idx.size)):
        raise ValueError(
            f"Expected t_idx to be a dense 0..{t_idx.size - 1} range; "
            f"got {t_idx.tolist()}."
        )
    return tuple(float(value) for value in (posix - posix[0]))


# ---------------------------------------------------------------------------
# Pyramid levels -> aligned edges
# ---------------------------------------------------------------------------


def _align_level_edges(store: OMEZarrImageDataStore | OMEZarrLabelDataStore) -> None:
    """Translate each pyramid level so its outer edges coincide with level 0's.

    A level whose voxels are ``f`` level-0 voxels wide, and which covers the same
    extent as level 0, has its voxel centers at ``f * i + (f - 1) / 2`` in level-0
    voxel units.  The dataset's metadata holds the scale ``f`` but not this
    translation, so the store places the coarse voxels half a coarse voxel too close
    to the origin.  The translation is ``(f - 1) / 2`` on every axis, which is zero
    on the axes that are not downsampled (t, c, z).

    The dataset does not include these translations so we calculate them ourselves.

    Parameters
    ----------
    store : OMEZarrImageDataStore or OMEZarrLabelDataStore
        The store to correct, in place.
    """
    systems = store.data_coordinate_systems
    # Level-k voxel size in level-0 voxels, one row per level.
    factors = np.asarray(store.level_scales, dtype=float)
    shapes = np.asarray(store.level_shapes, dtype=float)
    if not np.allclose(shapes * factors, shapes[0], rtol=1e-6):
        warnings.warn(
            f"The levels of {store.name!r} do not cover level 0's extent; "
            "their translations are left as the metadata gives them.",
            stacklevel=2,
        )
        return
    store.set_data_coordinate_systems(
        list(systems),
        [
            AffineTransform.from_axis_map(
                system,
                systems[0],
                axis_map={a.id: b.id for a, b in zip(system.axes, systems[0].axes)},
                scale={a.id: float(f) for a, f in zip(system.axes, factors[level])},
                translation={
                    a.id: float((f - 1.0) / 2.0)
                    for a, f in zip(system.axes, factors[level])
                },
            )
            for level, system in enumerate(systems)
        ],
    )


# ---------------------------------------------------------------------------
# GEFF graph -> GraphMemoryStore
# ---------------------------------------------------------------------------


def _load_track_store(path: Path) -> GraphMemoryStore:
    """Read the nucleus lineage graph into a store with a discrete time axis.

    The coordinate system is built here rather than derived from the file's axis
    metadata.  A GEFF file cannot say that a column holds frame indices, so a
    derived system would make every axis continuous.  ``t_idx`` is made discrete so
    the graph's slice snaps to the nearest frame, in step with the image and labels.
    ``z``/``y``/``x`` are measured world positions and stay continuous.

    Parameters
    ----------
    path : Path
        Path to the ``.geff`` group.

    Returns
    -------
    GraphMemoryStore
    """
    axes = (
        Axis(name=GRAPH_TIME_AXIS, axis_type="time", sampling="discrete"),
        *(
            Axis(name=name, axis_type="space", unit="micrometer")
            for name in ("z", "y", "x")
        ),
    )
    return GraphMemoryStore.from_geff(
        path,
        axis_names=[axis.name for axis in axes],
        data_coordinate_system=DataCoordinateSystem(
            name="nucleus_tracks", axes=axes, datastore_id=uuid4()
        ),
        name="nucleus_tracks",
    )


# ---------------------------------------------------------------------------
# data -> world transforms
# ---------------------------------------------------------------------------


def _build_transform(
    store_cs: DataCoordinateSystem,
    world: CoordinateSystem,
    time_axis_name: str,
    spatial_scale: dict[str, float],
    time_block: NonUniformAxisTransform,
    broadcast_axis_names: tuple[str, ...] = (),
) -> ByDimensionTransform:
    """Build one store's ``data -> world`` transform.

    Every axis but time is a plain scale.  The time axis goes through *time_block*,
    the same object for every store, so the image, the labels and the graph agree
    about when a frame is current.

    Parameters
    ----------
    store_cs : DataCoordinateSystem
        The store's level-0 coordinate system.
    world : CoordinateSystem
        The scene's world coordinate system.
    time_axis_name : str
        The store's own name for its time axis (``"t"`` or ``"t_idx"``).
    spatial_scale : dict[str, float]
        World units per data unit for every axis but time, keyed by the store's own
        axis names.
    time_block : NonUniformAxisTransform
        The shared frame-index to world-seconds lookup.
    broadcast_axis_names : tuple[str, ...]
        World axes the store has no data for (``"c"`` for the labels and graph).

    Returns
    -------
    ByDimensionTransform
    """
    return ByDimensionTransform.from_axis_map(
        store_cs,
        world,
        axis_map={time_axis_name: "t", **{name: name for name in spatial_scale}},
        scale=spatial_scale,
        broadcast_output_axes=list(broadcast_axis_names),
        axis_transforms={time_axis_name: time_block},
        name="to_world",
    )


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------


def build_viewer(dataset_root: Path) -> Viewer:
    """Build the viewer, with its three visuals and the bounding box.

    Parameters
    ----------
    dataset_root : Path
        The dataset folder, which holds ``deconv.ome.zarr``.

    Returns
    -------
    Viewer
    """
    paths = _dataset_paths(dataset_root)

    # World: t in seconds, c, then z/y/x in micrometers.
    viewer = Viewer(
        axes=[
            Axis(name="t", axis_type="time", unit="second"),
            Axis(name="c", axis_type="channel"),
            *spatial_axes("z", "y", "x"),
        ],
        displayed_axes=("z", "y", "x"),
        render_modes={"2d", "3d"},
        # Outlines are enabled here, at construction, because the render target
        # that holds the per-label outline key is allocated with the canvas.  The
        # selection layer carries the palette colors, so the boundaries layer is off.
        render_config=RenderManagerConfig(
            outline=OutlineConfig(
                enabled=True,
                boundaries=OutlineLayerConfig(enabled=False),
                selection=OutlineLayerConfig(
                    inward_thickness=LABEL_OUTLINE_THICKNESS,
                    outward_thickness=TRACK_OUTLINE_THICKNESS,
                ),
                inner_thickness=0,
                palette=[LABEL_OUTLINE_COLOR, TRACK_OUTLINE_COLOR],
            )
        ),
    )
    viewer.background = BackgroundAppearance(mode="uniform", color=(1.0, 1.0, 1.0, 1.0))
    world = viewer.scene.dims.world_coordinate_system

    # The stores.  Every chunk is a whole z-stack, much larger than a brick, so the
    # default tensorstore chunk cache is what stops each brick re-decompressing it.
    image_store = OMEZarrImageDataStore.from_path(
        paths["image"].as_uri(), name="deconv"
    )
    label_store = OMEZarrLabelDataStore.from_path(
        paths["labels"].as_uri(), name="nucleus_labels"
    )
    track_store = _load_track_store(paths["tracks"])
    # The metadata has no level translations; see "Pyramid levels" above.
    _align_level_edges(image_store)
    _align_level_edges(label_store)

    image_cs = image_store.data_coordinate_systems[0]
    label_cs = label_store.data_coordinate_systems[0]
    track_cs = track_store.data_coordinate_systems[0]

    # One non-uniform time axis, shared by all three visuals.
    time_block = NonUniformAxisTransform(
        coordinates=AxisCoordinates(values=_load_relative_seconds(paths["timestamps"])),
        input_coordinate_system=CoordinateSystem(
            name="t", axes=(image_cs.axis_by_name("t"),)
        ).id,
        output_coordinate_system=CoordinateSystem(
            name="t", axes=(world.axis_by_name("t"),)
        ).id,
    )

    # The physical scale comes from the OME-Zarr metadata the stores parsed.
    image_axis_names = image_cs.axis_names()  # (t, c, z, y, x)
    image_scale = dict(zip(image_axis_names, image_store.physical_scale))
    label_scale = dict(zip(label_cs.axis_names(), label_store.physical_scale))

    image_transform = _build_transform(
        image_cs,
        world,
        time_axis_name="t",
        spatial_scale={name: image_scale[name] for name in ("c", "z", "y", "x")},
        time_block=time_block,
    )
    labels_transform = _build_transform(
        label_cs,
        world,
        time_axis_name="t",
        spatial_scale={name: label_scale[name] for name in ("z", "y", "x")},
        time_block=time_block,
        broadcast_axis_names=("c",),
    )
    # The GEFF positions are already world positions, so there is no spatial scale.
    graph_transform = _build_transform(
        track_cs,
        world,
        time_axis_name=GRAPH_TIME_AXIS,
        spatial_scale={"z": 1.0, "y": 1.0, "x": 1.0},
        time_block=time_block,
        broadcast_axis_names=("c",),
    )

    viewer.add_image_multiscale(
        data=image_store,
        appearance=MultiscaleImageAppearance(
            depth_write=False,
            transparency_mode="multiply",
            coarsest_while_moving_2d=True,
        ),
        name="deconv",
        render_config=MultiscaleImageRenderConfig(block_size=BLOCK_SIZE),
        transform=image_transform,
        channel_axis=image_axis_names.index("c"),
        composite=True,
        single=MultiscaleImageSingleAppearance(
            color_map="grays",
            clim=CLIM,
            render_mode="mip",
            iso_threshold=0.2,
        ),
        channels={
            index: MultiscaleImageChannelAppearance(
                color_map=color_map, clim=CLIM, render_mode="mip"
            )
            for index, color_map in CHANNEL_COLORMAPS.items()
        },
        pick_write=False,
        controls=[
            LoadingIndicator(),
            ImageControl(
                fields=[
                    "visible",
                    "color_map",
                    "clim",
                    "render_mode",
                    "iso_threshold",
                    "attenuation",
                ],
                clim_range=CLIM_RANGE,
                decimals=0,
                # The channel names come from the image's omero metadata.
                channel_labels=dict(enumerate(image_store.channel_labels or ())),
            ),
            LevelOfDetailControl(),
            LoadingConfigControl(),
        ],
    )
    viewer.add_labels_multiscale(
        data=label_store,
        appearance=MultiscaleLabelsAppearance(
            render_order=-1, coarsest_while_moving_2d=True
        ),
        name="nucleus",
        render_config=MultiscaleLabelRenderConfig(block_size=BLOCK_SIZE),
        transform=labels_transform,
        outline=VisualOutline(slot=1, placement="inward"),
        # Every label's boundary, in the color of the visual's slot.
        outline_mode="all_boundaries",
        controls=[
            LoadingIndicator(),
            FieldControls(["visible"]),
            LevelOfDetailControl(),
        ],
    )
    viewer.add_graph(
        data=track_store,
        appearance=GraphAppearance(
            node_color=(1.0, 0.85, 0.2, 1.0),
            node_size=8.0,
            node_size_space="screen",
            edge_color=(1.0, 0.85, 0.2, 0.8),
            render_order=1,
        ),
        name="tracks",
        transform=graph_transform,
        # Keyed by the graph store's own data-axis index (0 is t_idx), not by a
        # world axis.
        trail={0: TrailConfig(before=TRAIL_BEFORE, after=0.0, fade=True)},
        outline=VisualOutline(slot=2, placement="outward"),
        controls=[
            FieldControls(
                [
                    "node_visible",
                    "node_color",
                    "node_size",
                    "node_size_space",
                    "edge_visible",
                    "edge_color",
                    "edge_thickness",
                    "edge_thickness_space",
                ]
            ),
            TrailControl(),
        ],
    )
    viewer.add_scene_overlay(
        SceneBoundingBox(
            name="bounding box",
            appearance=SceneBoundingBoxAppearance(
                color=BOX_COLOR, thickness=BOX_THICKNESS
            ),
        )
    )
    return viewer


def build_layout(viewer: Viewer) -> Layout:
    """Lay out the canvas and the control docks for *viewer*.

    The canvas carries the dims sliders.  ``axis_values_from_viewer`` builds each one
    from the extent of every visual mapped through its own transform, so ``t`` steps
    frame by frame in seconds and ``c`` is labelled with the channel names.  Only
    ``c`` gets tick marks: ``t`` has hundreds of values and a mark per value stalls
    the slider.

    The visuals' controls and the bounding box are in collapsible sections in the
    right dock.

    Parameters
    ----------
    viewer : Viewer
        The viewer to lay out.

    Returns
    -------
    Layout
    """
    return Layout(
        center=build_canvas_widget(
            viewer, axis_values_from_viewer(viewer, draw_ticks=["c"])
        ),
        right_dock=VStack(
            items=[
                AppearanceControls(presentation="collapsible_sections"),
                OverlayControls(),
            ]
        ),
        right_dock_min_width=RIGHT_DOCK_MIN_WIDTH,
    )


def main(argv: list[str] | None = None) -> None:
    """Build and show the viewer."""
    args = parse_args(argv)
    viewer = build_viewer(args.dataset_root)
    run(viewer, build_layout(viewer), fit="ready")


if __name__ == "__main__":
    sys.exit(main())
