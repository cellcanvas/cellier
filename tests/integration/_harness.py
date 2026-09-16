"""Drivers and the parameterised case matrix for the golden baseline.

Every render-layer visual family is driven here to produce, with no canvas
and no GPU device:

* ``node_local_matrix`` -- the ``(4, 4)`` matrix written to ``node.local.matrix``
  by the family's ``_update_node_matrix`` / ``get_node_for_dims``;
* ``selections`` -- the per-axis datastore selection tuples produced by
  ``build_slice_request`` and ``build_slice_request_2d`` (``axis_selections``
  for image / label families, ``slice_indices`` for geometry families),
  sorted for determinism;
* ``scale_indices`` -- the ``scale_index`` of each request;
* ``geometry_indices`` -- for the geometry families, the surviving original
  element indices from a real ``get_data`` call.

Nothing here imports from the v1 transform package.  Phase 0 changes no
behaviour; it only records it.

The three sites where the design (section 3.6) predicts a *deliberate*
behaviour change in a later phase are marked ``XFAIL_CASES`` with the
phase that owns the flip, so that phase turning them green is an
``xfail -> xpass`` and not a silent divergence.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from cellier._state import AxisAlignedSelectionState, DimsState
from cellier.render._spaces import build_render_spaces
from cellier.transform import (
    AffineTransform,
    Axis,
    ConvexRegion,
    DataCoordinateSystem,
    RegionSelection,
    RenderedCoordinateSystem,
    VisualCoordinateSystem,
    WorldCoordinateSystem,
)
from cellier.visuals import InMemoryImageSingleAppearance
from tests._v2 import level_transforms

# ---------------------------------------------------------------------------
# JSON canonicalisation
# ---------------------------------------------------------------------------

_FLOAT_NDIGITS = 12


def _canon_scalar(x: Any) -> Any:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        val = float(x)
        if val == 0.0:  # normalise -0.0
            return 0.0
        return round(val, _FLOAT_NDIGITS)
    return x


def canon(obj: Any) -> Any:
    """Recursively convert *obj* to a JSON-stable structure.

    Rounds floats to a fixed number of digits (the node matrices are
    float32 today, so 12 digits is well inside their precision and keeps
    the comparison from tripping on the last ULP), turns numpy arrays and
    tuples into lists, and leaves ``None`` / ``str`` alone.  UUIDs are
    never passed in -- callers extract only the fields that do not vary
    per run.
    """
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, np.ndarray):
        return canon(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canon(v) for v in obj]
    return _canon_scalar(obj)


def matrix_to_list(matrix: Any) -> list[list[float]]:
    """A pygfx ``node.local.matrix`` (row-major 4x4) as a nested list."""
    arr = np.asarray(matrix, dtype=np.float64)
    return canon(arr)


def selection_key(selections: list[Any]) -> list[Any]:
    """Sort a list of per-axis selection tuples for run-to-run determinism.

    In-memory families emit one request; multiscale families emit one per
    brick and the emission order depends on a distance sort that is stable
    but not meaningful to the baseline.  Sorting by the repr keeps mixed
    ``int`` / ``[start, stop]`` entries orderable.
    """
    return sorted(selections, key=repr)


# ---------------------------------------------------------------------------
# Transform specs -- the per-axis numbers for the case matrix
# ---------------------------------------------------------------------------


#: How the harness types a synthetic axis, by its conventional name.
_AXIS_TYPE = {"t": "time", "c": "channel"}


def _axes(labels: tuple[str, ...]) -> tuple[Axis, ...]:
    return tuple(
        Axis(name=label, axis_type=_AXIS_TYPE.get(label, "space")) for label in labels
    )


@dataclass(frozen=True)
class TransformSpec:
    """A named ``data -> world`` transform.

    Recorded as a scale and a translation rather than a matrix, so the same
    spec builds the v1 transform the baseline was captured with and the v2 one
    the migrated code takes.  ``baseline/*.json`` stays the frozen invariant:
    the harness changes shape, the recorded numbers do not.
    """

    name: str
    ndim: int
    scale: tuple[float, ...]
    translation: tuple[float, ...]

    def v2(
        self, data: DataCoordinateSystem, world: WorldCoordinateSystem
    ) -> AffineTransform:
        """The same transform, between two named coordinate systems."""
        return AffineTransform.from_axis_map(
            data,
            world,
            axis_map={
                data.axes[index].id: world.axes[index].id for index in range(data.ndim)
            },
            scale={
                data.axes[index].id: float(self.scale[index])
                for index in range(data.ndim)
            },
            translation={
                data.axes[index].id: float(self.translation[index])
                for index in range(data.ndim)
            },
            name="data_to_world",
        )


@dataclass(frozen=True)
class V2Context:
    """The v2 transform and render spaces one case is driven with.

    Phase 3 retyped ``BaseVisual.transform``, so the harness has to build the
    coordinate systems the render layer now composes through.  They are
    synthetic -- there is no controller here -- but they are built exactly the
    way the controller builds them: the world in cellier displayed order, the
    rendered system from ``displayed_axes`` in that order, and the visual
    system from the retained data axes **ascending**.
    """

    transform: AffineTransform
    spaces_for: Callable[[Any], Any]
    selection: Any


def _v2_context(
    tspec: TransformSpec, dspec: DimsSpec, pyramid: PyramidSpec | None = None
) -> V2Context:
    """Build the v2 transform and a spaces factory for one matrix cell.

    *pyramid* is supplied for the multiscale families and is what puts the
    per-level coordinate systems and their level-k -> level-0 transforms on
    ``RenderSpaces``.  Without them ``spaces.level_transforms`` is empty and
    ``_level_box`` has nothing to pull the region back to -- which is how
    every multiscale case in this baseline was recorded until Phase 8
    (F8.2): through the pre-migration fallback, not through the region.
    """
    labels = dspec.axis_labels
    data = DataCoordinateSystem(name="data", axes=_axes(labels), datastore_id=uuid4())
    world = WorldCoordinateSystem(name="world", axes=_axes(labels))
    transform = tspec.v2(data, world)
    displayed = dspec.displayed_axes
    canvas_id = uuid4()
    rendered = RenderedCoordinateSystem.from_world(
        world, [world.axes[axis].id for axis in displayed], canvas_id
    )
    rendered_to_world = AffineTransform.from_axis_map(
        rendered,
        world,
        axis_map={
            rendered.axes[index].id: world.axes[axis].id
            for index, axis in enumerate(displayed)
        },
        constant_output_axes={
            world.axes[axis].id: float(dspec.slice_indices.get(axis, 0.0))
            for axis in range(world.ndim)
            if axis not in displayed
        },
        name="rendered_to_world",
    )
    retained = sorted(displayed)

    levels: list[Any] = []
    level_transforms: list[AffineTransform] = []
    if pyramid is not None:
        store_id = uuid4()
        levels = [
            DataCoordinateSystem(
                name=f"level{k}", axes=_axes(labels), datastore_id=store_id
            )
            for k in range(len(pyramid.level_scales))
        ]
        level_transforms = [
            AffineTransform.from_axis_map(
                levels[k],
                levels[0],
                axis_map={
                    levels[k].axes[i].id: levels[0].axes[i].id
                    for i in range(len(labels))
                },
                scale={
                    levels[k].axes[i].id: float(scale)
                    for i, scale in enumerate(pyramid.level_scales[k])
                },
                translation={
                    levels[k].axes[i].id: (
                        0.0 if scale == 1.0 else (float(scale) - 1.0) / 2.0
                    )
                    for i, scale in enumerate(pyramid.level_scales[k])
                },
                name=f"level{k}_to_level0",
            )
            for k in range(len(pyramid.level_scales))
        ]
        data = levels[0]
        transform = tspec.v2(data, world)

    def spaces_for(visual_id):
        visual = VisualCoordinateSystem.from_data(
            data, [data.axes[axis].id for axis in retained], visual_id
        )
        return build_render_spaces(
            data,
            visual,
            world,
            rendered,
            rendered_to_world,
            transform,
            retained,
            data_levels=levels,
            level_transforms=level_transforms,
        )

    # The artifact the slicer actually consumes.  The baseline drives it
    # because it is what ships: every family prefers the region and falls back
    # to ``dims_state`` only when the controller has not placed it.
    selection = RegionSelection(
        transform=rendered_to_world,
        region=ConvexRegion.from_axis_slabs(
            world,
            {
                world.axes[axis].id: (float(position), 0.0)
                for axis, position in dspec.slice_indices.items()
            },
        ),
    )
    return V2Context(transform=transform, spaces_for=spaces_for, selection=selection)


def _place(visual: Any, ctx: V2Context) -> None:
    """Hand a driven visual the systems the controller would have pushed."""
    visual.set_render_spaces(ctx.spaces_for(visual.visual_model_id))


TRANSFORMS_3D: dict[str, TransformSpec] = {
    "identity": TransformSpec("identity", 3, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
    "aniso": TransformSpec("aniso", 3, (2.0, 0.5, 0.5), (0.0, 0.0, 0.0)),
    "scale_trans": TransformSpec("scale_trans", 3, (2.0, 0.5, 0.5), (10.0, 1.0, -3.0)),
}

# design 3.7: a 4-D tzyx with an awkward, non-integer time scale so the
# rounding step in the world -> voxel map has something to do.
TRANSFORMS_4D: dict[str, TransformSpec] = {
    "identity4": TransformSpec(
        "identity4", 4, (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0)
    ),
    "tzyx": TransformSpec("tzyx", 4, (0.5, 2.0, 0.5, 0.5), (0.25, 10.0, 0.0, 0.0)),
}


# ---------------------------------------------------------------------------
# Dims specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DimsSpec:
    """A named dims state: which axes are displayed and where the rest sit."""

    name: str
    axis_labels: tuple[str, ...]
    displayed_axes: tuple[int, ...]
    slice_indices: dict[int, float]

    def dims_state(self) -> DimsState:
        return DimsState(
            axis_labels=self.axis_labels,
            selection=AxisAlignedSelectionState(
                displayed_axes=self.displayed_axes,
            ),
        )


_LABELS_3D = ("z", "y", "x")
_LABELS_4D = ("t", "z", "y", "x")

DIMS_3D: dict[str, DimsSpec] = {
    "3d_all": DimsSpec("3d_all", _LABELS_3D, (0, 1, 2), {}),
    "2d_lead_sliced_zero": DimsSpec(
        "2d_lead_sliced_zero", _LABELS_3D, (1, 2), {0: 0.0}
    ),
    "2d_lead_sliced_nonzero": DimsSpec(
        "2d_lead_sliced_nonzero", _LABELS_3D, (1, 2), {0: 2.0}
    ),
}

DIMS_4D: dict[str, DimsSpec] = {
    "3d_all_but_time": DimsSpec("3d_all_but_time", _LABELS_4D, (1, 2, 3), {0: 0.0}),
    "3d_time_sliced_nonzero": DimsSpec(
        "3d_time_sliced_nonzero", _LABELS_4D, (1, 2, 3), {0: 1.0}
    ),
    "2d_two_sliced": DimsSpec("2d_two_sliced", _LABELS_4D, (2, 3), {0: 1.0, 1: 0.0}),
}


# ---------------------------------------------------------------------------
# Known behaviour changes -- design section 3.6
# ---------------------------------------------------------------------------

# Keyed by (family_group, reason).  The replay harness marks every case for
# a family in one of these groups xfail(strict=True): today the value in the
# JSON is what the code produces now, and when the owning phase flips the
# behaviour the assertion goes xpass, which is the signal to re-record the
# JSON and drop the marker.
XFAIL_GROUPS: dict[str, str] = {
    "geometry_slicing": (
        "design 3.12 / D4 -- geometry visuals slice against the pulled-back "
        "region in phase 6; a non-identity transform then selects different "
        "elements"
    ),
    "geometry_thickness": (
        "design 3.12 / D4 -- hardcoded thickness 0.5 becomes a world-unit "
        "half-thickness in phase 6"
    ),
    "node_matrix_non_block_diagonal": (
        "design 3.6 site E -- the node matrix stops being select_axes in "
        "phase 3; identical for block-diagonal transforms, changes for "
        "non-block-diagonal ones"
    ),
}


# ---------------------------------------------------------------------------
# Family drivers
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    """One (family, transform, dims) cell of the matrix."""

    status: str  # "ok" | "blocked:<why>" | "error:<msg>"
    node_local_matrix: Any = None
    selections: Any = None
    scale_indices: Any = None
    geometry_indices: Any = None
    notes: list[str] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "node_local_matrix": self.node_local_matrix,
            "selections": self.selections,
            "scale_indices": self.scale_indices,
            "geometry_indices": self.geometry_indices,
            "notes": self.notes,
        }


def _drive_slice_requests(
    visual: Any, dims_state: DimsState, *, is_3d: bool, selection: Any = None
) -> tuple[list[Any], list[Any]]:
    """Call the family's planning method and pull the fields that matter.

    Returns ``(selections, scale_indices)``.  ``selections`` is a sorted
    list of per-axis selection tuples: ``axis_selections`` for the image /
    label families, or ``[displayed_axes, retained_axes, region]`` for the
    geometry families (which do not resolve to voxel windows).

    The geometry entry recorded ``slice_indices`` and ``thickness`` until
    Phase 8 deleted both from the request (R8.3).  What replaces them is the
    thing that actually decides which vertices survive: the pulled-back
    region, as its per-axis bounds.  Wider than what it replaced -- a region
    says where the slab is *and* how thick -- and in **data** coordinates,
    which is the space the filter runs in.
    """
    if is_3d:
        reqs = visual.build_slice_request(
            camera_pos_world=np.array([0.0, 0.0, 0.0]),
            frustum_corners_world=None,
            fov_y_rad=1.0,
            screen_height_px=100.0,
            dims_state=dims_state,
            selection=selection,
        )
    else:
        reqs = visual.build_slice_request_2d(
            camera_pos_world=np.array([0.0, 0.0, 0.0]),
            viewport_width_px=100.0,
            world_width=10.0,
            view_min_world=None,
            view_max_world=None,
            dims_state=dims_state,
            selection=selection,
        )

    selections: list[Any] = []
    scale_indices: list[Any] = []
    for req in reqs:
        scale_indices.append(int(getattr(req, "scale_index", 0)))
        if hasattr(req, "axis_selections"):
            selections.append(
                [
                    [int(v[0]), int(v[1])] if isinstance(v, tuple) else int(v)
                    for v in req.axis_selections
                ]
            )
        else:  # geometry SliceRequest
            selections.append(
                {
                    "displayed_axes": list(req.displayed_axes),
                    "retained_axes": list(req.retained_axes),
                    "region": _region_bounds(req),
                }
            )
    return selection_key(selections), scale_indices


def _region_bounds(req: Any) -> Any:
    """A geometry request's filter, as per-axis ``[lo, hi]`` in data space.

    The graph keeps its own slab rather than a region (D6.2), so it reports
    the slab instead: its centre and the asymmetric extents around it.
    """
    if hasattr(req, "slice_positions"):
        return {
            str(axis): [
                _canon_scalar(position),
                [_canon_scalar(v) for v in req.extents.get(axis, (0.5, 0.5))],
            ]
            for axis, position in sorted(req.slice_positions.items())
        }
    box = req.region.simplify().bounding_box()
    return [
        [_canon_scalar(lo), _canon_scalar(hi)]
        for lo, hi in zip(box.min_coordinate, box.max_coordinate)
    ]


def _node_matrix(visual: Any, displayed_axes: tuple[int, ...]) -> Any:
    node = visual.get_node_for_dims(displayed_axes)
    if node is None:
        return None
    return matrix_to_list(node.local.matrix)


# --- image / label / multichannel in-memory -------------------------------


def _make_image_memory_visual(store, transform):
    from cellier.render.visuals import GFXImageMemoryVisual
    from cellier.visuals._image_memory import ImageVisual, InMemoryImageAppearance

    model = ImageVisual(
        name="baseline",
        data_store_id=str(store.id),
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="viridis"),
    )
    return GFXImageMemoryVisual(
        visual_model=model,
        data_store=store,
        render_modes={"2d", "3d"},
        transform=transform,
    )


def _make_label_memory_visual(store, transform):
    from cellier.render.visuals import GFXLabelMemoryVisual
    from cellier.visuals._label_memory import (
        InMemoryLabelsAppearance,
        LabelMemoryVisual,
    )

    model = LabelMemoryVisual(
        name="baseline",
        data_store_id=str(store.id),
        appearance=InMemoryLabelsAppearance(),
    )
    return GFXLabelMemoryVisual(
        visual_model=model,
        data_store=store,
        render_modes={"2d", "3d"},
        transform=transform,
    )


def _make_multichannel_image_memory_visual(store, transform):
    """A composite in-memory image with one channel on data axis 0.

    Kept under its pre-unification family name so the golden baseline pins
    that composite mode plans exactly what the multichannel visual did.
    """
    from cellier.render.visuals import GFXImageMemoryVisual
    from cellier.visuals._image_memory import (
        ImageVisual,
        InMemoryImageChannelAppearance,
    )

    model = ImageVisual(
        name="baseline",
        data_store_id=str(store.id),
        channel_axis=0,
        composite=True,
        channels={0: InMemoryImageChannelAppearance(color_map="red")},
    )
    return GFXImageMemoryVisual(
        visual_model=model,
        data_store=store,
        render_modes={"2d", "3d"},
        transform=transform,
    )


def _image_like_case(
    make_visual: Callable,
    store,
    tspec: TransformSpec,
    dspec: DimsSpec,
) -> CaseResult:
    ctx = _v2_context(tspec, dspec)
    visual = make_visual(store, ctx.transform)
    _place(visual, ctx)
    dims_state = dspec.dims_state()
    is_3d = len(dspec.displayed_axes) == 3

    node = _node_matrix(visual, dspec.displayed_axes)
    selections, scale_indices = _drive_slice_requests(
        visual, dims_state, is_3d=is_3d, selection=ctx.selection
    )
    return CaseResult(
        status="ok",
        node_local_matrix=node,
        selections=selections,
        scale_indices=scale_indices,
    )


# --- geometry families ---------------------------------------------------


def _make_points_visual(store, transform):
    from cellier.render.visuals._points_memory import GFXPointsMemoryVisual
    from cellier.visuals._points_memory import PointsMarkerAppearance, PointsVisual

    model = PointsVisual(
        name="baseline",
        data_store_id=str(store.id),
        appearance=PointsMarkerAppearance(),
    )
    return GFXPointsMemoryVisual(
        visual_model=model,
        render_modes={"2d", "3d"},
        transform=transform,
    )


def _make_lines_visual(store, transform):
    from cellier.render.visuals._lines_memory import GFXLinesMemoryVisual
    from cellier.visuals._lines_memory import LinesMemoryAppearance, LinesVisual

    model = LinesVisual(
        name="baseline",
        data_store_id=str(store.id),
        appearance=LinesMemoryAppearance(),
    )
    return GFXLinesMemoryVisual(
        visual_model=model,
        render_modes={"2d", "3d"},
        transform=transform,
    )


def _make_graph_visual(store, transform):
    from cellier.render.visuals._graph_memory import GFXGraphMemoryVisual
    from cellier.visuals._graph_memory import GraphAppearance, GraphVisual

    model = GraphVisual(
        name="baseline",
        data_store_id=str(store.id),
        appearance=GraphAppearance(),
    )
    return GFXGraphMemoryVisual(
        visual_model=model,
        render_modes={"2d", "3d"},
        transform=transform,
    )


def _make_mesh_visual(store, transform):
    from cellier.render.visuals._mesh_memory import GFXMeshMemoryVisual
    from cellier.visuals._mesh_memory import MeshPhongAppearance, MeshVisual

    model = MeshVisual(
        name="baseline",
        data_store_id=str(store.id),
        appearance=MeshPhongAppearance(),
    )
    return GFXMeshMemoryVisual(
        visual_model=model,
        render_modes={"2d", "3d"},
        transform=transform,
    )


def _geometry_case(
    make_visual: Callable,
    store,
    get_data_request_fn: Callable,
    tspec: TransformSpec,
    dspec: DimsSpec,
) -> CaseResult:
    ctx = _v2_context(tspec, dspec)
    visual = make_visual(store, ctx.transform)
    _place(visual, ctx)
    dims_state = dspec.dims_state()
    is_3d = len(dspec.displayed_axes) == 3

    node = _node_matrix(visual, dspec.displayed_axes)
    selections, scale_indices = _drive_slice_requests(
        visual, dims_state, is_3d=is_3d, selection=ctx.selection
    )

    # A real get_data call, to record the surviving element indices.
    if is_3d:
        reqs = visual.build_slice_request(
            camera_pos_world=np.zeros(3),
            frustum_corners_world=None,
            fov_y_rad=1.0,
            screen_height_px=100.0,
            dims_state=dims_state,
            selection=ctx.selection,
        )
    else:
        reqs = visual.build_slice_request_2d(
            camera_pos_world=np.zeros(3),
            viewport_width_px=100.0,
            world_width=10.0,
            view_min_world=None,
            view_max_world=None,
            dims_state=dims_state,
            selection=ctx.selection,
        )
    notes: list[str] = []
    try:
        data = asyncio.run(store.get_data(reqs[0]))
        geometry_indices = _extract_surviving_indices(data)
    except Exception as exc:
        geometry_indices = None
        notes.append(f"get_data raised: {type(exc).__name__}: {exc}")

    return CaseResult(
        status="ok",
        node_local_matrix=node,
        selections=selections,
        scale_indices=scale_indices,
        geometry_indices=geometry_indices,
        notes=notes,
    )


def _extract_surviving_indices(data: Any) -> Any:
    for attr in (
        "original_indices",
        "original_edge_indices",
        "original_face_indices",
        "original_node_indices",
    ):
        val = getattr(data, attr, None)
        if val is not None:
            return canon(np.asarray(val).tolist())
    return None


# --- multiscale image / label (4-D tzyx) --------------------------------


@dataclass(frozen=True)
class PyramidSpec:
    """A named 3-level 4-D ``tzyx`` pyramid.

    ``level_scales`` are the per-axis level-k -> level-0 scale factors; the
    translation is the standard ``(2^k - 1) / 2`` half-pixel offset on the
    downsampled axes and ``0`` elsewhere.
    """

    name: str
    level_shapes: tuple[tuple[int, int, int, int], ...]
    level_scales: tuple[tuple[float, float, float, float], ...]

    def level_transforms(self) -> list[AffineTransform]:
        """The per-level transforms, between synthetic level systems.

        A store reached through the controller names its own level systems;
        these visuals are built headlessly, so ``tests._v2.level_transforms``
        mints a matching set.  Only the matrices are read downstream -- the
        brick grid's per-axis scale and translation -- so the substitution is
        invisible.
        """
        translations = [
            [0.0 if s == 1.0 else (s - 1.0) / 2.0 for s in scale]
            for scale in self.level_scales
        ]
        return level_transforms(
            [list(scale) for scale in self.level_scales],
            translations,
            labels=_LABELS_4D,
        )


PYRAMIDS: dict[str, PyramidSpec] = {
    "uniform_2k": PyramidSpec(
        "uniform_2k",
        ((8, 16, 16, 16), (8, 8, 8, 8), (8, 4, 4, 4)),
        ((1, 1, 1, 1), (1, 2, 2, 2), (1, 4, 4, 4)),
    ),
    # design 3.11: z is not downsampled -- the case this repo has paid for
    # once already (the LUT anisotropic fix).
    "z_not_ds": PyramidSpec(
        "z_not_ds",
        ((8, 16, 16, 16), (8, 16, 8, 8), (8, 16, 4, 4)),
        ((1, 1, 1, 1), (1, 1, 2, 2), (1, 1, 4, 4)),
    ),
}

# The multiscale families select an LOD level from camera distance, which
# is not a stable input for a baseline.  Instead the level is pinned with
# ``force_level`` and swept over the whole pyramid, so every level's
# ``_build_axis_selections_multiscale`` path is recorded deterministically.
_MULTISCALE_CAMERA = np.array([64.0, 64.0, 64.0])


def _make_multiscale_image_visual(pyramid: PyramidSpec, transform, displayed_axes):
    from uuid import uuid4 as _uuid4

    from cellier.render.visuals._image import (
        MultiscaleBrickLayout3D,
        _MultiscaleImageSlot,
    )

    full_shapes = [tuple(s) for s in pyramid.level_shapes]
    full_tf = pyramid.level_transforms()
    disp_shapes = [tuple(s[ax] for ax in displayed_axes) for s in full_shapes]
    layout = MultiscaleBrickLayout3D(
        level_shapes=[list(s) for s in disp_shapes],
        level_transforms=full_tf,
        block_size=8,
        fetch_axes=tuple(displayed_axes),
    )
    return _MultiscaleImageSlot(
        visual_model_id=_uuid4(),
        volume_geometry=layout,
        image_geometry_2d=None,
        render_modes={"3d"},
        full_level_transforms=full_tf,
        full_level_shapes=full_shapes,
        transform=transform,
    )


def _make_multiscale_label_visual(pyramid: PyramidSpec, transform, displayed_axes):
    from uuid import uuid4 as _uuid4

    from cellier.render.visuals._image import MultiscaleBrickLayout3D
    from cellier.render.visuals._label_multiscale import GFXMultiscaleLabelVisual

    full_shapes = [tuple(s) for s in pyramid.level_shapes]
    full_tf = pyramid.level_transforms()
    disp_shapes = [tuple(s[ax] for ax in displayed_axes) for s in full_shapes]
    layout = MultiscaleBrickLayout3D(
        level_shapes=[list(s) for s in disp_shapes],
        level_transforms=full_tf,
        block_size=8,
        fetch_axes=tuple(displayed_axes),
    )
    return GFXMultiscaleLabelVisual(
        visual_model_id=_uuid4(),
        volume_geometry=layout,
        image_geometry_2d=None,
        render_modes={"3d"},
        full_level_transforms=full_tf,
        full_level_shapes=full_shapes,
        transform=transform,
    )


def _make_multichannel_multiscale_image_visual(
    pyramid: PyramidSpec, transform, displayed_axes
):
    """A composite multiscale image with one channel on data axis 0.

    Kept under its pre-unification family name; see
    ``_make_multichannel_image_memory_visual``.
    """
    from cellier.render.visuals import GFXMultiscaleImageVisual
    from cellier.visuals._image import (
        MultiscaleImageChannelAppearance,
        MultiscaleImageVisual,
    )

    full_shapes = [tuple(s) for s in pyramid.level_shapes]  # (c, z, y, x)
    model = MultiscaleImageVisual(
        name="baseline",
        data_store_id=str(uuid4()),
        channel_axis=0,
        composite=True,
        channels={0: MultiscaleImageChannelAppearance(color_map="red")},
        level_transforms=pyramid.level_transforms(),
        transform=transform,
    )
    return GFXMultiscaleImageVisual(
        visual_model=model,
        level_shapes=full_shapes,
        render_modes={"3d"},
        displayed_axes=displayed_axes,
    )


def _multiscale_case(
    make_visual: Callable,
    pyramid: PyramidSpec,
    tspec: TransformSpec,
    dspec: DimsSpec,
) -> CaseResult:
    displayed_axes = dspec.displayed_axes
    if len(displayed_axes) != 3:
        return CaseResult(status="blocked:multiscale 3D path needs 3 displayed axes")

    ctx = _v2_context(tspec, dspec, pyramid)
    visual = make_visual(pyramid, ctx.transform, displayed_axes)
    _place(visual, ctx)
    dims_state = dspec.dims_state()

    node = _node_matrix(visual, displayed_axes)

    # Each entry is [scale_index, [per-axis selection ...]].  Recording the
    # scale index inline keeps it associated with its window; a phase that
    # changes the level-k mapping then shows a readable diff.  Duplicate
    # (level, window) pairs across the force_level sweep are collapsed.
    seen: set = set()
    pairs: list[Any] = []
    scale_indices: set = set()
    for level in range(len(pyramid.level_shapes)):
        reqs = visual.build_slice_request(
            camera_pos_world=_MULTISCALE_CAMERA,
            frustum_corners_world=None,
            fov_y_rad=1.0,
            screen_height_px=200.0,
            dims_state=dims_state,
            force_level=level,
            selection=ctx.selection,
        )
        for req in reqs:
            window = [
                [int(v[0]), int(v[1])] if isinstance(v, tuple) else int(v)
                for v in req.axis_selections
            ]
            entry = [int(req.scale_index), window]
            key = repr(entry)
            if key not in seen:
                seen.add(key)
                pairs.append(entry)
            scale_indices.add(int(req.scale_index))

    return CaseResult(
        status="ok",
        node_local_matrix=node,
        selections=sorted(pairs, key=repr),
        scale_indices=sorted(scale_indices),
        notes=["LOD pinned via force_level swept over every pyramid level"],
    )


# ---------------------------------------------------------------------------
# Store factories
# ---------------------------------------------------------------------------


def _image_store(ndim: int):
    from cellier.data.image._image_memory_store import ImageMemoryStore

    rng = np.random.default_rng(0)
    shape = (5, 6, 7) if ndim == 3 else (4, 5, 6, 7)
    data = rng.random(shape, dtype=np.float64).astype(np.float32)
    return ImageMemoryStore(data=data, name="baseline")


def _label_store(ndim: int):
    from cellier.data.label._label_memory_store import LabelMemoryStore

    rng = np.random.default_rng(0)
    shape = (5, 6, 7) if ndim == 3 else (4, 5, 6, 7)
    data = rng.integers(0, 5, size=shape, dtype=np.int32)
    return LabelMemoryStore(data=data, name="baseline")


def _geom_positions(ndim: int) -> np.ndarray:
    """Five points laid out so every slab in the case matrix keeps >= 1.

    The sliced-axis positions used by the matrix are world 0, 1 and 2 (and
    a second axis at 0 for ``2d_two_sliced``).  Under an identity transform
    a half-thickness-0.5 slab at each of those must be non-empty; under the
    scaled transforms it may legitimately empty out, which is exactly the
    behaviour the geometry-slicing xfail records.
    """
    if ndim == 3:
        pts = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [2.0, 4.0, 5.0],
            [8.0, 9.0, 10.0],
        ]
    else:
        pts = [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 5.0, 3.0, 3.0],
            [2.0, 2.0, 2.0, 2.0],
            [8.0, 8.0, 8.0, 8.0],
        ]
    return np.array(pts, dtype=np.float32)


def _points_store(ndim: int):
    from cellier.data.points._points_memory_store import PointsMemoryStore

    return PointsMemoryStore(positions=_geom_positions(ndim), name="baseline")


def _lines_store(ndim: int):
    from cellier.data.lines._lines_memory_store import LinesMemoryStore

    # Three segments; both endpoints of each share their sliced-axis values
    # so a slab keeps or drops the whole segment (the store's "both
    # endpoints must pass" rule).  seg1 sits at the point every 4-D slab in
    # the matrix keeps, so at least one segment always survives.
    p = _geom_positions(ndim)
    positions = np.stack([p[0], p[0], p[1], p[1], p[3], p[3]], axis=0).astype(
        np.float32
    )
    return LinesMemoryStore(positions=positions, name="baseline")


def _graph_store(ndim: int):
    from cellier.data.graph._graph_memory_store import GraphMemoryStore

    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    return GraphMemoryStore(
        positions=_geom_positions(ndim), edges=edges, name="baseline"
    )


def _mesh_store(ndim: int):
    from cellier.data.mesh._mesh_memory_store import MeshMemoryStore

    # Two triangles; all three vertices of each share the sliced-axis
    # values (an empty slab still returns a safe placeholder here, unlike
    # the lines store).
    p = _geom_positions(ndim)
    d = np.zeros(p.shape[1], dtype=np.float32)
    d[-1] = 0.3
    positions = np.stack(
        [p[0], p[0] + d, p[0] - d, p[1], p[1] + d, p[1] - d], axis=0
    ).astype(np.float32)
    indices = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    return MeshMemoryStore(positions=positions, indices=indices, name="baseline")


# ---------------------------------------------------------------------------
# The family registry and the case matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    name: str
    kind: str  # "image" | "geometry" | "multiscale"
    make_visual: Callable
    store_factory: Callable
    xfail_groups: tuple[str, ...] = ()


FAMILIES: list[Family] = [
    Family("GFXImageMemoryVisual", "image", _make_image_memory_visual, _image_store),
    Family("GFXLabelMemoryVisual", "image", _make_label_memory_visual, _label_store),
    Family(
        "GFXMultichannelImageMemoryVisual",
        "image",
        _make_multichannel_image_memory_visual,
        _image_store,
    ),
    Family(
        "GFXPointsMemoryVisual",
        "geometry",
        _make_points_visual,
        _points_store,
        xfail_groups=("geometry_slicing", "geometry_thickness"),
    ),
    Family(
        "GFXLinesMemoryVisual",
        "geometry",
        _make_lines_visual,
        _lines_store,
        xfail_groups=("geometry_slicing", "geometry_thickness"),
    ),
    Family(
        "GFXGraphMemoryVisual",
        "geometry",
        _make_graph_visual,
        _graph_store,
        xfail_groups=("geometry_slicing", "geometry_thickness"),
    ),
    Family(
        "GFXMeshMemoryVisual",
        "geometry",
        _make_mesh_visual,
        _mesh_store,
        xfail_groups=("geometry_slicing", "geometry_thickness"),
    ),
    Family(
        "GFXMultiscaleImageVisual",
        "multiscale",
        _make_multiscale_image_visual,
        None,
    ),
    Family(
        "GFXMultiscaleLabelVisual",
        "multiscale",
        _make_multiscale_label_visual,
        None,
    ),
    Family(
        "GFXMultichannelMultiscaleImageVisual",
        "multiscale",
        _make_multichannel_multiscale_image_visual,
        None,
    ),
]


@dataclass(frozen=True)
class Case:
    family: Family
    case_id: str
    tspec: TransformSpec
    dspec: DimsSpec
    pyramid: PyramidSpec | None
    is_xfail: bool
    xfail_reason: str | None


_MULTISCALE_TRANSFORMS = {
    "identity4": TRANSFORMS_4D["identity4"],
    "tzyx": TRANSFORMS_4D["tzyx"],
}
_MULTISCALE_DIMS = {
    "3d_all_but_time": DIMS_4D["3d_all_but_time"],
    "3d_time_sliced_nonzero": DIMS_4D["3d_time_sliced_nonzero"],
}


def iter_cases():
    """Yield every :class:`Case` of the parameterised matrix."""
    for family in FAMILIES:
        if family.kind == "multiscale":
            for pyr_name, pyramid in PYRAMIDS.items():
                for tname, tspec in _MULTISCALE_TRANSFORMS.items():
                    for dname, dspec in _MULTISCALE_DIMS.items():
                        yield Case(
                            family,
                            f"pyr={pyr_name}|t={tname}|dims={dname}",
                            tspec,
                            dspec,
                            pyramid,
                            False,
                            None,
                        )
            continue

        for rank, transforms, dims in (
            (3, TRANSFORMS_3D, DIMS_3D),
            (4, TRANSFORMS_4D, DIMS_4D),
        ):
            for tname, tspec in transforms.items():
                for dname, dspec in dims.items():
                    is_xfail = bool(family.xfail_groups) and tname not in (
                        "identity",
                        "identity4",
                    )
                    reason = (
                        "; ".join(XFAIL_GROUPS[g] for g in family.xfail_groups)
                        if is_xfail
                        else None
                    )
                    yield Case(
                        family,
                        f"rank={rank}|t={tname}|dims={dname}",
                        tspec,
                        dspec,
                        None,
                        is_xfail,
                        reason,
                    )


def run_case(case: Case) -> CaseResult:
    """Drive one cell of the matrix, catching anything that goes wrong."""
    family = case.family
    try:
        if family.kind == "image":
            return _image_like_case(
                family.make_visual,
                family.store_factory(case.tspec.ndim),
                case.tspec,
                case.dspec,
            )
        if family.kind == "geometry":
            return _geometry_case(
                family.make_visual,
                family.store_factory(case.tspec.ndim),
                None,
                case.tspec,
                case.dspec,
            )
        if family.kind == "multiscale":
            return _multiscale_case(
                family.make_visual, case.pyramid, case.tspec, case.dspec
            )
    except Exception as exc:
        return CaseResult(status=f"error:{type(exc).__name__}: {exc}")
    return CaseResult(status="error:unhandled family kind")
