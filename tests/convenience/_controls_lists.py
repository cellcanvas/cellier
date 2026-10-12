"""Controls lists the tests share.

Test code only; nothing here is exported from ``src``.

``controls=`` has no "every field" value (ordered controls design D5), so a
test that wants every field control of a visual, and does not care which
they are, asks :func:`all_fields` or :func:`field_control`.
"""

from __future__ import annotations

from typing import Any

from cellier.convenience.gui._controls_config import (
    FIELD_KINDS,
    AmbientOcclusionControl,
    BoundingBoxControl,
    ClippingPlanesControl,
    Control,
    DatasetInfoControl,
    FieldControls,
    ImageControl,
    LevelOfDetailControl,
    LoadingConfigControl,
    LoadingIndicator,
    MeshLodControl,
    MeshSectionControl,
    OutlineControl,
    PickingControl,
    RenderPlanesControl,
    TrailControl,
)
from cellier.data._dataset_info import DatasetInfo, RowSection

IMAGE_KINDS = ("image", "multiscale_image")


_PHONG_ONLY = ("shininess", "flat_shading")


def all_fields(kind: str, model: Any = None) -> list[str]:
    """The field names of *kind*, in table order.

    *model* is a visual or an appearance model; with one, only the names it
    carries are returned.  A mesh carries the flat fields or the phong
    fields, not both: with no *model* a mesh gets the flat ones.
    """
    names = list(FIELD_KINDS[kind])
    if model is None:
        if kind in ("mesh", "multiscale_mesh"):
            names = [name for name in names if name not in _PHONG_ONLY]
        return names
    appearance = getattr(model, "appearance", model)
    single = getattr(model, "single", None)
    return [
        name
        for name in names
        if hasattr(appearance, name) or (single is not None and hasattr(single, name))
    ]


def field_control(kind: str, model: Any = None, **options: Any) -> Control:
    """The field control of *kind* naming every field (see :func:`all_fields`).

    *options* are ``ImageControl`` keywords.
    """
    fields = all_fields(kind, model)
    if kind in IMAGE_KINDS:
        return ImageControl(fields=fields, **options)
    return FieldControls(fields)


def panel(kind: str, model: Any = None, *extra: Control, **options: Any) -> list:
    """Every field control of *kind*, the bounding box, then *extra*.

    What ``appearance=True`` used to build, for a test that wants a full
    panel and does not care which fields it holds.
    """
    return [field_control(kind, model, **options), BoundingBoxControl(), *extra]


def fields(kind: str, *names: str, **options: Any) -> Control:
    """The field control of *kind* naming *names*."""
    if kind in IMAGE_KINDS:
        return ImageControl(fields=list(names), **options)
    return FieldControls(list(names))


# -- The lists of the appearance-spec parity golden ---------------------------
#
# One per case of ``data/appearance_specs_golden.json``: what the old config
# of ``scripts/ordered_controls/capture_spec_golden.py`` asked for, written as
# a list in the order the old code built (design 4.1).

_IMAGE = ["visible", "opacity", "color_map", "clim", "render_mode", "iso_threshold"]
_LABELS = ["visible", "opacity", "render_mode", "salt", "background_label"]
_MESH_FLAT = ["visible", "opacity", "color", "side", "wireframe", "wireframe_thickness"]
_MESH_PHONG = ["visible", "opacity", "color", "side", "shininess", "flat_shading"]
_IMAGE_OPTIONS = {
    "colormap_names": ["gray", "viridis", "magma"],
    "clim_range": (0.0, 10.0),
    "channel_labels": {0: "nuclei"},
    "decimals": 1,
}


def _render() -> list[Control]:
    """What ``outline_controls`` and ``ambient_occlusion_controls`` built."""
    return [OutlineControl(), AmbientOcclusionControl(), PickingControl()]


PARITY_KINDS: dict[str, str] = {
    "image": "image",
    "multiscale_image": "multiscale_image",
    "labels": "labels",
    "multiscale_labels": "multiscale_labels",
    "mesh_flat": "mesh",
    "mesh_phong": "mesh",
    "multiscale_mesh": "multiscale_mesh",
    "points": "points",
    "lines": "lines",
    "graph": "graph",
}
"""Parity case name (the part before the slash) to its visual kind."""

PARITY_CONTROLS: dict[str, Any] = {
    "image/all": lambda: [
        ImageControl(fields=_IMAGE, **_IMAGE_OPTIONS),
        BoundingBoxControl(),
        ClippingPlanesControl(),
        RenderPlanesControl(),
        *_render(),
        DatasetInfoControl(),
    ],
    "image/defaults": lambda: [
        ImageControl(fields=["color_map", "clim"]),
        BoundingBoxControl(),
    ],
    "multiscale_image/all": lambda: [
        ImageControl(fields=[*_IMAGE, "attenuation"], **_IMAGE_OPTIONS),
        LevelOfDetailControl(),
        BoundingBoxControl(),
        ClippingPlanesControl(),
        RenderPlanesControl(),
        *_render(),
        LoadingIndicator(),
        LoadingConfigControl(),
        DatasetInfoControl(),
    ],
    "labels/all": lambda: [
        FieldControls(_LABELS),
        BoundingBoxControl(),
        ClippingPlanesControl(),
        RenderPlanesControl(),
        *_render(),
        DatasetInfoControl(),
    ],
    "multiscale_labels/all": lambda: [
        FieldControls(_LABELS),
        LevelOfDetailControl(),
        BoundingBoxControl(),
        ClippingPlanesControl(),
        RenderPlanesControl(),
        *_render(),
        LoadingIndicator(),
        LoadingConfigControl(),
        DatasetInfoControl(),
    ],
    # The old config also set ``lod_controls=True``, which built nothing on
    # an in-memory mesh (design F1); the list cannot ask for it there.
    "mesh_flat/all": lambda: [
        FieldControls(_MESH_FLAT),
        BoundingBoxControl(),
        MeshSectionControl(),
        ClippingPlanesControl(),
        *_render(),
        LoadingIndicator(),
        DatasetInfoControl(),
    ],
    "mesh_phong/all": lambda: [
        FieldControls(_MESH_PHONG),
        BoundingBoxControl(),
        MeshSectionControl(),
        ClippingPlanesControl(),
        *_render(),
        DatasetInfoControl(),
    ],
    "multiscale_mesh/all": lambda: [
        FieldControls(_MESH_FLAT),
        BoundingBoxControl(),
        MeshSectionControl(),
        ClippingPlanesControl(),
        MeshLodControl(),
        *_render(),
        LoadingIndicator(),
        DatasetInfoControl(),
    ],
    "points/all": lambda: [
        FieldControls(["visible", "opacity", "color", "size", "size_space"]),
        BoundingBoxControl(),
        ClippingPlanesControl(),
        *_render(),
        DatasetInfoControl(),
    ],
    "lines/all": lambda: [
        FieldControls(["visible", "opacity", "color", "thickness", "thickness_space"]),
        BoundingBoxControl(),
        ClippingPlanesControl(),
        *_render(),
        DatasetInfoControl(),
    ],
    "graph/all": lambda: [
        FieldControls(list(FIELD_KINDS["graph"])),
        TrailControl(),
        BoundingBoxControl(),
        ClippingPlanesControl(),
        *_render(),
        DatasetInfoControl(),
    ],
    "graph/trail_axes": lambda: [
        FieldControls(["node_visible"]),
        TrailControl(axes=["t", 3]),
        BoundingBoxControl(),
    ],
    "points/dataset_info_object": lambda: [
        FieldControls(["visible"]),
        BoundingBoxControl(),
        DatasetInfoControl(
            info=DatasetInfo([RowSection(None, [("Voxel size", "1 um")])])
        ),
    ],
    "points/dataset_info_rows": lambda: [
        FieldControls(["visible"]),
        BoundingBoxControl(),
        DatasetInfoControl(info=[("Sample", "A1"), ("Count", 4)]),
    ],
}
"""Golden case name to a function returning its controls list."""
