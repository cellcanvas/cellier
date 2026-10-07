"""The hard break of plane rendering design v3 7.6 (Phase 2).

The old names are removed, not aliased: each is refused, and none is left in
``src`` or ``examples``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

import cellier.events
from cellier.controller import CellierController
from cellier.render._scene_config import VisualRenderConfig
from cellier.visuals import (
    MultiscaleImageAppearance,
    MultiscaleLabelsAppearance,
    ProgressiveLoadingConfig,
)
from cellier.visuals._base_visual import BaseVisual
from cellier.visuals._image import MultiscaleImageVisual
from cellier.visuals._labels import MultiscaleLabelVisual

ROOT = Path(__file__).resolve().parents[1]


def _old(*parts: str) -> str:
    """An old name, built so this file does not contain it."""
    return "_".join(parts)


LOD_BIAS = _old("lod", "bias")
DIMS_DRAG = _old("dims", "drag")
SCRUB_PROPERTY = _old("plans", "coarse", "on", "scrub")
CLIPPING_INTERACTION = _old("clipping", "interaction")


@pytest.mark.parametrize(
    "appearance_cls", [MultiscaleImageAppearance, MultiscaleLabelsAppearance]
)
def test_the_old_bias_name_is_refused(appearance_cls) -> None:
    with pytest.raises(ValidationError, match=f"{LOD_BIAS} was removed"):
        appearance_cls(**{LOD_BIAS: 2.0})
    appearance = appearance_cls(settled_lod_bias=2.0)
    assert appearance.settled_lod_bias == 2.0
    assert not hasattr(appearance, LOD_BIAS)
    with pytest.raises(ValueError, match=LOD_BIAS):
        setattr(appearance, LOD_BIAS, 3.0)


def test_the_render_config_carries_the_settled_bias() -> None:
    assert VisualRenderConfig().settled_lod_bias == 1.0
    with pytest.raises(TypeError):
        VisualRenderConfig(**{LOD_BIAS: 2.0})


@pytest.mark.parametrize("field", [DIMS_DRAG, "backstop"])
def test_the_removed_loading_fields_are_refused(field) -> None:
    assert field not in ProgressiveLoadingConfig.model_fields
    with pytest.raises(ValidationError, match=f"{field} was removed"):
        ProgressiveLoadingConfig(**{field: True})


@pytest.mark.parametrize(
    "appearance_cls", [MultiscaleImageAppearance, MultiscaleLabelsAppearance]
)
def test_the_moving_settings_defaults(appearance_cls) -> None:
    """Coarsest while moving in 3D, eager in 2D (design 7.1)."""
    appearance = appearance_cls()
    assert appearance.coarsest_while_moving_3d is True
    assert appearance.coarsest_while_moving_2d is False


@pytest.mark.parametrize(
    "visual_cls", [BaseVisual, MultiscaleImageVisual, MultiscaleLabelVisual]
)
def test_the_scrub_property_is_gone(visual_cls) -> None:
    assert not hasattr(visual_cls, SCRUB_PROPERTY)
    assert callable(visual_cls.plans_coarse_while_moving)


@pytest.mark.parametrize(
    "name",
    [
        f"begin_{CLIPPING_INTERACTION}",
        f"end_{CLIPPING_INTERACTION}",
        CLIPPING_INTERACTION,
        f"{CLIPPING_INTERACTION}_state",
        f"on_{CLIPPING_INTERACTION}",
    ],
)
def test_the_clipping_interaction_names_are_gone(name) -> None:
    assert not hasattr(CellierController, name)
    assert hasattr(CellierController, name.replace("clipping", "plane"))


def test_the_interaction_event_is_renamed() -> None:
    assert not hasattr(cellier.events, "Clipping" + "InteractionEvent")
    assert "PlaneInteractionEvent" in cellier.events.__all__


def test_the_clipping_gizmo_names_are_neutral() -> None:
    """One gizmo per canvas serves both kinds of plane (design 8.3, Phase 8)."""
    for old, new in (
        ("Clipping" + "PlaneGizmoChangedEvent", "PlaneGizmoChangedEvent"),
        ("Clipping" + "PlaneGizmoUpdateEvent", "PlaneGizmoUpdateEvent"),
    ):
        assert not hasattr(cellier.events, old)
        assert new in cellier.events.__all__
    for old, new in (
        ("get_" + "clipping_plane_gizmo", "get_plane_gizmo"),
        ("remove_" + "clipping_plane_gizmo", "remove_plane_gizmo"),
        ("on_" + "clipping_plane_gizmo_changed", "on_plane_gizmo_changed"),
    ):
        assert not hasattr(CellierController, old)
        assert hasattr(CellierController, new)
    # Adding stays per kind: the two take different planes.
    assert hasattr(CellierController, "add_clipping_plane_gizmo")
    assert hasattr(CellierController, "add_render_plane_gizmo")


def test_no_old_gizmo_name_is_left() -> None:
    pattern = re.compile(
        "Clipping" + "PlaneGizmo(Changed|Update)Event"
        "|(get|remove)_" + "clipping_plane_gizmo(?!_data)"
        "|on_" + "clipping_plane_gizmo_changed"
    )
    hits = _hits(pattern, set())
    assert not hits, "\n".join(hits)


def test_the_old_controls_key_is_refused() -> None:
    from cellier.convenience.gui._controls_config import (
        MultiscaleImageControlsConfig,
        MultiscaleLabelsControlsConfig,
    )

    for config_cls in (MultiscaleImageControlsConfig, MultiscaleLabelsControlsConfig):
        # The Phase 2 key went too, when the level of detail control came.
        for old in (LOD_BIAS, "settled_" + LOD_BIAS):
            with pytest.raises(ValueError, match="use 'level_of_detail'"):
                config_cls(appearance=[old])
        assert config_cls(appearance=["level_of_detail"]).appearance == [
            "level_of_detail"
        ]


# --------------------------------------------------------------------------- #
# No old name is left                                                          #
# --------------------------------------------------------------------------- #

#: Files where the bias is a planner argument (the settled bias is what it
#: carries) or an in-memory visual's unused argument: Phase 0 inventory, "keep".
_BIAS_ARGUMENT_FILES = {
    "src/cellier/render/_level_of_detail.py",
    "src/cellier/render/_level_of_detail_2d.py",
    "src/cellier/render/scene_manager.py",
    "src/cellier/render/visuals/_image.py",
    "src/cellier/render/visuals/_label_multiscale.py",
    "src/cellier/render/visuals/_image_memory.py",
    "src/cellier/render/visuals/_label_memory.py",
    "src/cellier/render/visuals/_lines_memory.py",
    "src/cellier/render/visuals/_points_memory.py",
    "src/cellier/render/visuals/_graph_memory.py",
}
#: The mesh keeps its own ``GeometryLodConfig`` setting of this name (D-P26).
_MESH_DRAG_FILES = {
    "src/cellier/visuals/_loading.py",
    "src/cellier/visuals/_mesh_memory.py",
    "src/cellier/render/visuals/_mesh.py",
    "src/cellier/render/_scene_config.py",
    "src/cellier/gui/_lod.py",
    "src/cellier/gui/_loading.py",
    "src/cellier/controller.py",
    "src/cellier/convenience/_viewer.py",
    "examples/mesh/timeseries_multiscale_mesh.py",
}
_SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", "__marimo__", "node_modules"}
_SUFFIXES = {".py", ".js", ".ipynb", ".md"}


def _sources():
    for top in ("src", "examples"):
        for path in sorted((ROOT / top).rglob("*")):
            if path.suffix not in _SUFFIXES or _SKIP_DIRS & set(path.parts):
                continue
            yield path.relative_to(ROOT).as_posix(), path.read_text(errors="replace")


def _word(name: str) -> re.Pattern:
    return re.compile(rf"(?<![A-Za-z0-9_]){name}(?![A-Za-z0-9_])")


#: The one place the removed names are spelled: the table that refuses them.
_REMOVED_TABLE = "src/cellier/visuals/_removed.py"


def _hits(pattern: re.Pattern, allowed: set[str], skip_line=lambda line: False):
    return [
        f"{name}:{number}: {line.strip()[:100]}"
        for name, text in _sources()
        if name not in allowed and name != _REMOVED_TABLE
        for number, line in enumerate(text.splitlines(), 1)
        if pattern.search(line) and not skip_line(line)
    ]


def test_no_old_bias_name_is_left() -> None:
    hits = _hits(_word(LOD_BIAS), _BIAS_ARGUMENT_FILES)
    assert not hits, "\n".join(hits)


def test_the_bias_slider_widgets_are_gone() -> None:
    """The level of detail control replaced them (design 9.3)."""
    import cellier.gui.anywidget.visuals as any_visuals
    import cellier.gui.qt.visuals as qt_visuals

    assert not hasattr(qt_visuals, "QtLodBias" + "Slider")
    assert not hasattr(any_visuals, "AnywidgetLodBias" + "Slider")
    static = ROOT / "src/cellier/gui/anywidget/visuals/static"
    assert not list(static.glob(LOD_BIAS + ".*"))
    assert (static / "level_of_detail.js").exists()


def test_no_image_dims_drag_is_left() -> None:
    hits = _hits(re.compile(DIMS_DRAG), _MESH_DRAG_FILES)
    assert not hits, "\n".join(hits)


@pytest.mark.parametrize(
    "pattern",
    [
        SCRUB_PROPERTY,
        CLIPPING_INTERACTION,
        "Clipping" + "InteractionEvent",
        "_clip" + "_driver",
        r"loading\.backstop(?![A-Za-z0-9_])",
        r"(?<![A-Za-z0-9_])backstop=(True|False)",
    ],
)
def test_no_other_old_name_is_left(pattern) -> None:
    hits = _hits(re.compile(pattern), set())
    assert not hits, "\n".join(hits)
