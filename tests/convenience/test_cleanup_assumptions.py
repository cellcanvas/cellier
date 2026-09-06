"""Load-bearing assumptions of ``plans/convenience_cleanup.md``.

Stage 0 section 6.2 pins the facts that later stages rest on, so an unrelated
change (an anywidget upgrade, a widget refactor, a model edit) surfaces here as
a failing test rather than as a broken stage.  Each test names the assumption
and the stage that assumes it.

If one of these fails, the corresponding stage rests on a false premise -- the
plan needs revisiting, not the test.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import anywidget
import numpy as np
import pytest
import traitlets
from cmap import Colormap
from psygnal import EventedModel

from cellier.convenience import AppearanceControls, Viewer
from cellier.convenience._backend import ANYWIDGET_BACKEND
from cellier.convenience._hosts import QtLayoutHost
from cellier.convenience.gui._controls_config import (
    BaseControlsConfig,
    InMemoryImageControlsConfig,
    MultiscaleImageControlsConfig,
)
from cellier.convenience.layout._walk import build_appearance_widgets, render_dock
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.events import (
    AABBChangedEvent,
    AppearanceChangedEvent,
)
from cellier.gui.anywidget.visuals import (
    AnywidgetAABBWidget,
    AnywidgetClimRangeSlider,
)
from cellier.visuals._base_visual import AABBParams, BaseVisual
from cellier.visuals._image_memory import InMemoryImageAppearance

_STATIC = Path(__file__).parents[2] / "src/cellier/gui/anywidget/visuals/static"


class _ThrowawayColormapModel(EventedModel):
    """A stand-in for the real appearance models.

    ``test_writing_color_map_poisons_model_equality_process_wide`` deliberately
    triggers a class-level side effect, so it must not touch a real model
    class.  Declared at module scope because the annotation has to resolve
    under ``from __future__ import annotations``.
    """

    color_map: Colormap = Colormap("grays")


def _image_store() -> ImageMemoryStore:
    return ImageMemoryStore(data=np.zeros((8, 16, 24), dtype=np.float32))


def _two_image_viewer() -> tuple[Viewer, list]:
    """A viewer holding two independent in-memory image visuals.

    Stands in for the ``OrthoViewer`` sibling group: what the fan-out tests
    need is N visuals one widget can be subscribed to, not four panels.
    """
    viewer = Viewer(("z", "y", "x"), gui="anywidget")
    visuals = [
        viewer.add_image(
            _image_store(),
            appearance=InMemoryImageAppearance(color_map="grays", clim=(0.0, 1.0)),
        )
        for _ in range(2)
    ]
    return viewer, visuals


def _all_visual_models() -> list[type[BaseVisual]]:
    """Every concrete ``BaseVisual`` subclass, found transitively."""
    # Import for the side effect of defining every visual model class.
    import cellier.visuals  # noqa: F401

    found: list[type[BaseVisual]] = []
    stack = list(BaseVisual.__subclasses__())
    while stack:
        cls = stack.pop()
        found.append(cls)
        stack.extend(cls.__subclasses__())
    return found


# ---------------------------------------------------------------------------
# Assumption: a shared ``_esm`` works across anywidget subclasses
# Assumed by: stage 4 asset sharing (one .js per control type, many field
# classes).
# ---------------------------------------------------------------------------


def test_esm_path_is_shared_by_subclasses_and_read_as_source():
    """One ``_esm`` on a base is the *same* object on every subclass.

    Stage 4's layer-2 control-type base declares ``_esm``; its layer-3 field
    classes inherit it.  ``AnyWidget.__init_subclass__`` only coerces ``_esm``
    for classes that declare it in their own ``__dict__``, so the subclasses
    share the base's single ``FileContents`` -- which is what makes one asset
    per control type possible.
    """

    class _Base(anywidget.AnyWidget):
        _esm = _STATIC / "lod_bias.js"
        value = traitlets.Float(0.0).tag(sync=True)

    class _FieldA(_Base):
        _field = "a"

    class _FieldB(_Base):
        _field = "b"

    # The Path was coerced once, on the declaring class.
    assert _FieldA._esm is _Base._esm
    assert _FieldA._esm is _FieldB._esm
    assert "_esm" not in _FieldA.__dict__

    # ...and it is read as the file's source text, not as a path string.
    source = str(_Base._esm)
    assert source == (_STATIC / "lod_bias.js").read_text()
    assert "export default" in source

    # Instances sync identical source but stay distinguishable for CSS scoping.
    a, b = _FieldA(), _FieldB()
    assert a._esm == b._esm == source
    assert a._anywidget_id != b._anywidget_id


# ---------------------------------------------------------------------------
# Assumption: the echo filter drops all N echoes under fan-out
# Assumed by: stage 2 (appearance fan-out to the four ortho panels).
# Status before stage 0: proven for channels, not for appearance.
# ---------------------------------------------------------------------------


def test_appearance_echo_filter_drops_every_echo_under_fanout():
    """A group write stamped with the widget's id re-emits nothing.

    Stage 2 makes one appearance widget drive N visuals, so a single user edit
    produces N ``AppearanceChangedEvent`` echoes instead of one.  The
    ``source_id`` filter must drop all N -- otherwise each echo re-enters the
    widget and emits again.

    Driven through ``clim`` rather than ``color_map`` on purpose; see
    ``test_writing_color_map_poisons_model_equality_process_wide``.
    """
    viewer, visuals = _two_image_viewer()
    widget = AnywidgetClimRangeSlider(
        visuals[0].id, clim_range=(0.0, 100.0), initial_clim=(0.0, 1.0)
    )

    emitted: list = []
    widget.changed.connect(emitted.append)
    # Subscribe-to-all: the shape stage 2 gives every appearance widget.
    owner = uuid4()
    for visual in visuals:
        viewer.controller._outgoing_events.subscribe(
            AppearanceChangedEvent,
            widget._on_appearance_changed,
            entity_id=visual.id,
            owner_id=owner,
        )

    for visual in visuals:
        viewer.controller.update_appearance_field(
            visual.id, "clim", (5.0, 50.0), source_id=widget._id
        )

    assert [v.appearance.clim for v in visuals] == [(5.0, 50.0)] * 2
    assert emitted == []  # all N echoes dropped
    assert list(widget.clim) == [0.0, 1.0]  # never applied its own echo


def test_appearance_foreign_writes_apply_idempotently_under_fanout():
    """N foreign echoes for the same value leave the widget in one state.

    The other half of the stage-2 argument: echoes that are *not* the widget's
    own are applied, and applying the same value N times is idempotent and
    emits nothing (each apply runs under the ``_applying`` guard).
    """
    viewer, visuals = _two_image_viewer()
    widget = AnywidgetClimRangeSlider(
        visuals[0].id, clim_range=(0.0, 100.0), initial_clim=(0.0, 1.0)
    )

    emitted: list = []
    widget.changed.connect(emitted.append)
    owner = uuid4()
    for visual in visuals:
        viewer.controller._outgoing_events.subscribe(
            AppearanceChangedEvent,
            widget._on_appearance_changed,
            entity_id=visual.id,
            owner_id=owner,
        )

    for visual in visuals:
        viewer.controller.update_appearance_field(
            visual.id, "clim", (10.0, 90.0), source_id=uuid4()
        )

    assert list(widget.clim) == [10.0, 90.0]
    assert emitted == []


def test_aabb_echo_filter_drops_every_echo_under_fanout():
    """The AABB widget's echo filter behaves the same on its own event type."""
    viewer, visuals = _two_image_viewer()
    widget = AnywidgetAABBWidget(visuals[0].id)

    emitted: list = []
    widget.changed.connect(emitted.append)
    owner = uuid4()
    for visual in visuals:
        viewer.controller._outgoing_events.subscribe(
            AABBChangedEvent,
            widget._on_aabb_changed,
            entity_id=visual.id,
            owner_id=owner,
        )

    for visual in visuals:
        viewer.controller.update_aabb_field(
            visual.id, "enabled", True, source_id=widget._id
        )

    assert [v.aabb.enabled for v in visuals] == [True, True]
    assert emitted == []
    assert widget.enabled is False


# ---------------------------------------------------------------------------
# Assumption: AABB travels on ``AABBChangedEvent`` / ``update_aabb_field``,
# not on ``AppearanceChangedEvent``.
# Assumed by: stage 2 (AABB needs its own group helper and fan-out test).
# ---------------------------------------------------------------------------


def test_aabb_and_appearance_travel_on_separate_events():
    viewer, visuals = _two_image_viewer()
    visual = visuals[0]

    aabb_events: list = []
    appearance_events: list = []
    owner = uuid4()
    viewer.controller._outgoing_events.subscribe(
        AABBChangedEvent, aabb_events.append, entity_id=visual.id, owner_id=owner
    )
    viewer.controller._outgoing_events.subscribe(
        AppearanceChangedEvent,
        appearance_events.append,
        entity_id=visual.id,
        owner_id=owner,
    )

    viewer.controller.update_aabb_field(visual.id, "enabled", True)
    assert len(aabb_events) == 1
    assert appearance_events == []

    viewer.controller.update_appearance_field(visual.id, "clim", (0.0, 2.0))
    assert len(aabb_events) == 1  # unchanged
    assert len(appearance_events) == 1
    assert aabb_events[0].field_name == "enabled"


# ---------------------------------------------------------------------------
# Assumption: every visual has ``.aabb``.
# Assumed by: stages 1 and 4 (the AABB group is the one control valid for
# every visual type).
# ---------------------------------------------------------------------------


def test_every_visual_model_has_an_aabb_field():
    models = _all_visual_models()
    assert len(models) >= 9  # guard against an import that finds nothing

    for model in models:
        assert "aabb" in model.model_fields, f"{model.__name__} has no aabb field"
        assert model.model_fields["aabb"].default_factory is AABBParams


# ---------------------------------------------------------------------------
# Assumption: ``MultiscaleImageControlsConfig`` subclasses
# ``InMemoryImageControlsConfig``, so ``isinstance`` passes one way only.
# Assumed by: stages 3 and 4 (per-subclass vocabulary narrowing).
# ---------------------------------------------------------------------------


def test_multiscale_config_isinstance_asymmetry():
    in_memory = InMemoryImageControlsConfig()
    multiscale = MultiscaleImageControlsConfig()

    assert issubclass(MultiscaleImageControlsConfig, InMemoryImageControlsConfig)
    assert isinstance(multiscale, InMemoryImageControlsConfig)
    assert not isinstance(in_memory, MultiscaleImageControlsConfig)

    # Both are BaseControlsConfig, which is what the renderers' non-channel
    # filter actually tests.
    assert isinstance(in_memory, BaseControlsConfig)
    assert isinstance(multiscale, BaseControlsConfig)


# ---------------------------------------------------------------------------
# Assumption: ``appearance=True`` is a dead value today.
# Assumed by: stage 5, which repurposes it as "use the default field list".
# ---------------------------------------------------------------------------


def test_appearance_true_now_means_the_default_panel_in_both_renderers(qtbot):
    """**Stage 5 landed here.**  ``appearance=True`` was a dead value.

    This row of the section 6.2 table was pinned *because* stage 5 repurposes
    it: before, ``True`` took the same branch as ``False`` and produced no
    panel on either front end, and the docstring only documented
    ``list[str]`` / ``False``.  It now resolves to the config class's default
    field list -- everything it can drive (section 11.2).

    The assumption held while it needed to; this is the change it existed to
    make visible, not a failure of it.
    """
    viewer = Viewer(("z", "y", "x"), gui="qt")
    visual = viewer.add_image(
        _image_store(),
        appearance=InMemoryImageAppearance(color_map="grays", clim=(0.0, 1.0)),
        controls=InMemoryImageControlsConfig(appearance=True),
    )
    config = viewer._controls_configs[visual.id]

    from tests.convenience._qt_acceptance import (
        control_labels,
        control_labels_anywidget,
    )

    container = render_dock(AppearanceControls(), viewer, QtLayoutHost(), [])
    assert container is not None
    assert control_labels(container) == [
        "Visible",
        "Opacity",
        "Colormap",
        "Contrast limits",
        "Render mode",
        "Bounding box",
    ]

    built = build_appearance_widgets(
        visual, config, viewer.controller, None, backend=ANYWIDGET_BACKEND
    )
    assert control_labels_anywidget(built) == control_labels(container)


@pytest.mark.parametrize("value", [False, True])
def test_appearance_bool_accepted_by_the_config(value):
    """Both booleans are valid values, so stage 5 changes meaning, not type."""
    config = InMemoryImageControlsConfig(appearance=value)
    assert config.appearance is value


# ---------------------------------------------------------------------------
# Not one of the six section 6.2 rows -- found while drafting the section 6.5
# proposals, and pinned here because stage 4 section 10.4 lists a
# ``*ColormapModeCombo`` among the labels defaults.
# ---------------------------------------------------------------------------


def test_labels_colormap_mode_is_frozen_so_no_widget_can_write_it():
    """``colormap_mode`` raises on mutation, so it cannot have a control.

    ``BaseLabelsAppearance.colormap_mode`` is declared ``frozen=True`` (it is
    baked into the shader at construction), so a combo box wired to it would
    raise ``ValidationError`` on every user edit.  ``render_mode`` and ``salt``
    on the same model are mutable, so the rest of the labels row stands.
    """
    from pydantic import ValidationError

    from cellier.visuals._label_memory import InMemoryLabelsAppearance
    from cellier.visuals._labels import MultiscaleLabelsAppearance

    for model_cls in (InMemoryLabelsAppearance, MultiscaleLabelsAppearance):
        appearance = model_cls()
        with pytest.raises(ValidationError):
            appearance.colormap_mode = "direct"

        # The neighbouring fields stage 4 wants controls for are writable.
        appearance.render_mode = "flat_categorical"
        appearance.salt = 7
        appearance.background_label = 3
        assert appearance.salt == 7


def test_writing_color_map_poisons_model_equality_process_wide():
    """Writing ``color_map`` permanently breaks ``==`` on the appearance class.

    Not a section 6.2 assumption -- a latent bug found while writing them, and
    the reason the fan-out tests above drive ``clim`` instead.

    ``cmap.Colormap.__eq__`` compares LUT arrays and raises ``ValueError`` when
    the two colormaps have different LUT lengths (``"grays"`` is 9 colors,
    ``"viridis"`` is 256).  ``psygnal._group_descriptor._check_field_equality``
    catches that, and because a ``Colormap`` has no ``__array__`` it downgrades
    the field's operator to ``operator.is_`` -- **caching that on the class**.
    From then on any two distinct-but-equal appearance models compare unequal,
    so ``ViewerModel.__eq__`` (and therefore every serialization round-trip
    assertion) fails for the rest of the process.

    That makes it an ordering bomb: today ``tests/convenience`` runs before
    ``tests/gui/qt/test_qt_colormap.py``, which is the only place that writes
    the field, so nothing notices.

    Run against a throwaway model so this test cannot poison the real classes.
    """
    import operator

    from psygnal._group_descriptor import _get_eq_operator_map

    before = _ThrowawayColormapModel()
    after = _ThrowawayColormapModel()
    assert before == after

    # psygnal only compares old vs new when something is listening -- which is
    # always true in cellier, where ``Controller._wire_appearance`` subscribes
    # to ``visual.appearance.events``.
    after.events.color_map.connect(lambda _value: None)
    after.color_map = Colormap("viridis")
    assert _get_eq_operator_map(_ThrowawayColormapModel)["color_map"] is operator.is_

    # And now equality is identity: two independent, identical models differ.
    poisoned_a = _ThrowawayColormapModel(color_map=Colormap("viridis"))
    poisoned_b = _ThrowawayColormapModel(color_map=Colormap("viridis"))
    assert poisoned_a != poisoned_b
