"""Visual-type-specific controls configuration dataclasses.

These are convenience-layer objects only -- they are not part of the core
cellier model and are not serialized.  Pass an instance to the ``controls=``
argument of the ``Viewer`` / ``OrthoViewer`` ``add_*`` methods.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from cellier.data._dataset_info import DatasetInfo

AppearanceField = Literal[
    # Every visual type
    "visible",
    "opacity",
    # Image
    "color_map",
    "clim",
    "render_mode",
    "iso_threshold",
    "attenuation",
    "lod_bias",
    # Labels ("render_mode" is shared with the image models, and means a
    # different set of values there -- see BaseControlsConfig.APPEARANCE_CONTROLS)
    "salt",
    "background_label",
    # Mesh, points, lines
    "color",
    "side",
    "wireframe",
    "wireframe_thickness",
    "shininess",
    "flat_shading",
    "size",
    "size_space",
    "thickness",
    "thickness_space",
    # Graph
    "node_visible",
    "node_color",
    "node_size",
    "node_size_space",
    "edge_visible",
    "edge_color",
    "edge_thickness",
    "edge_thickness_space",
]
"""Every appearance field name any config class accepts today.

Deliberately a ``Literal`` and not an ``Enum``: ``cellier`` uses ``Literal``
for closed vocabularies throughout -- including on the appearance models these
names refer to -- plain strings keep every call site unchanged, and no import
is needed at the call site, which matters for the notebook half of this
module's audience.  An enum also could not express the per-subclass narrowing
below: ``AppearanceField.LOD_BIAS`` on an in-memory image would be a valid
member that still silently did nothing (design section 9.2).

**A name is in this alias if and only if a widget exists for it** (section
10.1).  Widening it to a field with no control would let the name pass
validation and then render nothing, which is the bug the validation removes.
"""


def _validate_field_names(
    requested: Iterable[str],
    valid: Iterable[str],
    *,
    config_name: str,
    argument: str,
) -> None:
    """Raise on any name in *requested* that is not in *valid*.

    The annotation cannot enforce this -- there is no mypy in the tooling
    (design section 5) -- so the dataclass enforces its own contract at
    construction, which is far earlier than the render-time silence it
    replaces.
    """
    valid = list(valid)
    for name in requested:
        if name in valid:
            continue
        if not valid:
            raise ValueError(
                f"{config_name} accepts no {argument} fields, but {name!r} was "
                f"requested. Use a visual-type-specific config class."
            )
        close = difflib.get_close_matches(name, valid, n=1)
        suggestion = f" Did you mean {close[0]!r}?" if close else ""
        raise ValueError(
            f"{name!r} is not a valid {argument} field for {config_name}."
            f"{suggestion} Valid fields: {valid}."
        )


@dataclass
class BaseControlsConfig:
    """Base controls configuration shared by all visual types.

    Parameters
    ----------
    appearance : list[AppearanceField] or bool
        Fields to show in the appearance panel.  ``False`` (default) hides
        the appearance panel entirely; ``True`` shows this config class's
        default panel (see ``default_appearance_fields``).  A name this
        config class has no control for raises ``ValueError`` here, rather
        than being silently dropped at render time (design section 9.1).

        Note that the panel's group order follows ``APPEARANCE_CONTROLS``,
        not this list.
    outline_controls : bool
        Show this visual's outline controls -- the palette slot it is
        outlined in, the placement, and for a labels visual the per-label
        selection.  ``False`` (default) omits them.  Opt-in rather than
        always-on because the outline pass is itself off by default, so an
        always-present group would lengthen every panel for everyone.
    ambient_occlusion_controls : bool
        Show this visual's ambient-occlusion control.  ``False`` (default)
        omits it.  Separate from *outline_controls* because the two passes
        are independent.

        Either flag also brings in the picking checkbox: both features are
        derived from the pick buffer, so ``pick_write`` decides whether they
        work, and choosing an outline slot turns it on with a warning.
    dataset_info : bool or DatasetInfo or Sequence[tuple[str, str]]
        The read-only dataset-info block, appended last in the panel.

        ``False`` (default) hides it.  ``True`` asks the visual's data store
        to describe itself via ``BaseDataStore.dataset_info()`` -- the usual
        choice, since it stays correct as the store changes and costs no
        array reads.  A :class:`~cellier.data.DatasetInfo` is displayed as
        given.  A sequence of ``(label, value)`` pairs is the escape hatch
        for rows that no store knows about; both halves are coerced to
        ``str``.

        Values are displayed, never interpreted as markup, so anything read
        off a store is safe to pass verbatim.  Rendered on both front ends:
        a ``QFormLayout`` inside a ``QCollapsible`` on Qt, a table inside a
        ``<details>`` on anywidget.

    Attributes
    ----------
    APPEARANCE_CONTROLS : dict[str, str]
        Class-level map from an appearance **field name** to the **control
        kind** that drives it.  This is the config class's whole vocabulary:
        a name absent from it has no control for this kind of visual.

        It is declared per config class rather than as one global table
        because field names collide across visual types -- ``render_mode`` is
        ``mip``/``iso``/``minip`` on an image and
        ``iso_categorical``/``flat_categorical`` on labels, and a global
        name-to-widget table would hand a labels visual the volume-render
        widget (design section 6.5.2 decision 6).

        Several fields may share one kind; the control is then built once,
        which is how the three volume-render fields collapse into a single
        group.  Iteration order is display order, so the panel's group order
        is this dict's order and not the caller's field order.
    """

    APPEARANCE_CONTROLS: ClassVar[dict[str, str]] = {
        "visible": "visible",
        "opacity": "opacity",
    }

    DEFAULT_APPEARANCE_FIELDS: ClassVar[tuple[str, ...] | None] = None

    appearance: list[AppearanceField] | bool = False
    dataset_info: bool | DatasetInfo | Sequence[tuple[str, str]] = False
    outline_controls: bool = False
    ambient_occlusion_controls: bool = False

    @classmethod
    def default_appearance_fields(cls) -> tuple[str, ...]:
        """The fields ``appearance=True`` resolves to.

        Defaults to **everything this config class can drive**, which is what
        design section 10.4's catalog asks for -- every row there is starred.
        Deriving it rather than restating it means a config class that gains a
        control gains it in the default panel by the same line, and the two
        cannot drift.  ``DEFAULT_APPEARANCE_FIELDS`` overrides that for a
        class that wants a narrower default.
        """
        if cls.DEFAULT_APPEARANCE_FIELDS is not None:
            return tuple(cls.DEFAULT_APPEARANCE_FIELDS)
        return tuple(cls.APPEARANCE_CONTROLS)

    def __post_init__(self) -> None:
        """Reject appearance field names this config class cannot drive.

        ``False`` and ``True`` are untouched: they are the hide/show switch,
        not a field list.
        """
        if isinstance(self.appearance, list):
            _validate_field_names(
                self.appearance,
                self.APPEARANCE_CONTROLS,
                config_name=type(self).__name__,
                argument="appearance",
            )


@dataclass
class InMemoryImageControlsConfig(BaseControlsConfig):
    """Controls configuration for in-memory image visuals.

    Every image field is drawn by one unified image control (unified image
    design 3.10): a shared section, a composite switch when the visual has a
    channel axis, and a page per mode.  The fields listed choose which rows
    that control shows; the blending and interpolation rows and the switch
    are always there.

    Parameters
    ----------
    appearance : list[str] or bool
        Fields to show, e.g. ``["visible", "color_map", "clim"]``.  ``True``
        shows them all.
    colormap_names : list[str] or None
        Names available in the colormap dropdowns.  Defaults to a curated
        list when ``None``.
    clim_range : tuple[float, float] or None
        ``(min, max)`` bounds for the contrast-limits sliders.  Inferred
        from the current limits when ``None``.
    channel_labels : dict[int, str] or None
        Per-channel names on the composite page.  ``"Channel {i}"`` when
        ``None``.
    """

    APPEARANCE_CONTROLS: ClassVar[dict[str, str]] = {
        "visible": "image",
        "opacity": "image",
        "color_map": "image",
        "clim": "image",
        "render_mode": "image",
        "iso_threshold": "image",
    }

    colormap_names: list[str] | None = None
    clim_range: tuple[float, float] | None = None
    channel_labels: dict[int, str] | None = None


@dataclass
class MultiscaleImageControlsConfig(InMemoryImageControlsConfig):
    """Controls configuration for multiscale image visuals.

    Parameters
    ----------
    appearance : list[str] or False
        Appearance fields in display order, e.g.
        ``["color_map", "clim", "render_mode", "iso_threshold",
        "attenuation", "lod_bias"]``.
    colormap_names : list[str] or None
        Names available in the colormap dropdown.
    clim_range : tuple[float, float] or None
        ``(min, max)`` bounds for the contrast-limits slider.

    Notes
    -----
    ``dataset_info`` is inherited from ``BaseControlsConfig``: it used to
    live here, which meant only a multiscale image could show the block at
    all.  Every store now describes itself, so every config class offers it.
    """

    APPEARANCE_CONTROLS: ClassVar[dict[str, str]] = {
        **InMemoryImageControlsConfig.APPEARANCE_CONTROLS,
        "attenuation": "image",
        "lod_bias": "lod_bias",
    }


@dataclass
class LabelsControlsConfig(BaseControlsConfig):
    """Controls configuration for in-memory label visuals.

    Parameters
    ----------
    appearance : list[AppearanceField] or bool
        Appearance fields, e.g. ``["visible", "render_mode", "salt"]``.
        ``True`` shows every field this class drives; ``False`` (default)
        hides the panel.

    Notes
    -----
    ``render_mode`` maps to a *labels* render-mode combo, not the image one:
    the labels models spell it ``iso_categorical``/``flat_categorical`` where
    the image models spell it ``mip``/``iso``/``minip``.  The combo's options
    come from whichever labels model the visual actually carries, so this one
    config serves both variants.

    ``colormap_mode`` is deliberately absent: it is ``frozen=True`` on
    ``BaseLabelsAppearance``, so a control wired to it could only raise
    (design section 6.5.1 proposal 4).
    """

    APPEARANCE_CONTROLS: ClassVar[dict[str, str]] = {
        **BaseControlsConfig.APPEARANCE_CONTROLS,
        "render_mode": "labels_render_mode",
        "salt": "salt",
        "background_label": "background_label",
    }


@dataclass
class MultiscaleLabelsControlsConfig(LabelsControlsConfig):
    """Controls configuration for multiscale label visuals.

    Adds ``lod_bias``, mirroring the image pair.

    Parameters
    ----------
    appearance : list[AppearanceField] or bool
        Appearance fields.  ``True`` shows every field this class drives.
    """

    APPEARANCE_CONTROLS: ClassVar[dict[str, str]] = {
        **LabelsControlsConfig.APPEARANCE_CONTROLS,
        "lod_bias": "lod_bias",
    }


@dataclass
class MeshControlsConfig(BaseControlsConfig):
    """Controls configuration for mesh visuals.

    Covers both mesh appearance models.  ``wireframe`` and
    ``wireframe_thickness`` exist only on the flat model and ``shininess`` and
    ``flat_shading`` only on the phong one; a field the visual's model does
    not carry is skipped when the panel is built, which is why ``True`` is
    safe on either.

    Parameters
    ----------
    appearance : list[AppearanceField] or bool
        Appearance fields.  ``True`` shows every field this class drives.
    """

    APPEARANCE_CONTROLS: ClassVar[dict[str, str]] = {
        **BaseControlsConfig.APPEARANCE_CONTROLS,
        "color": "color",
        "side": "side",
        "wireframe": "wireframe",
        "wireframe_thickness": "wireframe_thickness",
        "shininess": "shininess",
        "flat_shading": "flat_shading",
    }


@dataclass
class PointsControlsConfig(BaseControlsConfig):
    """Controls configuration for points visuals.

    Parameters
    ----------
    appearance : list[AppearanceField] or bool
        Appearance fields.  ``True`` shows every field this class drives.
    """

    APPEARANCE_CONTROLS: ClassVar[dict[str, str]] = {
        **BaseControlsConfig.APPEARANCE_CONTROLS,
        "color": "color",
        "size": "size",
        "size_space": "size_space",
    }


@dataclass
class LinesControlsConfig(BaseControlsConfig):
    """Controls configuration for lines visuals.

    Parameters
    ----------
    appearance : list[AppearanceField] or bool
        Appearance fields.  ``True`` shows every field this class drives.
    """

    APPEARANCE_CONTROLS: ClassVar[dict[str, str]] = {
        **BaseControlsConfig.APPEARANCE_CONTROLS,
        "color": "color",
        "thickness": "thickness",
        "thickness_space": "thickness_space",
    }


@dataclass
class GraphControlsConfig(BaseControlsConfig):
    """Controls configuration for graph visuals.

    The node and edge halves are separate fields on one flat appearance model
    (``GraphAppearance`` is flat on purpose -- psygnal does not propagate
    nested model changes), so they are separate controls here too.

    Parameters
    ----------
    appearance : list[AppearanceField] or bool
        Appearance fields.  ``True`` shows every field this class drives --
        ten controls, which is a lot; a list is usually the better choice.
    trail_controls : bool or list[int | str]
        Show the trail-window controls -- on/off, the extent before and after
        the slice position, and fade -- for one or more of the graph's data
        axes.  ``False`` (default) omits them.  ``True`` offers the axes that
        already have a window in ``GraphVisual.trail``, or the store's time
        axes when none do.  A list names the axes by data-axis index or name,
        which is how to offer an axis that has no window yet.  An axis is
        resolved against the store when the panel is built, since the config
        does not know the store.

        Like ``outline_controls`` this adds to the appearance panel, so it
        shows nothing while ``appearance`` is ``False``.
    """

    APPEARANCE_CONTROLS: ClassVar[dict[str, str]] = {
        **BaseControlsConfig.APPEARANCE_CONTROLS,
        "node_visible": "node_visible",
        "node_color": "node_color",
        "node_size": "node_size",
        "node_size_space": "node_size_space",
        "edge_visible": "edge_visible",
        "edge_color": "edge_color",
        "edge_thickness": "edge_thickness",
        "edge_thickness_space": "edge_thickness_space",
    }

    trail_controls: bool | list[int | str] = False

    def __post_init__(self) -> None:
        """Check the shape of ``trail_controls`` as well as ``appearance``.

        Only the shape: whether an axis name or index exists depends on the
        graph's store, which is checked when the panel is built.
        """
        super().__post_init__()
        axes = self.trail_controls
        if isinstance(axes, bool):
            return
        if not isinstance(axes, (list, tuple)) or not all(
            isinstance(axis, (int, str)) and not isinstance(axis, bool) for axis in axes
        ):
            raise TypeError(
                "trail_controls must be a bool or a list of data-axis indices "
                f"or names; got {axes!r}."
            )
