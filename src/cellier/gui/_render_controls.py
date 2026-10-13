"""What the render-settings panels contain, described once for both toolkits.

The Qt panels and the anywidget panels draw the same controls over the same
render-config fields, and the labels, ranges and explanations are the part
most likely to drift between two hand-written copies.  They live here
instead, as data, so a change lands in both front ends at once and
``test_render_controls_parity`` can check that neither has quietly grown a
control the other lacks.

What is **not** here is anything a spec cannot express: the occlusion
panel's auto/explicit radius coupling, its effective-radius readout, and
the temporal panel's convergence readout and reset button.  Those are
per-toolkit code in the panels themselves, because they are behaviour
rather than a field with a range.

Every ``field`` here must be settable through
``CellierController.update_render_config_field``; a control over a field
with no route looks like it works and raises when touched.  The parity test
checks that too.
"""

from __future__ import annotations

import colorsys
from typing import Literal, NamedTuple

#: Hue step for a newly added palette slot, in turns.  The golden angle, so
#: successive additions land far apart on the colour wheel rather than
#: converging on a shade of the last one.
_GOLDEN_ANGLE = 0.618033988749895


def next_palette_color(entries) -> tuple[float, float, float, float]:
    """Return a colour distinct from the ones already in the palette.

    Walking the hue circle by the golden angle from the last entry: a new
    slot exists to be told apart from the others, so a default that lands
    next to one of them would be the wrong default.
    """
    if not entries:
        return (1.0, 0.0, 1.0, 1.0)
    r, g, b = (float(c) for c in tuple(entries[-1])[:3])
    hue, _lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    hue = (hue + _GOLDEN_ANGLE) % 1.0
    # Full saturation and mid lightness: a selection outline has to read
    # against an arbitrary colormapped fill.
    red, green, blue = colorsys.hls_to_rgb(hue, 0.5, max(saturation, 1.0))
    return (red, green, blue, 1.0)


#: Upper bound on the occlusion radius slider, as a multiple of the radius
#: actually in use.  A radius is in scene units, so no absolute maximum
#: means anything across cellier's coordinate systems -- a bounding box may
#: be 96 units or 0.0003.  This is why the ``radius`` control below carries
#: no maximum of its own: each front end sizes the slider from the scene.
RADIUS_SLIDER_HEADROOM = 4.0

#: Thickness is in internal pixels, and the renderer defaults to
#: ``pixel_scale = 2``, so anything past a handful is already a slab.
MAX_OUTLINE_THICKNESS = 10

_THICKNESS_NOTE = (
    "In *internal* pixels: the renderer defaults to pixel_scale 2, so the "
    "on-screen band is half this number.  Changing it recompiles the shader."
)


def render_config_path(section: str, field: str) -> str:
    """The attribute path a global control writes, e.g. for a tooltip."""
    return f"render_config.{section}.{field}"


def visual_path(field: str) -> str:
    """The attribute path a per-visual control writes."""
    return f"visual.{field}"


def with_api_path(tooltip: str, path: str) -> str:
    """Append the API path a control drives to its tooltip.

    Generated rather than written by hand, so a label can be readable
    *and* findable instead of trading one for the other -- and so it
    cannot drift from the field it names.  It also quietly answers a
    question the layout cannot: a group is only real nesting if the paths
    inside it say so, which is how "Contrast band / Thickness" reveals
    itself as the flat ``outline.inner_thickness``.
    """
    return f"{tooltip}\n\n{path}" if tooltip else path


class RenderControl(NamedTuple):
    """One control in a render-settings panel.

    Attributes
    ----------
    field : str
        Dotted attribute path within the section, e.g. ``"power"`` or
        ``"selection.inward_thickness"``.  Also the key both front ends use
        to route an inbound change back into the control.
    label : str
        What the panel calls this control.
    kind : str
        ``"bool"``, ``"int"``, ``"float"``, ``"color"``, ``"palette"``, or --
        for the per-visual controls -- ``"slot"``, ``"choice"``,
        ``"label_selection"`` or ``"range"``.  A range is a pair of
        numbers, the second never below the first, set as one value.
        A palette is a fixed-length row of RGB swatches; its length is a
        config decision rather than something a panel lets a user change.
    minimum, maximum : float or None
        Range, for the numeric kinds.
    decimals : int
        Decimal places shown, for ``"float"``.
    tooltip : str
        Why this control exists and what it costs.  Shown on hover in Qt
        and as a ``title`` attribute in the notebook.
    group : str or None
        Heading of the sub-group this control belongs to, or ``None`` for
        the top level.  Rows grouped together are ones that only mean
        something together.
    """

    field: str
    label: str
    kind: Literal[
        "bool",
        "int",
        "float",
        "color",
        "palette",
        "slot",
        "choice",
        "label_selection",
        "range",
    ]
    minimum: float | None = None
    maximum: float | None = None
    decimals: int = 2
    tooltip: str = ""
    group: str | None = None


_SSAO_CONTROLS: tuple[RenderControl, ...] = (
    RenderControl(
        "enabled",
        "Enabled",
        "bool",
        tooltip="Runs in 3D only; a 2D scene is a plane at constant depth.",
    ),
    RenderControl(
        "strength",
        "Strength",
        "float",
        0.0,
        1.0,
        tooltip="0 leaves the frame pixel-identical to the pass switched off.",
    ),
    RenderControl(
        "power",
        "Power",
        "float",
        0.25,
        4.0,
        tooltip="Above 1, only the deepest creases stay dark.",
    ),
    RenderControl(
        "auto_radius_fraction",
        "Scene fraction",
        "float",
        0.001,
        0.2,
        decimals=3,
        tooltip=(
            "Fraction of the scene bounding box diagonal used when the "
            "radius is derived rather than given."
        ),
        group="Radius",
    ),
    RenderControl(
        "radius",
        "Radius",
        "float",
        0.0,
        None,  # scene units: the panel sizes this from the effective radius
        decimals=3,
        tooltip="Occlusion hemisphere radius, in scene units.",
        group="Radius",
    ),
    RenderControl(
        "n_samples",
        "Samples",
        "int",
        4,
        64,
        tooltip=(
            "Hemisphere samples per pixel.  Recompiles the shader, so expect "
            "a hitch.  16 is enough because temporal accumulation averages "
            "the per-frame kernel rotation away."
        ),
        group="Quality",
    ),
    RenderControl(
        "blur_radius",
        "Blur radius",
        "int",
        0,
        8,
        tooltip=(
            "Box-blur half-width in internal pixels.  Recompiles.  0 shows "
            "the raw per-pixel noise the blur exists to remove."
        ),
        group="Quality",
    ),
    RenderControl(
        "bias",
        "Bias",
        "float",
        0.0,
        0.5,
        decimals=3,
        tooltip=(
            "Stops a flat surface occluding itself from depth quantisation.  "
            "A fraction of the effective radius, so it survives a change of "
            "coordinate system."
        ),
        group="Quality",
    ),
)


def _layer_controls(layer: str, group: str, *, with_color: bool):
    """The four (or five) controls of one outline layer."""
    controls = [
        RenderControl(f"{layer}.enabled", "Enabled", "bool", group=group),
        RenderControl(
            f"{layer}.inward_thickness",
            "Inward",
            "int",
            0,
            MAX_OUTLINE_THICKNESS,
            tooltip=(
                "Band inside the region's own footprint, so it never appears "
                f"to grow.  {_THICKNESS_NOTE}"
            ),
            group=group,
        ),
        RenderControl(
            f"{layer}.outward_thickness",
            "Outward",
            "int",
            0,
            MAX_OUTLINE_THICKNESS,
            tooltip=(
                "Band outside the region, reading as a halo and leaving the "
                f"region intact.  {_THICKNESS_NOTE}"
            ),
            group=group,
        ),
    ]
    if with_color:
        controls.append(RenderControl(f"{layer}.color", "Color", "color", group=group))
    return controls


#: The selection layer has no colour control, and that is not an omission:
#: it takes its colour from the palette slot carried in the visual LUT, so
#: the ``color`` field it inherits from the shared layer model does nothing
#: there.  ``cellier.controller._RENDER_CONFIG_ROUTES`` records the same
#: fact by leaving the field unroutable.
_OUTLINE_CONTROLS: tuple[RenderControl, ...] = (
    RenderControl("enabled", "Enabled", "bool"),
    *_layer_controls("boundaries", "Boundaries layer", with_color=True),
    *_layer_controls("selection", "Selection layer", with_color=False),
    RenderControl(
        "inner_thickness",
        "Thickness",
        "int",
        0,
        MAX_OUTLINE_THICKNESS,
        tooltip=(
            "A contrasting band immediately inside the selection outline, so "
            "a coloured outline stays legible against an arbitrary "
            f"colormapped fill.  0 disables it.  {_THICKNESS_NOTE}"
        ),
        group="Contrast band",
    ),
    RenderControl("inner_color", "Color", "color", group="Contrast band"),
    # No label: the group is named "Outline colors" and a row inside it
    # repeating that is a line of screen wasted on saying nothing.
    RenderControl(
        "palette",
        "",
        "palette",
        tooltip=(
            "The colours a visual can be outlined in: the per-visual slot "
            "picker offers exactly these, and a visual in slot N is drawn "
            "in entry N.  The boundaries layer has its own single colour, "
            "set in its own group.  Alpha is preserved rather than edited, "
            "because a translucent outline reads as a rendering error."
        ),
        group="Outline colors",
    ),
)


_TEMPORAL_CONTROLS: tuple[RenderControl, ...] = (
    RenderControl(
        "enabled",
        "Enabled",
        "bool",
        tooltip=(
            "Averages successive jittered frames.  Runs in 3D only.  "
            "Switching it off shows what a single frame looks like."
        ),
    ),
    RenderControl(
        "blend_weight",
        "Blend weight",
        "float",
        0.01,
        1.0,
        tooltip=(
            "Minimum weight given to the newest frame.  Lower is smoother "
            "once settled and slower to get there after a camera move.  "
            "1 disables the averaging without disabling the pass."
        ),
    ),
)


#: Every render-settings control, by config section.
RENDER_CONTROLS: dict[str, tuple[RenderControl, ...]] = {
    "ambient_occlusion": _SSAO_CONTROLS,
    "outline": _OUTLINE_CONTROLS,
    "temporal": _TEMPORAL_CONTROLS,
}


#: Human titles for each section, used by both front ends and by the
#: convenience layout when it labels a dock.
RENDER_SECTION_TITLES: dict[str, str] = {
    "ambient_occlusion": "Ambient occlusion",
    "outline": "Outlines",
    "temporal": "Temporal accumulation",
}


# ---------------------------------------------------------------------------
# Per-visual controls
#
# The other half of the two screen-space features: which visual is outlined,
# in which slot, and whether it receives occlusion.  Described here for the
# same reason the global controls are -- so both toolkits draw the same thing
# and the parity test can say so.
# ---------------------------------------------------------------------------

#: How ``ambient_occlusion``'s three states are named in a menu.  A menu
#: rather than a tri-state checkbox: a half-checked box has no name for its
#: third state, which is why the demos that use one need a tooltip to explain
#: it.
AMBIENT_OCCLUSION_CHOICES: tuple[tuple[str, object], ...] = (
    ("Auto", None),
    ("Always", True),
    ("Never", False),
)

#: How the outline pass treats a labels volume.  Each value is a ``kind`` in
#: the LUT entry, which decides both what the outline key is and where the
#: colour comes from.  Appended to rather than reordered: the Qt panel tests
#: drive this menu by index.
OUTLINE_MODE_CHOICES: tuple[tuple[str, object], ...] = (
    ("Per label", "per_label"),
    ("Whole volume", "whole_object"),
    ("All boundaries", "all_boundaries"),
    ("Depth threshold", "depth_threshold"),
)

#: Modes in which ``outline.slot`` chooses a colour rather than mere
#: participation, so the panel shows the swatch row and drops the per-label
#: rows.  Shared by both toolkits so they cannot drift apart.
SLOT_IS_COLOUR_MODES: frozenset[str] = frozenset(
    {"whole_object", "all_boundaries", "depth_threshold"}
)

#: Modes that draw depth bands, so the panel shows the depth range row.
DEPTH_RANGE_MODES: frozenset[str] = frozenset({"depth_threshold"})

#: The two boxes of a depth range row, in the order of the pair.
DEPTH_RANGE_LABELS: tuple[str, str] = ("Start", "Solid")

#: Smallest start a panel offers.  Below this the jump is inside the noise
#: of the depth buffer at the far end of a wide depth range.
DEPTH_RANGE_MINIMUM: float = 0.001

#: Largest solid value a panel offers.  A jump is a fraction of the distance
#: to the nearer surface, so it has no upper bound; past a few times that
#: distance every line is already solid.
DEPTH_RANGE_SOLID_MAXIMUM: float = 10.0

#: What a click on a spin arrow changes a depth range value by.
DEPTH_RANGE_STEP: float = 0.005


def coerce_depth_range(start: float, solid: float) -> tuple[float, float]:
    """Return the pair a depth range row holds for what its boxes say.

    The start is kept inside what the row offers, and the solid value is
    never below it: raising the start past the solid value carries the solid
    value up with it, and the solid value cannot be set below the start.  So
    the row never holds a pair the model would reject.  Shared by both
    toolkits so they cannot drift apart.
    """
    start = min(max(float(start), DEPTH_RANGE_MINIMUM), 1.0)
    solid = min(float(solid), DEPTH_RANGE_SOLID_MAXIMUM)
    return start, max(solid, start)


_OUTLINE_MODE_TOOLTIP = (
    "Per label gives each label its own band, so touching labels keep a "
    "line between them, and the colour comes from the selected labels "
    "below.  Whole volume treats the volume as one region, exactly as a "
    "mesh or an intensity volume is treated: one band around everything, "
    "coloured by the slot above.  All boundaries bands every label, "
    "boundaries between touching labels included, all in that same single "
    "colour -- whole volume's colour with per label's boundaries.  The "
    "selected labels below do not apply in it.  Depth threshold is all "
    "boundaries and one thing more: a line where one part of a label passes "
    "in front of another part of the same label, as where two branches of a "
    "tree cross.  It adds lines in a 3D view only."
)

_DEPTH_RANGE_TOOLTIP = (
    "How large a jump in depth inside one label is outlined, and how "
    "strongly.  Both values are fractions of the distance from the camera "
    "to the nearer surface.  A jump of Start is where a line begins, at no "
    "alpha; from Solid up the line is drawn in full; between the two it "
    "fades in.  Equal values give a hard edge.  The steps between voxels "
    "are jumps in depth too: zoomed in until a voxel covers several pixels, "
    "or while a coarse level is on screen, they are outlined as well, and a "
    "larger Start removes them.  A Start near 0.001 is inside the noise of "
    "the depth buffer when the scene's depth range is very wide."
)


#: ``outline.placement``'s three states.  The ``Auto`` label is completed at
#: build time with the value it resolves to for the visual in hand.
PLACEMENT_CHOICES: tuple[tuple[str, object], ...] = (
    ("Auto", None),
    ("Inward", "inward"),
    ("Outward", "outward"),
)

_PLACEMENT_TOOLTIP = (
    "Where the band sits relative to the region's own edge.  Inward never "
    "makes the region look bigger but consumes one thinner than twice the "
    "band; outward reads as a halo and leaves the region intact.  Auto picks "
    "outward for lines, points and graphs, which are a few screen pixels "
    "wide, and inward for everything else."
)

_SLOT_TOOLTIP = (
    "Which palette slot outlines this visual, or Off.  Two visuals in the "
    "same slot share a colour, which is how you say they belong together.  "
    "The slots offered are the palette's; add more in the outline panel."
)

_AO_TOOLTIP = (
    "Whether this visual receives ambient occlusion.  Auto excludes it while "
    "it renders in a MIP-family mode, whose depth is that of the brightest "
    "sample rather than of a surface, so occlusion derived from it shimmers."
)

_PICK_WRITE_TOOLTIP = (
    "Whether this visual writes to the pick buffer.  Outlines and occlusion "
    "exclusions are both derived from it, so turning it off silently stops "
    "them; choosing an outline slot turns it back on."
)

_LABEL_SELECTION_TOOLTIP = (
    "Which label values the selection layer outlines, and in which slot.  A "
    "labels visual is outlined per label rather than as one silhouette, so "
    "its own slot only makes it eligible for the boundaries layer -- the "
    "colour comes from here."
)


#: The per-visual controls, by widget.  Three widgets rather than one because
#: ``outline.slot`` means two different things: on a mesh it chooses the
#: colour, on a labels visual it only decides whether the volume participates
#: at all, and the colour comes from the per-label rows.  A single widget
#: would have to disable half of itself.
VISUAL_RENDER_CONTROLS: dict[str, tuple[RenderControl, ...]] = {
    "visual_outline": (
        RenderControl("outline.slot", "Slot", "slot", tooltip=_SLOT_TOOLTIP),
        RenderControl(
            "outline.placement", "Placement", "choice", tooltip=_PLACEMENT_TOOLTIP
        ),
    ),
    # The labels panel changes shape with its mode, because ``outline.slot``
    # means two different things.  In whole-volume mode it chooses the
    # colour, exactly as on every other visual, so it gets the swatch row.
    # In per-label mode it only decides whether the volume participates --
    # the colour comes from the rows below -- so it gets a checkbox, and a
    # swatch there would let a user pick a colour that does nothing.
    "labels_outline": (
        RenderControl("outline_mode", "Mode", "choice", tooltip=_OUTLINE_MODE_TOOLTIP),
        RenderControl("outline.slot", "Outline", "slot", tooltip=_SLOT_TOOLTIP),
        # Drawn only in the modes of ``DEPTH_RANGE_MODES``.
        RenderControl(
            "outline_depth_range",
            "Depth range",
            "range",
            minimum=DEPTH_RANGE_MINIMUM,
            maximum=DEPTH_RANGE_SOLID_MAXIMUM,
            decimals=3,
            tooltip=_DEPTH_RANGE_TOOLTIP,
        ),
        RenderControl(
            "outline.placement", "Placement", "choice", tooltip=_PLACEMENT_TOOLTIP
        ),
        RenderControl(
            "outline_selected_labels",
            "Selected labels",
            "label_selection",
            tooltip=_LABEL_SELECTION_TOOLTIP,
        ),
    ),
    "visual_occlusion": (
        RenderControl(
            "ambient_occlusion", "Ambient occlusion", "choice", tooltip=_AO_TOOLTIP
        ),
    ),
    "visual_picking": (
        RenderControl("pick_write", "Pickable", "bool", tooltip=_PICK_WRITE_TOOLTIP),
    ),
}

#: What the whole global render-settings dock is called.
#:
#: The per-visual groups are named for the feature they drive ("Outline",
#: "Ambient occlusion"); so were these, which left the two sides of a canvas
#: distinguishable only by a plural.  Naming the dock for what it *is* --
#: settings that belong to the renderer rather than to any one visual --
#: does the job once, for every panel it holds.
RENDER_DOCK_TITLE: str = "Renderer effects"


#: Titles for the per-visual groups.
VISUAL_RENDER_TITLES: dict[str, str] = {
    "visual_outline": "Outline",
    "labels_outline": "Outline",
    "visual_occlusion": "Ambient occlusion",
    "visual_picking": "Picking",
}
"""What each per-visual group is called.

Keyed by the same names as :data:`VISUAL_RENDER_CONTROLS` and as the
``ControlSpec.kind`` the renderers dispatch on, so the vocabulary is one
list rather than three that have to be kept level.  The layout layer's
``_CONTROL_TITLES`` reads these rather than restating them.
"""
