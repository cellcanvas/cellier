"""Anywidget view layer for the shared appearance-control specs.

A dispatch table from ``ControlSpec.kind`` to an anywidget class, and nothing
else.  Which controls a panel contains, in what order, seeded with what, is
decided once in ``convenience.layout._shared.appearance_specs``; the walk that
builds and wires them is ``convenience.layout._walk``.  The Qt half of this is
``_appearance_widgets_qt.py``.

It draws no chrome: each control carries its own name -- ``label`` on a
single-field control, ``title`` on a multi-row one
(``plans/label_ownership_unification.md``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cellier.convenience.layout._shared import ControlSpec


def _any_color_map(spec: ControlSpec, visual_ids, controller=None):
    from cellier.gui.anywidget.visuals import AnywidgetColormapCombo

    return AnywidgetColormapCombo(
        visual_ids,
        initial_colormap=spec.values["initial_colormap"],
        colormap_names=spec.values["colormap_names"],
        title=spec.title,
    )


def _any_clim(spec: ControlSpec, visual_ids, controller=None):
    from cellier.gui.anywidget.visuals import AnywidgetClimRangeSlider

    return AnywidgetClimRangeSlider(
        visual_ids,
        clim_range=spec.values["clim_range"],
        initial_clim=spec.values["initial_clim"],
        title=spec.title,
    )


def _any_render(spec: ControlSpec, visual_ids, controller=None):
    from cellier.gui.anywidget.visuals import AnywidgetVolumeRenderControls

    # No ``dtype_max`` here: the anywidget control derives its own slider
    # bounds.  Its Qt counterpart takes one, which is why ``clim_range`` is on
    # the shared spec and the keyword is not.
    return AnywidgetVolumeRenderControls(
        visual_ids,
        initial_render_mode=spec.values["initial_render_mode"],
        initial_threshold=spec.values["initial_threshold"],
        initial_attenuation=spec.values["initial_attenuation"],
        title=spec.title,
    )


def _any_lod_bias(spec: ControlSpec, visual_ids, controller=None):
    from cellier.gui.anywidget.visuals import AnywidgetLodBiasSlider

    return AnywidgetLodBiasSlider(
        visual_ids,
        initial_lod_bias=spec.values["initial_lod_bias"],
        title=spec.title,
    )


def _any_aabb(spec: ControlSpec, visual_ids, controller=None):
    from cellier.gui.anywidget.visuals import AnywidgetAABBWidget

    return AnywidgetAABBWidget(
        visual_ids,
        initial_enabled=spec.values["initial_enabled"],
        initial_line_width=spec.values["initial_line_width"],
        initial_color=spec.values["initial_color"],
        title=spec.title,
    )


def _any_dataset_info(spec: ControlSpec, visual_ids, controller=None):
    """Build the read-only dataset-info block.

    The spec carries either an ``info`` (a store's sectioned
    self-description) or flat ``rows``; the widget has a constructor for
    each.  The Qt twin dispatches identically.
    """
    from cellier.gui.anywidget import AnywidgetDatasetInfo

    if "info" in spec.values:
        return AnywidgetDatasetInfo.from_info(spec.values["info"], title=spec.title)
    return AnywidgetDatasetInfo(spec.values["rows"], title=spec.title)


def _any_field_control(spec: ControlSpec, visual_ids, controller=None):
    """Build any of the 23 single-field controls from the shared table.

    The anywidget twin of ``_qt_field_control``; see it for why one builder
    serves them all.
    """
    from cellier.gui._appearance_fields import field_widget_class

    widget_class = field_widget_class(spec.kind, "anywidget")
    kwargs = {"initial_value": spec.values["initial_value"]}
    if "choices" in spec.values:
        kwargs["choices"] = spec.values["choices"]
    return widget_class(visual_ids, **kwargs)


def _any_visual_outline(spec: ControlSpec, visual_ids, controller=None):
    from cellier.gui.anywidget.render import AnywidgetVisualOutlineControls

    return AnywidgetVisualOutlineControls(
        visual_ids, spec.values, palette=spec.values.get("palette", ())
    )


def _any_labels_outline(spec: ControlSpec, visual_ids, controller=None):
    from cellier.gui.anywidget.render import AnywidgetLabelsOutlineControls

    return AnywidgetLabelsOutlineControls(
        visual_ids, spec.values, palette=spec.values.get("palette", ())
    )


def _any_visual_occlusion(spec: ControlSpec, visual_ids, controller=None):
    from cellier.gui.anywidget.render import AnywidgetVisualOcclusionControls

    return AnywidgetVisualOcclusionControls(visual_ids, spec.values)


def _any_visual_picking(spec: ControlSpec, visual_ids, controller=None):
    from cellier.gui.anywidget.render import AnywidgetVisualPickingControls

    return AnywidgetVisualPickingControls(visual_ids, spec.values)


ANYWIDGET_BUILDERS = {
    "color_map": _any_color_map,
    "clim": _any_clim,
    "render": _any_render,
    "lod_bias": _any_lod_bias,
    "aabb": _any_aabb,
    "visual_outline": _any_visual_outline,
    "labels_outline": _any_labels_outline,
    "visual_occlusion": _any_visual_occlusion,
    "visual_picking": _any_visual_picking,
    "dataset_info": _any_dataset_info,
}
"""``ControlSpec.kind`` -> anywidget widget constructor."""
