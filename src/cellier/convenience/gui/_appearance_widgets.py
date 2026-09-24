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


def _any_image(spec: ControlSpec, visual_ids, controller=None):
    from cellier.gui._image_controls import display_seed
    from cellier.gui.anywidget.visuals import AnywidgetImageControls

    # Seeded now and then followed: DimsChangedEvent fires only on a change.
    scene_ids, n_displayed = display_seed(controller, visual_ids)
    return AnywidgetImageControls(
        visual_ids,
        spec.values,
        title=spec.title,
        n_displayed_dimensions=n_displayed,
        scene_ids=scene_ids,
    )


def _any_lod_bias(spec: ControlSpec, visual_ids, controller=None):
    from cellier.gui.anywidget.visuals import AnywidgetLodBiasSlider

    return AnywidgetLodBiasSlider(
        visual_ids,
        initial_lod_bias=spec.values["initial_lod_bias"],
        title=spec.title,
    )


def _any_loading(spec: ControlSpec, visual_ids, controller=None):
    from cellier.gui.anywidget.visuals import AnywidgetLoadingIndicator

    initial = (
        {vid: controller.loading_progress(vid) for vid in visual_ids}
        if controller is not None
        else None
    )
    return AnywidgetLoadingIndicator(visual_ids, initial=initial, title=spec.title)


def _any_loading_config(spec: ControlSpec, visual_ids, controller=None):
    from cellier.gui.anywidget.visuals import AnywidgetLoadingConfigControls

    return AnywidgetLoadingConfigControls(
        visual_ids,
        loading=spec.values["loading"],
        n_levels=spec.values["n_levels"],
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


def _any_trail(spec: ControlSpec, visual_ids, controller=None):
    from cellier.gui.anywidget.visuals import AnywidgetTrailControls

    return AnywidgetTrailControls(
        visual_ids, spec.values["axes"], spec.values["trail"], title=spec.title
    )


ANYWIDGET_BUILDERS = {
    "image": _any_image,
    "lod_bias": _any_lod_bias,
    "trail": _any_trail,
    "aabb": _any_aabb,
    "loading": _any_loading,
    "loading_config": _any_loading_config,
    "visual_outline": _any_visual_outline,
    "labels_outline": _any_labels_outline,
    "visual_occlusion": _any_visual_occlusion,
    "visual_picking": _any_visual_picking,
    "dataset_info": _any_dataset_info,
}
"""``ControlSpec.kind`` -> anywidget widget constructor."""
