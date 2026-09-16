"""Qt view layer for the shared appearance-control specs.

The Qt half of what ``convenience/gui/_appearance_widgets.py`` does for
anywidget: a dispatch table from ``ControlSpec.kind`` to a widget class.  The
decision of *which* controls, in what order, seeded with what, is made once in
``convenience.layout._shared.appearance_specs`` and shared by both.

It lived inside ``layout/_qt_renderer.py`` until the two halves were put side
by side, which is what made their symmetry visible
(``plans/gui_backend_seam.md``).

Builders take no ``parent``: every widget built here is added to a layout
immediately, and Qt reparents it then.
"""

from __future__ import annotations

# Each builder takes ``visual_ids`` -- one id on a ``Viewer``, the four panel
# siblings on an ``OrthoViewer``.  Every widget below accepts either shape via
# ``VisualIdGroup`` (design section 8.3 step 1), so the fan-out costs the
# builders nothing.


def _qt_image(spec, visual_ids, controller):
    from cellier.gui.qt.visuals import QtImageControls

    return QtImageControls(visual_ids, spec.values, title=spec.title)


def _qt_lod_bias(spec, visual_ids, controller):
    from cellier.gui.qt.visuals import QtLodBiasSlider

    return QtLodBiasSlider(
        visual_ids,
        initial_lod_bias=spec.values["initial_lod_bias"],
        title=spec.title,
    )


def _qt_aabb(spec, visual_ids, controller):
    from cellier.gui.qt.visuals import QtAABBWidget

    return QtAABBWidget(
        visual_ids,
        initial_enabled=spec.values["initial_enabled"],
        initial_line_width=spec.values["initial_line_width"],
        initial_color=spec.values["initial_color"],
        title=spec.title,
    )


def _qt_field_control(spec, visual_ids, controller):
    """Build any of the 23 single-field controls from the shared table.

    One builder rather than 22 dispatch entries: the layer-3 classes have a
    uniform constructor (``initial_value=``, plus ``choices=`` for a combo),
    so the only thing that varies is which class, and that is a table lookup
    (``cellier.gui._appearance_fields.APPEARANCE_FIELD_WIDGETS``).
    """
    from cellier.gui._appearance_fields import field_widget_class

    widget_class = field_widget_class(spec.kind, "qt")
    kwargs = {"initial_value": spec.values["initial_value"]}
    if "choices" in spec.values:
        kwargs["choices"] = spec.values["choices"]
    return widget_class(visual_ids, parent=None, **kwargs)


def _qt_dataset_info(spec, visual_ids, controller):
    """Build the read-only dataset-info block.

    Takes neither *visual_ids* nor *controller*: it displays what the spec
    handed it and drives nothing, which is why ``STATIC_CONTROL_KINDS`` keeps
    it off the bus.

    The spec carries either an ``info`` (a store's sectioned self-description)
    or flat ``rows`` (the hand-authored escape hatch); the widget has a
    constructor for each.
    """
    from cellier.gui.qt import QtDatasetInfo

    if "info" in spec.values:
        return QtDatasetInfo.from_info(
            spec.values["info"], title=spec.title, parent=None
        )
    return QtDatasetInfo(spec.values["rows"], title=spec.title, parent=None)


def _qt_visual_outline(spec, visual_ids, controller):
    from cellier.gui.qt.render import QtVisualOutlineControls

    return QtVisualOutlineControls(
        visual_ids,
        spec.values,
        palette=spec.values.get("palette", ()),
        parent=None,
    )


def _qt_labels_outline(spec, visual_ids, controller):
    from cellier.gui.qt.render import QtLabelsOutlineControls

    return QtLabelsOutlineControls(
        visual_ids,
        spec.values,
        palette=spec.values.get("palette", ()),
        parent=None,
    )


def _qt_visual_occlusion(spec, visual_ids, controller):
    from cellier.gui.qt.render import QtVisualOcclusionControls

    return QtVisualOcclusionControls(visual_ids, spec.values, parent=None)


def _qt_visual_picking(spec, visual_ids, controller):
    from cellier.gui.qt.render import QtVisualPickingControls

    return QtVisualPickingControls(visual_ids, spec.values, parent=None)


def _qt_trail(spec, visual_ids, controller):
    from cellier.gui.qt.visuals import QtTrailControls

    return QtTrailControls(
        visual_ids, spec.values["axes"], spec.values["trail"], title=spec.title
    )


QT_BUILDERS = {
    "image": _qt_image,
    "lod_bias": _qt_lod_bias,
    "trail": _qt_trail,
    "aabb": _qt_aabb,
    "visual_outline": _qt_visual_outline,
    "labels_outline": _qt_labels_outline,
    "visual_occlusion": _qt_visual_occlusion,
    "visual_picking": _qt_visual_picking,
    "dataset_info": _qt_dataset_info,
}
"""``ControlSpec.kind`` -> Qt widget constructor."""
