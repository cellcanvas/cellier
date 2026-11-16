# Anywidget widgets

Notebook widgets (Jupyter, marimo) for building your own layout around a
`CellierController`.  Each one reads and writes the controller's models
through its event bus.  The convenience API's `controls=` models build these
for you; the same widgets for Qt are on the [Qt page](gui_qt.md).

A widget for one appearance field (a toggle, a spin box, a colour) lists only
what it adds; what they all share (`widget`, `close`, `subscription_specs`,
`visual_ids`, `field`, `control`) is documented once, on
[`AnywidgetAppearanceField`](#cellier.gui.anywidget.visuals.AnywidgetAppearanceField) under Building blocks.

## Dims

The canvas itself comes from [`build_canvas_widget`](convenience.md#cellier.convenience.gui.build_canvas_widget), which pairs it with this panel.

::: cellier.gui.anywidget.AnywidgetDimsPanel

## Image

::: cellier.gui.anywidget.visuals.AnywidgetImageControls

## Any visual

::: cellier.gui.anywidget.visuals.AnywidgetVisibleToggle
    options:
      inherited_members: false
::: cellier.gui.anywidget.visuals.AnywidgetOpacitySlider
    options:
      inherited_members: false
::: cellier.gui.anywidget.visuals.AnywidgetUniformColorPicker
    options:
      inherited_members: false
::: cellier.gui.anywidget.visuals.AnywidgetAABBWidget

## Planes

The data these are built from comes from the [toolkit-neutral helpers](gui.md#clipping-planes-helpers).

::: cellier.gui.anywidget.visuals.AnywidgetClippingPlanesControls
::: cellier.gui.anywidget.visuals.AnywidgetRenderPlanesControls

## Level of detail and loading

::: cellier.gui.anywidget.visuals.AnywidgetLevelOfDetailControls
::: cellier.gui.anywidget.visuals.AnywidgetLoadingIndicator
::: cellier.gui.anywidget.visuals.AnywidgetLoadingConfigControls
::: cellier.gui.anywidget.visuals.AnywidgetLodConfigControls

## Labels

::: cellier.gui.anywidget.visuals.AnywidgetLabelsRenderModeCombo
    options:
      inherited_members: false
::: cellier.gui.anywidget.visuals.AnywidgetBackgroundLabelSpin
    options:
      inherited_members: false
::: cellier.gui.anywidget.visuals.AnywidgetSaltSpin
    options:
      inherited_members: false
::: cellier.gui.anywidget.visuals.AnywidgetSmoothGeometryLevelSlider
    options:
      inherited_members: false

## Points

::: cellier.gui.anywidget.visuals.AnywidgetSizeSpin
    options:
      inherited_members: false
::: cellier.gui.anywidget.visuals.AnywidgetSizeSpaceCombo
    options:
      inherited_members: false

## Lines

::: cellier.gui.anywidget.visuals.AnywidgetThicknessSpin
    options:
      inherited_members: false
::: cellier.gui.anywidget.visuals.AnywidgetThicknessSpaceCombo
    options:
      inherited_members: false

## Meshes

::: cellier.gui.anywidget.visuals.AnywidgetMeshSectionControls
::: cellier.gui.anywidget.visuals.AnywidgetFlatShadingToggle
    options:
      inherited_members: false
::: cellier.gui.anywidget.visuals.AnywidgetShininessSpin
    options:
      inherited_members: false
::: cellier.gui.anywidget.visuals.AnywidgetSideCombo
    options:
      inherited_members: false
::: cellier.gui.anywidget.visuals.AnywidgetWireframeToggle
    options:
      inherited_members: false
::: cellier.gui.anywidget.visuals.AnywidgetWireframeThicknessSpin
    options:
      inherited_members: false

## Graphs

::: cellier.gui.anywidget.visuals.AnywidgetNodeVisibleToggle
    options:
      inherited_members: false
::: cellier.gui.anywidget.visuals.AnywidgetNodeColorPicker
    options:
      inherited_members: false
::: cellier.gui.anywidget.visuals.AnywidgetNodeSizeSpin
    options:
      inherited_members: false
::: cellier.gui.anywidget.visuals.AnywidgetNodeSizeSpaceCombo
    options:
      inherited_members: false
::: cellier.gui.anywidget.visuals.AnywidgetEdgeVisibleToggle
    options:
      inherited_members: false
::: cellier.gui.anywidget.visuals.AnywidgetEdgeColorPicker
    options:
      inherited_members: false
::: cellier.gui.anywidget.visuals.AnywidgetEdgeThicknessSpin
    options:
      inherited_members: false
::: cellier.gui.anywidget.visuals.AnywidgetEdgeThicknessSpaceCombo
    options:
      inherited_members: false
::: cellier.gui.anywidget.visuals.AnywidgetTrailControls

## Render settings

One panel per section of the canvas render config: ambient occlusion, outlines and temporal accumulation.

::: cellier.gui.anywidget.AnywidgetRenderConfigPanel
::: cellier.gui.anywidget.AnywidgetAmbientOcclusionControls
::: cellier.gui.anywidget.AnywidgetOutlineControls
::: cellier.gui.anywidget.AnywidgetTemporalControls

## Dataset info

What it shows comes from [`dataset_info_from_path`](gui.md#cellier.gui.dataset_info_from_path) or a store's `dataset_info()`.

::: cellier.gui.anywidget.AnywidgetDatasetInfo

## Containers

Layout containers that hold other widgets in a notebook output.

::: cellier.gui.anywidget.AnywidgetBox
::: cellier.gui.anywidget.AnywidgetSlot

## Building blocks

The generic controls the widgets above are made from: each drives one appearance field of a type (boolean, choice, number, colour) on one or more visuals.  Use them for a field that has no named widget.

::: cellier.gui.anywidget.visuals.AnywidgetAppearanceField
::: cellier.gui.anywidget.visuals.AnywidgetToggle
::: cellier.gui.anywidget.visuals.AnywidgetChoice
::: cellier.gui.anywidget.visuals.AnywidgetFloatSpin
::: cellier.gui.anywidget.visuals.AnywidgetIntSpin
::: cellier.gui.anywidget.visuals.AnywidgetBoundedSlider
::: cellier.gui.anywidget.visuals.AnywidgetColorPicker
