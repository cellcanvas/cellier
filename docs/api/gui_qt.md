# Qt widgets

Qt widgets for building your own window around a `CellierController`.  Each one
reads and writes the controller's models through its event bus.  The
convenience API's `controls=` models build these for you; the same widgets
for notebooks are on the [anywidget page](gui_anywidget.md).

A widget for one appearance field (a toggle, a spin box, a colour) lists only
what it adds; what they all share (`widget`, `close`, `subscription_specs`,
`visual_ids`, `field`, `control`, `value`) is documented once, on
[`QtAppearanceField`](#cellier.gui.qt.visuals.QtAppearanceField) under Building blocks.

## Canvas and dims

::: cellier.gui.qt.QtCanvasWidget
::: cellier.gui.qt.QtDimsControl

## Image

::: cellier.gui.qt.visuals.QtImageControls

## Any visual

::: cellier.gui.qt.visuals.QtVisibleToggle
    options:
      inherited_members: false
::: cellier.gui.qt.visuals.QtOpacitySlider
    options:
      inherited_members: false
::: cellier.gui.qt.visuals.QtUniformColorPicker
    options:
      inherited_members: false
::: cellier.gui.qt.visuals.QtAABBWidget

## Planes

The data these are built from comes from the [toolkit-neutral helpers](gui.md#clipping-planes-helpers).

::: cellier.gui.qt.visuals.QtClippingPlanesControls
::: cellier.gui.qt.visuals.QtRenderPlanesControls

## Level of detail and loading

::: cellier.gui.qt.visuals.QtLevelOfDetailControls
::: cellier.gui.qt.visuals.QtLoadingIndicator
::: cellier.gui.qt.visuals.QtLoadingConfigControls
::: cellier.gui.qt.visuals.QtLodConfigControls

## Labels

::: cellier.gui.qt.visuals.QtLabelsRenderModeCombo
    options:
      inherited_members: false
::: cellier.gui.qt.visuals.QtBackgroundLabelSpin
    options:
      inherited_members: false
::: cellier.gui.qt.visuals.QtSaltSpin
    options:
      inherited_members: false
::: cellier.gui.qt.visuals.QtSmoothGeometryLevelSlider
    options:
      inherited_members: false

## Points

::: cellier.gui.qt.visuals.QtSizeSpin
    options:
      inherited_members: false
::: cellier.gui.qt.visuals.QtSizeSpaceCombo
    options:
      inherited_members: false

## Lines

::: cellier.gui.qt.visuals.QtThicknessSpin
    options:
      inherited_members: false
::: cellier.gui.qt.visuals.QtThicknessSpaceCombo
    options:
      inherited_members: false

## Meshes

::: cellier.gui.qt.visuals.QtMeshSectionControls
::: cellier.gui.qt.visuals.QtFlatShadingToggle
    options:
      inherited_members: false
::: cellier.gui.qt.visuals.QtShininessSpin
    options:
      inherited_members: false
::: cellier.gui.qt.visuals.QtSideCombo
    options:
      inherited_members: false
::: cellier.gui.qt.visuals.QtWireframeToggle
    options:
      inherited_members: false
::: cellier.gui.qt.visuals.QtWireframeThicknessSpin
    options:
      inherited_members: false

## Graphs

::: cellier.gui.qt.visuals.QtNodeVisibleToggle
    options:
      inherited_members: false
::: cellier.gui.qt.visuals.QtNodeColorPicker
    options:
      inherited_members: false
::: cellier.gui.qt.visuals.QtNodeSizeSpin
    options:
      inherited_members: false
::: cellier.gui.qt.visuals.QtNodeSizeSpaceCombo
    options:
      inherited_members: false
::: cellier.gui.qt.visuals.QtEdgeVisibleToggle
    options:
      inherited_members: false
::: cellier.gui.qt.visuals.QtEdgeColorPicker
    options:
      inherited_members: false
::: cellier.gui.qt.visuals.QtEdgeThicknessSpin
    options:
      inherited_members: false
::: cellier.gui.qt.visuals.QtEdgeThicknessSpaceCombo
    options:
      inherited_members: false
::: cellier.gui.qt.visuals.QtTrailControls

## Render settings

One panel per section of the canvas render config: ambient occlusion, outlines and temporal accumulation.

::: cellier.gui.qt.QtRenderConfigPanel
::: cellier.gui.qt.QtAmbientOcclusionControls
::: cellier.gui.qt.QtOutlineControls
::: cellier.gui.qt.QtTemporalControls

## Dataset info

What it shows comes from [`dataset_info_from_path`](gui.md#cellier.gui.dataset_info_from_path) or a store's `dataset_info()`.

::: cellier.gui.qt.QtDatasetInfo

## Building blocks

The generic controls the widgets above are made from: each drives one appearance field of a type (boolean, choice, number, colour) on one or more visuals.  Use them for a field that has no named widget.

::: cellier.gui.qt.visuals.QtAppearanceField
::: cellier.gui.qt.visuals.QtToggle
::: cellier.gui.qt.visuals.QtChoice
::: cellier.gui.qt.visuals.QtFloatSpin
::: cellier.gui.qt.visuals.QtIntSpin
::: cellier.gui.qt.visuals.QtBoundedSlider
::: cellier.gui.qt.visuals.QtColorPicker
::: cellier.gui.qt.visuals.labelled_row
::: cellier.gui.qt.visuals.titled_group
