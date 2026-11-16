# Convenience

The convenience API builds a viewer with little code. It is designed to provide a single way to declare a viewer configuration and launch it on any of the GUI backends (Qt and Anywidget).

## Viewers

A `Viewer` holds one scene; an `OrthoViewer` holds four that share one world coordinate system (three orthogonal 2D slices and a 3D view).

::: cellier.convenience.Viewer
::: cellier.convenience.OrthoViewer

## Launching and display

Use `run` as the entry point for launching a configured viewer. It picks the host from the viewer's `gui=` argument, constructs the GUI, and launches the application.

::: cellier.convenience.run
::: cellier.convenience.launch
::: cellier.convenience.show
::: cellier.convenience.display
::: cellier.convenience.DisplayHandle
::: cellier.convenience.SidecarOptions
::: cellier.convenience.StartupState

## Layout

A `Layout` describes what goes in the centre of the window and in each dock.

::: cellier.convenience.Layout
::: cellier.convenience.HStack
::: cellier.convenience.VStack
::: cellier.convenience.Grid

### Dock specs

::: cellier.convenience.AppearanceControls
::: cellier.convenience.OverlayControls
::: cellier.convenience.RenderControls
::: cellier.convenience.OrthoClippingControls

## Canvas builders

For an application that lays out its own window instead of using `Layout`:
each builder makes a viewer's canvas with its dims sliders, wired to the
viewer, on the viewer's toolkit.

::: cellier.convenience.gui.build_canvas_widget
::: cellier.convenience.gui.AnywidgetCanvasView
::: cellier.convenience.gui.build_ortho_grid_widget
::: cellier.convenience.gui.OrthoCanvasGrid

## Controls

The Controls are models to declare GUI widgets for updating model properties.

```python
viewer.add_image_multiscale(
    store,
    controls=[
        ImageControl(fields=["visible", "color_map", "clim"], decimals=0),
        LevelOfDetailControl(),
        BoundingBoxControl(),
        ClippingPlanesControl(),
    ],
)
```

::: cellier.convenience.Control

### Appearance fields

::: cellier.convenience.ImageControl
::: cellier.convenience.FieldControls

### Planes

::: cellier.convenience.ClippingPlanesControl
::: cellier.convenience.RenderPlanesControl

### Drawing and picking

::: cellier.convenience.BoundingBoxControl
::: cellier.convenience.OutlineControl
::: cellier.convenience.AmbientOcclusionControl
::: cellier.convenience.PickingControl

### Level of detail and loading

::: cellier.convenience.LevelOfDetailControl
::: cellier.convenience.LoadingIndicator
::: cellier.convenience.LoadingConfigControl

### Meshes and graphs

::: cellier.convenience.MeshSectionControl
::: cellier.convenience.MeshLodControl
::: cellier.convenience.TrailControl

### Dataset info

::: cellier.convenience.DatasetInfoControl

## Axis values

Utility functions to derive the values each dims slider can take from data a viewer holds. They return [`ContinuousAxisValues`](gui.md#cellier.gui.ContinuousAxisValues) and [`DiscreteAxisValues`](gui.md#cellier.gui.DiscreteAxisValues), keyed by world axis index.

::: cellier.convenience.axis_values_from_viewer
::: cellier.convenience.axis_values_from_ortho

## Window capture

To render a viewer to a PNG without a window, see [Capture screenshots](../How_To/capture_screenshots.md).

::: cellier.convenience.screenshot_window
