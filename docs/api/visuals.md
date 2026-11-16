# Visuals

Visuals describe how a data store is rendered, together with the appearance
models that configure their look.

## Images

One image visual draws single-channel or composited.  `channel_axis` names
the data axis channels lie along; `composite` picks the mode.  The shared
appearance applies to both modes, `single` to single mode, and `channels`
to composite mode (one appearance per channel index).

::: cellier.visuals.BaseImageVisual
::: cellier.visuals.BaseImageAppearance
::: cellier.visuals.BaseImageSingleAppearance
::: cellier.visuals.effective_transparency_mode

### In-memory image

::: cellier.visuals.ImageVisual
::: cellier.visuals.InMemoryImageAppearance
::: cellier.visuals.InMemoryImageSingleAppearance
::: cellier.visuals.InMemoryImageChannelAppearance
::: cellier.visuals.InMemoryImageRenderConfig

### Multiscale image

::: cellier.visuals.MultiscaleImageVisual
::: cellier.visuals.MultiscaleImageAppearance
::: cellier.visuals.MultiscaleImageSingleAppearance
::: cellier.visuals.MultiscaleImageChannelAppearance
::: cellier.visuals.MultiscaleImageRenderConfig
::: cellier.visuals.ProgressiveLoadingConfig
::: cellier.visuals.GeometryLodConfig

### Masked image

The isosurface of a multiscale mask, coloured by a second (value) image.
Build it with `Viewer.add_masked_image`.

::: cellier.visuals.MaskedImageVisual
::: cellier.visuals.MaskedImageAppearance
::: cellier.visuals.MaskedImageSingleAppearance
::: cellier.visuals.MaskedImageRenderConfig

## In-memory labels

::: cellier.visuals.LabelMemoryVisual
::: cellier.visuals.BaseLabelsAppearance
::: cellier.visuals.InMemoryLabelsAppearance
::: cellier.visuals.InMemoryLabelRenderConfig
::: cellier.visuals.OutlineMode

## Multiscale labels

::: cellier.visuals.MultiscaleLabelVisual
::: cellier.visuals.MultiscaleLabelsAppearance
::: cellier.visuals.MultiscaleLabelRenderConfig

## Points

::: cellier.visuals.PointsVisual
::: cellier.visuals.PointsMarkerAppearance

## Lines

::: cellier.visuals.LinesVisual
::: cellier.visuals.LinesMemoryAppearance

## Graphs

A spatial graph from a `GraphMemoryStore`: nodes drawn as points and
edges as lines.  `TrailConfig` widens the slab on one sliced axis, for
example to show a time window.

::: cellier.visuals.GraphVisual
::: cellier.visuals.GraphAppearance
::: cellier.visuals.TrailConfig

## Meshes

::: cellier.visuals.MeshVisual
::: cellier.visuals.MultiscaleMeshVisual
::: cellier.visuals.MeshSectionConfig
::: cellier.visuals.MeshAppearance
::: cellier.visuals.MeshFlatAppearance
::: cellier.visuals.MeshPhongAppearance

## Overlays

### Canvas overlays

Drawn in screen space, on top of one canvas.

::: cellier.visuals.CanvasOverlay
::: cellier.visuals.CenteredAxes2D
::: cellier.visuals.CenteredAxes2DAppearance
::: cellier.visuals.DrawStatusOverlay
::: cellier.visuals.CanvasOverlayType

### Scene overlays

Drawn in world space, in every canvas that shows the scene.

::: cellier.visuals.SceneOverlay
::: cellier.visuals.SceneBoundingBox
::: cellier.visuals.SceneBoundingBoxAppearance
::: cellier.visuals.SceneOverlayType

## Clipping planes

Every visual has a `clipping_planes` tuple, in the level-0 data coordinates
of the store it reads.  See [Clip a visual with
planes](../How_To/clipping_planes.md).

::: cellier.visuals.ClippingPlane

## Render planes

An image or labels visual whose render mode is `"plane"` draws its data on
its `render_planes` tuple in a 3D view, instead of as a volume.  A render
plane is in the scene's world space.

::: cellier.visuals.RenderPlane
::: cellier.visuals.MAX_RENDER_PLANES

## Plane outlines

A clipping plane and a render plane each carry an `outline`: a line round
the polygon the plane makes in its visual, in a 3D view.

::: cellier.visuals.PlaneOutline

## Common

::: cellier.visuals.AABBParams
::: cellier.visuals.VisualOutline
::: cellier.visuals.VisualType
