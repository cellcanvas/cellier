# Visuals

Visuals describe how a data store is rendered, together with the appearance
models that configure their look.

## Images

One image visual draws single-channel or composited.  `channel_axis` names
the data axis channels lie along; `composite` picks the mode.  The shared
appearance applies to both modes, `single` to single mode, and `channels`
to composite mode (one appearance per channel index).

::: cellier.visuals.BaseImageAppearance
::: cellier.visuals.effective_transparency_mode

### In-memory image

::: cellier.visuals.ImageVisual
::: cellier.visuals.InMemoryImageAppearance
::: cellier.visuals.InMemoryImageSingleAppearance
::: cellier.visuals.InMemoryImageChannelAppearance

### Multiscale image

::: cellier.visuals.MultiscaleImageVisual
::: cellier.visuals.MultiscaleImageAppearance
::: cellier.visuals.MultiscaleImageSingleAppearance
::: cellier.visuals.MultiscaleImageChannelAppearance
::: cellier.visuals.MultiscaleImageRenderConfig

## In-memory labels

::: cellier.visuals.LabelMemoryVisual
::: cellier.visuals.BaseLabelsAppearance
::: cellier.visuals.InMemoryLabelsAppearance

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

## Meshes

::: cellier.visuals.MeshVisual
::: cellier.visuals.MeshAppearance
::: cellier.visuals.MeshFlatAppearance
::: cellier.visuals.MeshPhongAppearance

## Overlays

::: cellier.visuals.CanvasOverlay
::: cellier.visuals.CenteredAxes2D
::: cellier.visuals.CenteredAxes2DAppearance
::: cellier.visuals.CanvasOverlayType

## Common

::: cellier.visuals.AABBParams
::: cellier.visuals.VisualType
