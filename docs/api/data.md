# Data

Data stores hold the arrays backing each visual, along with the request and
slice-result types used to fetch data for the current view.

## Common

::: cellier.data.DataStoreType

## Images

::: cellier.data.image.ImageMemoryStore
::: cellier.data.image.MultiscaleZarrDataStore
::: cellier.data.image.OMEZarrImageDataStore
::: cellier.data.image.ChunkRequest

## Labels

::: cellier.data.label.LabelMemoryStore
::: cellier.data.label.OMEZarrLabelDataStore

## Points

::: cellier.data.points.PointsMemoryStore
::: cellier.data.points.PointsSliceRequest
::: cellier.data.points.PointsData

## Lines

::: cellier.data.lines.LinesMemoryStore
::: cellier.data.lines.LinesSliceRequest
::: cellier.data.lines.LinesData

## Meshes

::: cellier.data.mesh.MeshMemoryStore
::: cellier.data.mesh.MultiscaleMeshStore
::: cellier.data.mesh.MeshLevel
::: cellier.data.mesh.MeshSliceRequest
::: cellier.data.mesh.MeshData
