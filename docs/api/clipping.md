# Clipping

A plane gizmo edits one plane of one visual by dragging in a 3D canvas.
`Viewer.add_clipping_plane_gizmo` and `Viewer.add_render_plane_gizmo`
return a controller; `CellierController.get_plane_gizmo` returns the one a
canvas has.  A canvas has one gizmo, of either kind.  See [Clip a visual
with planes](../How_To/clipping_planes.md).

::: cellier.clipping.PlaneGizmoController
::: cellier.clipping.ClippingPlaneGizmoController
::: cellier.clipping.RenderPlaneGizmoController
