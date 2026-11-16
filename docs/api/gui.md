# GUI

What both toolkits share: helpers for building controls and the models they
read.  The widgets themselves are on the [Qt](gui_qt.md) and
[anywidget](gui_anywidget.md) pages.

## Image control values

::: cellier.gui.image_control_values

## Axis values

The values a dims slider can take on one world axis: any position in a
range, or only the listed positions (a channel axis, say).
[`axis_values_from_viewer`](convenience.md#cellier.convenience.axis_values_from_viewer)
derives them from a viewer's data.

::: cellier.gui.ContinuousAxisValues
::: cellier.gui.DiscreteAxisValues

## Clipping planes helpers

What a clipping planes control of either toolkit is built from. See [Clip a
visual with planes](../How_To/clipping_planes.md).

::: cellier.gui.get_clipping_planes_data_from_visual
::: cellier.gui.get_axis_bounds_from_store
::: cellier.gui.ClippingPlaneGizmoTarget
::: cellier.gui.get_clipping_plane_gizmo_data

## Render planes helpers

The planes a visual in the `"plane"` render mode draws its data on: the
clipping planes rows with the extents added, in world units.

::: cellier.gui.get_render_planes_data_from_visual
::: cellier.gui.RenderPlaneGizmoTarget
::: cellier.gui.get_render_plane_gizmo_data

## Dataset info

::: cellier.gui.dataset_info_from_path
