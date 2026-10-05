# Clip a visual with planes

A clipping plane hides the part of a visual on one side of it. Every visual
type takes them: images and labels (in memory or multiscale), meshes,
points, lines and graphs. Each visual has its own planes; there are no
scene-wide ones.

## The short version

```python
from cellier.visuals import ClippingPlane

system = store.data_coordinate_systems[0]
plane = ClippingPlane.from_point_normal(
    system, point=(0, 0, 100), normal=(0, 0, 1), axes=("z", "y", "x")
)
visual = viewer.add_image(store, clipping_planes=(plane,))
```

The visual is drawn where `normal . p >= normal . point`: the side the
normal points to. Here that is `x >= 100`.

## Store first, then planes, then the visual

A plane is written in the **level-0 data coordinates** of the store the
visual reads, so the store comes first and the plane is built from its
coordinate system.

```python
store = OMEZarrImageDataStore.from_path(uri)          # has its system
# or
store = ImageMemoryStore(data=array, data_coordinate_systems=[system])

plane = ClippingPlane.from_point_normal(store.data_coordinate_systems[0], ...)
visual = viewer.add_image(store, clipping_planes=(plane,))
```

A store built with no coordinate system takes the scene's axes when its
first visual is added. It has nothing to build a plane from before that, so
add the visual and then assign the planes:

```python
visual = viewer.add_points(PointsMemoryStore(positions=positions))
visual.clipping_planes = (
    ClippingPlane.from_point_normal(store.data_coordinate_systems[0], ...),
)
```

A plane belongs to one store object. Opening the same file twice gives two
stores with two coordinate systems, and a plane built for one is refused by
a visual of the other. Saving and loading a viewer keeps the planes.

## Units and direction

- **Data units.** Voxels for an image; the positions' own units for a mesh
  or points. Not world units.
- **The normal is not a direction in the sample when voxels are not
  cubes.** It acts on voxel indices. With a z spacing four times the x
  spacing, a plane tilted 45 degrees in the sample has a normal of
  `(1, 0, 4)` over `(z, y, x)`, not `(1, 0, 1)`.
- The normal need not be unit length.

## Planes on some axes only

`axes=` names the axes `point` and `normal` are given on. The other axes are
not constrained, so a `zyx` plane on `tczyx` data cuts every timepoint and
every channel in the same place:

```python
ClippingPlane.from_point_normal(
    system, point=(40, 0, 0), normal=(1, 0, 0), axes=("z", "y", "x")
)
```

Any axis may carry a component. A plane with a component on `t` moves as the
time slider moves. On a composite image, a plane with a component on the
channel axis cuts each channel in a different place.

## Several planes

A visual keeps the **intersection**: what is on the kept side of every
enabled plane. Two opposed planes keep a slab; six keep a box.

```python
visual.clipping_planes = (left, right)
```

## Changing planes

Planes are frozen. Assign a new tuple; one assignment is one event.

```python
visual.clipping_planes = (moved,)                       # move
visual.clipping_planes = (*visual.clipping_planes, new) # add
visual.clipping_planes = ()                             # remove all
```

`enabled=False` keeps a plane in the list without clipping. Switching it on
and off is free. Adding or removing a plane changes the number of planes,
which compiles a shader the first time that number is used (a short hitch,
once).

`controller.set_clipping_planes(visual_id, planes, source_id=...)` does the
same with a `source_id` on the resulting `ClippingPlanesChangedEvent`.

Each plane has an `id` that survives a move, so a plane can be followed
through replaced tuples. To move one plane and leave the others:

```python
controller.set_clipping_plane(visual.id, plane.id, new_plane)  # a transform Plane
```

## What a cut looks like

| Visual | In a 3D view | In a 2D view |
|---|---|---|
| Image, MIP | The projection of the kept part | The slice, cut at a line |
| Image, ISO; labels | A solid face on the plane, lit with the plane's normal | The slice, cut at a line |
| Mesh | An open (hollow) cut | Its section, cut at a line |
| Lines, graph edges | Cut at the plane | Cut at the plane |
| Points, graph nodes | Whole markers, kept or dropped by their centre | The same |

In a 2D view the plane is the line where it meets the slice, and the line
moves with the slider. Geometry drawn through a slab thicker than zero is
clipped by where it really is, not by where it lands on the slice.

The bounding-box wireframe and overlays are not clipped. Camera fit uses the
whole data, not the kept part.

## What it costs

- Moving a plane does not reload a mesh, points or an in-memory image in 3D:
  it is a uniform update.
- A multiscale image or label volume plans again, and does not fetch bricks
  or tiles that are wholly clipped. The budget that frees goes to the part
  that is kept.
- Geometry in a 2D slab is read again on each move, because the read does
  the clipping there. A large mesh keeps its last picture on screen until
  the new one is ready.

## Picking and painting

A clipped part cannot be picked. The paint brush writes only voxels on the
kept side.

## The control

Every controls config takes `clipping_controls=True`:

```python
viewer.add_image(
    store,
    controls=InMemoryImageControlsConfig(appearance=True, clipping_controls=True),
)
```

The "Clipping planes" group has a row per plane: on or off, a flip button,
a remove button, the normal, and a position slider along the normal.

The flag needs a dock to show the control in. On a `Viewer` the control is
part of the appearance dock, so the config also needs `appearance` and the
layout an `AppearanceControls()` dock. On an `OrthoViewer` it is in the
`OrthoClippingControls()` dock ([below](#in-an-orthoviewer)). A flag with
no dock for it raises `ValueError`, when the layout is rendered or when
the visual is added to a viewer that is already shown.

The normal has one column per data axis, in the same order as the entries
of `normal`. The axis names come from the store's data coordinate system
(`store.data_coordinate_systems[0]`). For an axis named `z`, the column has:

- a `+z` and a `-z` button. They face the plane along that axis and keep the
  side toward higher (`+z`) or lower (`-z`) values. The plane turns in place.
- under them, the normal's entry on that axis, for an oblique plane. Only the direction
  of the normal matters; the entries are not rescaled as you type.

The position slider's range is the data's bounding box along the normal, and
follows the store when its extent changes.

## The gizmo

A gizmo in a 3D view moves and tilts one plane by dragging:

```python
gizmo = viewer.add_clipping_plane_gizmo(visual, visual.clipping_planes[0])
...
gizmo.close()            # or viewer.remove_clipping_plane_gizmo()
```

On an `OrthoViewer` the gizmo is drawn in the 3D panel and edits the 3D
panel's planes; the 2D panels follow when they are linked to it
([below](#in-an-orthoviewer)). Without a viewer:
`controller.add_clipping_plane_gizmo(visual_id, canvas_id, plane_id)`.

- The arrow along the normal slides the plane. The two rings that tilt the
  normal turn it about the gizmo.
- The other handles (the two in-plane arrows, the in-plane square, the ring
  about the normal) move the gizmo on the plane and leave the plane where
  it is. Use them to bring the gizmo to the part you are looking at.
- A plane changed any other way (the slider, an assignment) moves the gizmo.
- A view has one gizmo at a time. Adding one on another plane closes the
  one it had.
- The gizmo closes itself when its plane or its visual is removed, when the
  view switches to 2D, and when the plane is given a component on an axis
  the view does not show (a plane tilted in time). Such a plane cannot have
  one.
- A disabled plane can have a gizmo: place it, then switch it on.

It first appears on the plane at the point nearest the one the camera
orbits about. If the plane does not pass through the view, the gizmo is off
screen until the camera or the plane moves.

In the control, each row has a "Gizmo" toggle that does the same. The
toggles of every control of a view act as one set: switching one on
switches the others off. A toggle is greyed out, with the reason as its
tooltip, where a gizmo is not possible.

A control has the toggle only when it is told where the gizmo goes:

- A `Viewer`'s dock names its visual and its canvas when the viewer has
  exactly one canvas. With several canvases the control has no toggle
  (the viewer does not choose one): call
  `viewer.add_clipping_plane_gizmo(..., canvas=)`. A viewer that never
  shows 3D has no toggle either.
- An `OrthoViewer`'s dock names the 3D panel.
- A control built by hand takes `gizmo_target=`:

  ```python
  from cellier.gui import (
      get_clipping_plane_gizmo_data,
      get_clipping_planes_data_from_visual,
  )
  from cellier.gui.qt.visuals import QtClippingPlanesControls

  controls = QtClippingPlanesControls(
      visual.id,
      **get_clipping_planes_data_from_visual(visual, store),
      **get_clipping_plane_gizmo_data(controller, visual.id, canvas_id),
  )
  ```

  Without it the rows have no toggle. Nothing picks a visual or a canvas
  for you.

The target is read when the control is built, so build the canvas first:
the layout flows do, and a dock built before its canvas raises. A canvas
added later neither adds nor removes the toggle.

A drag is announced by `ClippingInteractionEvent` (`controller.
on_clipping_interaction`): one start and one end, with the plane changes
between them. A script can group its own changes the same way with
`with controller.clipping_interaction(visual.id): ...`. The plane a view's
gizmo is on is announced by `ClippingPlaneGizmoChangedEvent`
(`controller.on_clipping_plane_gizmo_changed`).

A plane change during a drag costs the same as any other. On a multiscale
volume of about a million level-0 bricks that is 13 to 38 ms a frame.

Known limit: with outlines enabled, a handle over an outlined visual gets a
thin contour in that visual's outline colour.

## In an OrthoViewer

One `add_*` call makes four visuals, one per panel. They read one store, so
one tuple of planes is the same cut in each. Which of them carry the same
tuple is the viewer's link mode:

| Mode | Linked |
|---|---|
| `"all"` (default) | xy, xz, yz and the 3D panel carry one tuple. |
| `"2d"` | xy, xz and yz carry one tuple. The 3D panel has its own. |
| `None` | No link. Each panel's visual has its own planes. |

```python
ortho = OrthoViewer(axes, link_clipping_planes="all")
visuals = ortho.add_image(store, clipping_planes=(plane,))

visuals["vol"].clipping_planes = (moved,)   # "all": the 2D panels follow
ortho.clipping_controller.mode = "2d"       # at any time
```

- The mode is for the whole viewer. Links are within the panels of one
  `add_*` call, never between two datasets.
- `add_*(clipping_planes=...)` gives the same planes to all four panels in
  every mode. The mode says what happens after.
- Linked visuals are changed through the controller, one
  `ClippingPlanesChangedEvent` each, with the `source_id` of the change
  that caused them. No plane is stored outside the visuals.
- **To a less linked mode** nothing is written: the panels keep their
  planes and may differ from then on.
- **To a more linked mode** the panels that become linked must agree, and
  the 3D panel's planes win, because it has the gizmo. From `None` to
  `"2d"` there is no 3D panel among them and the xy panel's planes win.
- A drag is forwarded: while a plane of one panel is dragged, each panel
  linked to it reports the drag too (`ClippingInteractionEvent`, one start
  and one end). When the dragged panel's drag ends because the handle was
  held still (`"settle"`), the linked panels end with `"release"`.
- The gizmo (`ortho.add_clipping_plane_gizmo`) edits a plane of the 3D
  panel's visual. In `"2d"` and `None` a plane that only a 2D panel has is
  not one of them.

### The dock

On an `OrthoViewer` the clipping controls are not in the appearance docks.
They are in their own dock node:

```python
ortho.add_image(
    store,
    name="cells",
    controls=InMemoryImageControlsConfig(clipping_controls=True),
)
layout = Layout(center=grid, right_dock=OrthoClippingControls())
```

At the top is a "Link" selector for the mode, bound both ways to
`ortho.clipping_controller.mode`. Under it, each visual added with
`clipping_controls=True` has one control per group of linked panels:

| "Link" | Controls per visual | "Gizmo" toggle |
|---|---|---|
| All views | "cells: All views" | Yes |
| 2D views | "cells: 2D views", "cells: 3D view" | Only "3D view" |
| Not linked | "cells: XY", "cells: XZ", "cells: YZ", "cells: 3D view" | Only "3D view" |

- A control only writes to panels that are linked, so it always shows the
  planes of every panel it edits.
- `clipping_controls=True` with no `OrthoClippingControls()` in the layout
  raises `ValueError`. The dock with no flagged visual is fine: it shows
  the selector and "No visuals with clipping controls".
- A visual that has lost a panel (`controller.remove_visual` on one of the
  four) keeps controls for the panels that are left.

### Save and load

The planes of each panel are saved. The mode is not: pass it again.

```python
ortho.to_file("viewer.json")
ortho = OrthoViewer.from_file("viewer.json", link_clipping_planes="2d")
```

A load does not rewrite what was saved. A file whose panels differ where
the mode would link them (saved in `"2d"` with the 3D panel's planes moved,
loaded with the default `"all"`) raises `ValueError`, naming the visual.
Load it with the mode it was saved in. The `controls=` configs are not in
the file, so after a load the dock lists nothing until visuals are added
with configs again.

See `examples/clipping_planes/clipping_planes_viewer.py`,
`examples/clipping_planes/clipping_plane_gizmo.py`, for multiscale
visuals `examples/clipping_planes/multiscale_clipping_planes_viewer.py`,
and for an `OrthoViewer`
`examples/clipping_planes/ortho_clipping_planes_viewer.py`.
