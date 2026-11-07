# Draw a volume on planes

An image or labels visual in the `"plane"` render mode draws its data on up
to four flat planes in a 3D view, instead of as a volume rendering. It works
for images and labels, in memory and multiscale. Each visual has its own
planes.

## The short version

```python
from cellier.visuals import InMemoryImageSingleAppearance, RenderPlane

viewer = Viewer(spatial_axes("z", "y", "x"), displayed_axes=("z", "y", "x"))
plane = RenderPlane.from_point_normal(
    viewer.scene.dims.world_coordinate_system,
    point=(64, 64, 64),
    normal=(1, 0, 0),
    axes=("z", "y", "x"),
)
visual = viewer.add_image(
    store,
    single=InMemoryImageSingleAppearance(render_mode="plane"),
    render_planes=(plane,),
)
```

Two things are needed, and neither sets the other:

- **the render mode** `"plane"`, on the appearance;
- **the planes**, `render_planes`, on the visual.

`render_planes` is ignored in every other render mode, and changing the mode
does not touch the tuple. Plane mode with no enabled plane draws nothing.

## The render mode

`"plane"` is a value of `render_mode`, beside the volume modes:

| Visual | Where |
|---|---|
| In-memory image | `InMemoryImageSingleAppearance.render_mode`, and each channel's |
| Multiscale image | `MultiscaleImageSingleAppearance.render_mode`, and each channel's |
| In-memory labels | `InMemoryLabelsAppearance.render_mode` |
| Multiscale labels | `MultiscaleLabelsAppearance.render_mode` |

**Every channel of an image has the same render mode**, hidden channels
included. A `channels` dict that mixes modes is refused, for the volume
modes too (`mip` beside `iso`). Change the mode of all channels at once:

```python
controller.set_image_render_mode(visual.id, "plane")
```

Assigning the mode to one channel of several is refused and put back. A
control's per-channel "Render mode" combo goes through
`set_image_render_mode`, so the combos move together. `single.render_mode`
is separate: it is what the visual uses when it is not in composite mode.

## A plane is in world coordinates

A clipping plane is written in a store's data coordinates. A render plane is
written in the **scene's world coordinates**, on three named world axes:

```python
class RenderPlane:      # frozen
    id
    coordinate_system   # the scene's world system
    axes                # three world axes
    origin              # a point on the plane, on `axes`
    in_plane_axis_0     # unit, orthogonal; the normal is their cross product
    in_plane_axis_1
    extent_0            # (min, max) from the origin along in_plane_axis_0
    extent_1            # (min, max) from the origin along in_plane_axis_1
    enabled
```

- A plane has a frame and a rectangle, and those are in world units: a
  rectangle of 100 by 100 is 100 world units a side whatever the voxel size.
- **A plane does not follow its visual's transform.** Move the visual and
  its planes stay where they are in the world. A visual translated along x
  needs planes written at that x.
- The same plane can be given to two visuals of one scene (an image and its
  labels): a plane is a value, and each visual keeps its own tuple.
- A plane constrains only its three axes. A `zyx` plane in a `tzyx` scene
  holds at every time point, and the time slider chooses the volume it cuts.
- The visual's transform on those three axes must be a per-axis scale and
  translation. A rotated or sheared visual refuses render planes.

Two constructors:

```python
world = viewer.scene.dims.world_coordinate_system

# The frame is chosen for you.  `up` fixes its spin.
RenderPlane.from_point_normal(world, point, normal, axes=("z", "y", "x"), up=None)

# The frame is yours: a 3 x 3 rotation matrix or a quaternion whose columns
# are the two in-plane axes and the normal.
RenderPlane.from_rotation(world, origin, rotation, axes=("z", "y", "x"))
```

Both take `extent_0`, `extent_1`, `enabled` and `id`.

### Extents

Each of the four sides is finite or unbounded on its own:

```python
RenderPlane.from_point_normal(
    world, point, normal, axes=axes,
    extent_0=(-50.0, 50.0),   # 100 world units along in_plane_axis_0
    extent_1=(None, 20.0),    # to the edge of the data on one side
)
```

`None` is unbounded: the plane ends at the data's bounding box there. A plane
is always cut by the data box and by the visual's `clipping_planes`. A
bounded pair needs `min < max`.

On a multiscale visual, extents and clipping planes reduce what is read, not
only what is drawn: bricks the rectangle does not reach are not loaded.

## Changing planes

Planes are frozen. Assign a new tuple; one assignment is one event
(`RenderPlanesChangedEvent`).

```python
moved = plane.model_copy(update={"origin": (80.0, 64.0, 64.0)})   # keeps the id
visual.render_planes = (moved,)                                   # move
visual.render_planes = (*visual.render_planes, another)           # add
visual.render_planes = ()                                         # remove all
```

`controller.set_render_planes(visual_id, planes, source_id=...)` does the
same with a `source_id` on the event. To replace one plane and leave the
others, by its id:

```python
controller.set_render_plane(visual.id, plane.id, moved)
controller.get_render_plane(visual.id, plane.id)
```

A plane's `id` survives `model_copy`, so a gizmo or a control row follows a
plane through replaced tuples. `enabled=False` keeps a plane in the tuple
and draws nothing of it. Adding or removing a plane is a uniform write, not
a shader compile: a visual always has room for four.

A refused assignment is put back to the last good tuple and the error is
raised. What is refused:

- a plane in another coordinate system than the scene's world, or on axes
  that are not the world's;
- two planes with one id; more than four planes;
- a visual whose transform on the plane's axes is not a scale and
  translation.

## When the planes are not drawn

| Situation | What is drawn |
|---|---|
| A 2D view | The normal 2D slice. The mode and the planes are ignored. |
| A 3D view that displays other axes than a plane's `axes` | Nothing of that plane, and one warning per visual. |
| Plane mode, no planes or all disabled | Nothing. |

A visual that draws nothing loads nothing. It loads when a plane appears.

The scene's bounding box, the camera fit and the orbit pivot use the whole
data, not the planes.

## How it looks

- **One sample per pixel**, where the view ray meets the nearest plane: no
  ray marching, so a frame costs the same for one plane or four.
- The visual's own `interpolation`, colour map and contrast limits. Image
  pixels below the contrast limits are drawn (the plane is not see-through
  where the data is dark). The background of labels is transparent.
- The visual's own depth and blending settings; the plane adds no rule.
  - A single-channel image plane at the default settings is opaque and
    occludes correctly against meshes and other planes.
  - **A composite with more than one channel drawn is additive and writes no
    depth**, exactly as a composite volume is: what is behind the plane
    shows through it, and where two planes of the visual cross, the far one
    shows through the near one.
- Both sides of a plane look the same. Flipping the normal changes nothing
  drawn. It changes which way the gizmo's arrow points and which way the
  control's position slider runs.
- **Picking** returns the same typed pick events as a volume: the position
  hit, and for labels the label at that voxel. A plane cut away by a
  clipping plane or outside its rectangle is not picked.
- Outlines and ambient occlusion work on a plane as on a mesh in the same
  place.
- With `interpolation="nearest"`, a plane lying exactly on a voxel face (a
  half-integer position on an axis-aligned plane of unit voxels) shows the
  voxel on either side pixel by pixel, and looks speckled. Put the plane on
  a voxel centre, or use linear interpolation.

## An outline round a plane

A plane can have an outline: a line along the edge of what is drawn of it.

```python
from cellier.visuals import PlaneOutline

plane = RenderPlane.from_point_normal(
    world,
    point=(60, 256, 256),
    normal=(1, 0, 0),
    axes=("z", "y", "x"),
    outline=PlaneOutline(enabled=True, color=(1.0, 0.8, 0.2, 1.0), width=2.0),
)
```

| Field | Meaning | Default |
|---|---|---|
| `enabled` | Whether the line is drawn | `False` |
| `color` | RGBA, each from 0 to 1 | opaque white |
| `width` | The line's width in screen pixels | `2.0` |

The line follows the polygon the plane makes in the visual: the plane cut
by its extents, by the data's bounding box and by the visual's enabled
clipping planes. So it shows where the plane is where the data is dark,
where labels are background, and before a multiscale visual has loaded
anything: the outline does not wait for data.

An outline is part of its plane, so it is changed as a plane is moved:

```python
outlined = plane.model_copy(
    update={"outline": PlaneOutline(enabled=True, color=(1.0, 0.0, 0.0, 1.0))}
)
controller.set_render_plane(visual.id, plane.id, outlined)
```

- **A change of the outline alone loads nothing.** It is announced with
  `RenderPlanesChangedEvent`, as every change of the tuple is, and drawn.
- The outline is drawn while the plane is: the plane and the visual shown,
  a 3D view of the plane's axes, the `"plane"` render mode.
- **A plane two visuals share has one outline** when the plane ends in the
  same place in both (the same data box, the same clipping planes).
  Otherwise each visual's part of the plane has its own.
- The line is depth tested: a mesh in front hides it. It is not picked.
- A visual that writes no depth (a maximum projection, an additive
  composite) does not hide the line. The line is drawn after the visuals.
- Turning an outline off keeps its colour and width.

A clipping plane has an outline too, round its cut face: see [Clip a
visual with planes](clipping_planes.md#an-outline-round-a-cut).

## The gizmo

```python
gizmo = viewer.add_render_plane_gizmo(visual, visual.render_planes[0])
...
gizmo.close()            # or viewer.remove_plane_gizmo()
```

Without a viewer:
`controller.add_render_plane_gizmo(visual_id, canvas_id, plane_id)`.

- The arrow along the normal slides the plane. The two rings that tilt the
  normal turn it.
- **The arrow is the normal.** It points the way the normal points, from
  whichever side the camera looks, so it turns over when the plane is
  flipped and can point away from you. The in-plane handles point along
  `in_plane_axis_0` and `in_plane_axis_1` in the same way.
- The in-plane arrows and the in-plane square move `origin` in the plane,
  and the rectangle with it. The ring about the normal spins the rectangle.
- The two in-plane scale handles resize the rectangle **symmetrically about
  the origin**. An axis with an unbounded side has no scale handle: make the
  side finite first (in the control, or from code). A one-sided resize is
  done in the control.
- **A view has one gizmo at a time, clipping planes included.** Adding one
  on another plane, of either kind, closes the one the view had.
  `controller.get_plane_gizmo(canvas_id)` returns it; its `kind` is
  `"render"` or `"clipping"`. `PlaneGizmoChangedEvent`
  (`controller.on_plane_gizmo_changed`) announces a change.
- The gizmo closes itself when its plane or its visual is removed, when the
  visual leaves plane mode, when the view switches to 2D, and when the view
  stops displaying the plane's axes.
- A disabled plane can have a gizmo: place it, then switch it on.

The gizmo needs the plane on screen, so add it after the first data is
drawn: `viewer.on_ready(lambda: viewer.add_render_plane_gizmo(visual, plane))`.

## The control

Put `RenderPlanesControl()` in the `controls=` list of an image or labels
visual. The dock shows the controls in the order of the list:

```python
viewer.add_image_multiscale(
    store,
    single=MultiscaleImageSingleAppearance(render_mode="plane"),
    render_planes=(plane,),
    controls=[
        ImageControl(fields=["visible", "color_map", "clim", "render_mode"]),
        BoundingBoxControl(),
        RenderPlanesControl(),
    ],
)
layout = Layout(center=canvas, right_dock=AppearanceControls())
```

The "Render planes" group is the "Clipping planes" group with the extents
added. A row per plane:

- on or off, "Gizmo", "Flip", "Remove";
- the normal: a `+` and a `-` button per axis to face the plane along it,
  and under them the normal's entries for an oblique plane. The normal
  shown is the unit normal. An edit turns the plane about its origin by the
  smallest rotation and keeps its spin;
- a position slider along the normal. It shows the plane's **depth into
  the visual's bounding box**, in world units: 0 where a plane first
  touches the box, on the side the normal points away from, and at the
  slider's far end the size of the box along the normal, where it last
  touches it. Dragging right moves the plane the way the gizmo's arrow
  points. A flip turns a depth `d` into `span - d` and moves nothing. A
  plane taken out of the box (by the gizmo, or from code) pins the slider
  at the nearer end; the number, drawn in orange italics, is then below 0
  or above the slider's end. The depth is the control's own number: a
  plane's `origin` is in world coordinates, as before;
- per in-plane axis, a minimum and a maximum, each with an "unbounded"
  toggle (an "inf" button on Qt, a check box in a notebook). Switching it
  off fills in where the data box cuts the plane on that side, so the
  picture does not jump;
- "Outline": a box that draws the plane's [outline](#an-outline-round-a-plane)
  and its colour. The colour picker sets red, green and blue; the
  outline's alpha and width are set from code and are kept.

"Add plane" adds a plane through the centre of the data, facing the first
displayed axis, unbounded; it is disabled at four planes. The in-plane spin
has no entry: set it with the gizmo's ring or from code.

The control is disabled, with one line saying why, while the visual is not
in plane mode or the view is 2D. The "Gizmo" toggles of the render planes
and the clipping planes controls of a view act as one set.

**Choosing "Plane" in a "Render mode" combo on a visual with no planes also
adds one**, so the choice shows something. This is the control's doing:
setting the mode from code adds nothing.

The control is drawn in the appearance dock, so the layout needs an
`AppearanceControls()` dock: without one, rendering the layout raises
`ValueError`. The list may hold `RenderPlanesControl()` alone. On an
`OrthoViewer` the control is in the appearance dock too, in the 3D view's
group only: the 2D views draw no render planes.

A control built by hand:

```python
from cellier.gui import (
    get_render_plane_gizmo_data,
    get_render_planes_data_from_visual,
)
from cellier.gui.qt.visuals import QtRenderPlanesControls

controls = QtRenderPlanesControls(
    visual.id,
    **get_render_planes_data_from_visual(controller, visual.id),
    **get_render_plane_gizmo_data(controller, visual.id, canvas_id),
)
```

## A multiscale visual: what is loaded, and when

### The level of detail on a plane

A plane picks pyramid levels by its own rule: about **one screen pixel per
voxel, measured in the plane**, at `settled_lod_bias=1.0`. The coarsest
direction in the plane counts, so a pyramid that does not downsample every
axis (z kept fine, say) is not drawn coarse on a plane that contains z. A
higher bias is coarser.

- In a perspective view each brick takes its level from its distance, so the
  near part of a plane is finer than the far part.
- In an orthographic view the whole plane takes one level. If that level
  does not fit the cache, the next coarser one is taken, until it fits.
- `render_config.force_level` replaces the rule, as in every mode.

Only the bricks a plane crosses are read: a full-screen plane is a few
hundred to about a thousand bricks.

### While a plane moves

Three settings on `MultiscaleImageAppearance` and
`MultiscaleLabelsAppearance` say how every multiscale image and labels
visual loads, in every render mode:

| Field | Default | Meaning |
|---|---|---|
| `settled_lod_bias` | `1.0` | The level bias when nothing moves. 2D and 3D. |
| `coarsest_while_moving_3d` | `True` | In a 3D view, plan no detail while the visual moves. |
| `coarsest_while_moving_2d` | `False` | The same in a 2D view. |

A visual **moves** while its planes are dragged (render planes or clipping
planes: one tracker per visual serves both) or its scene's dims are
scrubbed. With the setting on:

| Motion | During | At the end |
|---|---|---|
| A plane drag (a gizmo, or `controller.plane_interaction`) | Nothing is planned or read. The bricks already on the GPU are drawn, and the coarse backstop where a plane reveals more | The full detail, once |
| A dims scrub | The backstop of each new slice | The full detail, once |

A motion ends on release, or when the input has been still for 0.15 s
(`SchedulerConfig.dims_settle_s`), so holding a handle still refines under
the cursor. A slide keeps the fine picture while the plane stays in the
bricks it had; a rotation loses it sooner, at the far edge first.

With the setting off, every tick plans the full detail.

A drag from code:

```python
with controller.plane_interaction(visual.id):
    for position in positions:
        controller.set_render_plane(visual.id, plane.id, moved_to(position))
        await asyncio.sleep(1 / 30)
```

Every change of `render_planes` or `clipping_planes` inside the block is a
tick of one drag, announced by `PlaneInteractionEvent`
(`controller.on_plane_interaction`): one start and one end. Outside a block
every change is a jump and plans at once. The settings are plain fields:
assigning one plans nothing, and the next plan reads it.

The "Level of detail" control (`"level_of_detail"` in a controls config's
`appearance` list) has the bias slider and a "Coarsest while moving"
checkbox, which edits the 3D field in a 3D view and the 2D field in a 2D
view.

An in-memory visual holds its whole texture: nothing is planned or read, the
level settings do not apply, and a plane change is a uniform write and a
draw.

### The cache budget

A plan that does not fit `render_config.gpu_budget_bytes` (1 GiB by default)
is cut short nearest first: the far part of the plane stays on the coarse
backstop. One single-channel visual fits in every pose, with three
orthogonal full-screen planes too.

**The budget is split between a composite's drawn channels**, so a composite
reaches it in ordinary use: with four channels drawn, an oblique full-screen
plane already drops bricks, and three planes drop many. If the far part of a
plane stays blocky:

- raise `gpu_budget_bytes`;
- draw fewer channels;
- bound the planes' extents, or clip, so less is planned;
- raise `settled_lod_bias`.

### A level change

When a zoom changes the level a plane is drawn at, the plane drops to the
coarse backstop until the new level has loaded, although the old level is
still on the GPU. It lasts a third of a second in a perspective view and
longer in an orthographic one, where the whole plane changes level at once.
This is how progressive loading draws every multiscale visual today; it is
more visible on a plane.

## In an OrthoViewer

One `add_*` call makes four visuals, one per panel. `render_planes=` goes to
the 3D panel's visual only. `render_mode="plane"` goes to all four and acts
in the 3D panel alone, since a 2D view ignores it.

```python
ortho = OrthoViewer(spatial_axes("z", "y", "x"))
visuals = ortho.add_image(
    store, single=InMemoryImageSingleAppearance(render_mode="plane")
)
ortho.plane_controller.mode = "slices"
```

| `plane_controller.mode` | Behaviour |
|---|---|
| `None` (default) | No link. The 3D panel's visuals keep the planes they are given. |
| `"slices"` | Every image and labels visual of the 3D panel carries exactly three planes, one per 2D panel, at that panel's slice position and facing the axis it slices. |

In `"slices"`:

- Turning the mode on **replaces** the planes of the 3D panel's image and
  labels visuals with the three slice planes, whatever their render mode.
  A visual not in plane mode carries them and draws them once it is. The
  mode sets no render mode.
- **A slider moves its plane.** While a 2D slider is dragged, each visual
  of the 3D panel holds one plane interaction: a multiscale visual keeps
  the bricks it has and loads once when the slider is released.
- **A plane changed from code is carried over** when the tuple is still
  three slice planes. A plane moved along its normal moves its panel's
  slider; its extents, `enabled` flag and spin are copied to the other
  visuals of the 3D panel. So `enabled=False` on one plane hides that slice
  everywhere, and stays hidden as the sliders move. An outline is copied
  the same way.
- **The slices' outlines** are set on the plane controller, since the
  render planes control shows the slice planes read-only:

  ```python
  ortho.plane_controller.outlines = (
      PlaneOutline(enabled=True, color=(1.0, 0.8, 0.2, 1.0)),  # the xy panel's
      PlaneOutline(enabled=True, color=(0.3, 0.9, 1.0, 1.0)),  # xz
      PlaneOutline(enabled=True, color=(1.0, 0.4, 0.8, 1.0)),  # yz
  )
  ```

  One `PlaneOutline` is given to all three planes; `None` is no outline.
  Each slice has one line in the 3D panel, however many visuals carry the
  planes.
- **A tuple that is not three slice planes** (a plane tilted, added, removed
  or replaced) is put back, with one warning. The assignment itself raises
  nothing.
- `ortho.add_render_plane_gizmo` is refused, and the render planes control
  shows the three rows disabled, with the reason.
- Turning the mode off writes nothing: the planes stay, free to edit.

No plane is stored outside the visuals. The planes are saved with the
viewer; the mode is not, so set it again after `OrthoViewer.from_file`.

`ortho.clipping_controller` links clipping planes as before; see
[Clip a visual with planes](clipping_planes.md#in-an-orthoviewer).

## Examples

Under `examples/plane_rendering/`, each with a marimo twin (`*_marimo.py`):

| Script | Shows |
|---|---|
| `plane_gizmo.py` | A multiscale image on one plane, the gizmo with its extent handles, the render planes and level-of-detail controls |
| `plane_interaction_script.py` | A render plane and a clipping plane moved from code as one interaction, loaded once at the end |
| `plane_labels_composite.py` | In-memory labels and a two-channel composite image on planes |
| `ortho_slice_planes.py` | An `OrthoViewer` in `"slices"` mode, each slice with an outline in its own colour |
| `plane_slicing_validation.py` | A `tczyx` scene that checks itself: points, multiscale labels and a multiscale composite image that must line up on a plane, in one colour per octant and time point, with voxels that are not cubes. The shared plane has one outline |
