# Outline a visual

An outline is a line along the edge of a visual on screen. It is drawn after
the scene, from what each pixel shows, so it follows a mesh, a volume, points
or labels alike and costs the same whatever the data.

A labels visual has more than one way to be outlined: round the whole volume,
round each label, and, in 3D, where one part of a label passes in front of
another part of the same label. That last one is the
[depth outline](#a-line-where-one-part-passes-in-front-of-another).

## The short version

```python
from cellier.convenience import Viewer
from cellier.render import OutlineConfig, RenderManagerConfig
from cellier.visuals import MeshFlatAppearance, VisualOutline

viewer = Viewer(
    world,
    displayed_axes=("z", "y", "x"),
    render_config=RenderManagerConfig(outline=OutlineConfig(enabled=True)),
)
viewer.add_mesh(
    store,
    MeshFlatAppearance(color=(0.4, 0.7, 1.0, 1.0)),
    outline=VisualOutline(slot=1),
)
```

Two things are needed: the outline pass is on (`OutlineConfig(enabled=True)`),
and the visual has a slot. Turn the pass on when the viewer is built, as
here.

## Slots, colours and placement

- **`slot`** chooses the colour: slot `n` is drawn in entry `n - 1` of
  `OutlineConfig.palette`. Slot 0, the default, is no outline. Two visuals
  in one slot share a colour.
- **`placement`** says which side of the visual's edge the line is on.
  - `"inward"`: inside the visual. The visual never looks larger, but one
    thinner than twice the line is covered by it.
  - `"outward"`: outside it, as a halo.
  - `None`, the default: outward for points, lines and graphs, inward for
    everything else.
- **The thickness** is not per visual. It belongs to the pass:
  `OutlineConfig.selection.inward_thickness` and `outward_thickness`. It is
  in *internal* pixels, and the renderer draws at twice the size of the
  screen by default, so a thickness of 2 is 1 pixel on screen.

```python
from cellier.render import OutlineLayerConfig

OutlineConfig(
    enabled=True,
    selection=OutlineLayerConfig(inward_thickness=2, outward_thickness=2),
    palette=[(0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0)],   # slots 1 and 2
)
```

`OutlineConfig` also has a second, thin layer drawn round every outlined
visual (`boundaries`) and a contrast band beside the line
(`inner_thickness`, `inner_color`), both on by default. Turn them off for
one plain line:

```python
OutlineConfig(
    enabled=True,
    boundaries=OutlineLayerConfig(enabled=False),
    inner_thickness=0,
)
```

An outline is read from the pick buffer, so an outlined visual has
`pick_write` on. Asking for an outline on a visual with `pick_write=False`
turns it back on, with a warning.

## Changing an outline later

```python
controller = viewer.controller

controller.set_visual_outline(visual.id, slot=2, placement="outward")
controller.set_visual_outline(visual.id, slot=0)            # no outline

controller.update_render_config_field("outline", "selection.outward_thickness", 4)
controller.outline_enabled = False
```

A change of thickness recompiles the outline shader, once; colours and
slots do not.

## Labels: four modes

A labels visual takes `outline_mode`:

| `outline_mode` | What is outlined | Colour |
|---|---|---|
| `"per_label"` (default) | The labels you select, each on its own. Touching labels keep a line between them. | Per label: `outline_selected_labels` |
| `"whole_object"` | The volume as one shape. No lines between labels. | The visual's slot |
| `"all_boundaries"` | Every label, lines between touching labels included. | The visual's slot |
| `"depth_threshold"` | As `"all_boundaries"`, and in 3D also where one part of a label passes in front of another part of it. | The visual's slot |

```python
viewer.add_labels(
    store,
    outline=VisualOutline(slot=1),
    outline_mode="all_boundaries",
)

# per_label: which labels, and in which slot each.
visual = viewer.add_labels(
    store,
    outline=VisualOutline(slot=1),
    outline_selected_labels={3: 1, 7: 2},
)
controller.set_label_selection(visual.id, {3: 1})

# The mode can be changed while the visual is shown.
controller.update_visual_render_field(visual.id, "outline_mode", "whole_object")
```

- In `"per_label"` mode the visual's own slot only says that the volume
  takes part; the colours come from the selection.
- In the other three modes the selection has no effect. It is kept, and
  comes back when the mode is `"per_label"` again.
- With `placement="outward"` the line between two touching labels is twice
  as wide, since each label casts one onto the other. Inward is the default
  for labels.

## A line where one part passes in front of another

A segmentation of a tree (airways, vessels, neurons) is often **one label**.
Drawn in 3D in one colour with an outline round it, its branches cannot be
told apart where they overlap on screen: there is no edge between two parts
of the same label.

`"depth_threshold"` adds that edge. It draws a line wherever the depth
under two neighbouring pixels of one label jumps: along the edge of the
nearer surface where it hides a farther one.

```python
visual = viewer.add_labels_multiscale(
    store,
    outline=VisualOutline(slot=1, placement="outward"),
    outline_mode="depth_threshold",
)
```

It works for labels in memory and multiscale, in every surface render mode.

What it draws, all in the colour of the visual's slot:

1. the silhouette of the visual,
2. the boundary between two different labels, at any depth,
3. a jump in depth inside one label.

What it does not draw: a line between two branches that join at the same
depth, or along a branch that curves smoothly away. It follows depth, not
anatomy.

**Which side the line is on** follows the placement, as for any outline:

- `"inward"`: on the nearer surface, along its edge.
- `"outward"`: on the farther surface, just outside the edge of the nearer.
  Use this for thin structures, so that a branch a few pixels wide is not
  covered by its own outline.

### How deep a jump is drawn: `outline_depth_range`

```python
visual = viewer.add_labels_multiscale(
    store,
    outline=VisualOutline(slot=1, placement="outward"),
    outline_mode="depth_threshold",
    outline_depth_range=(0.01, 0.06),      # the default
)
controller.update_visual_render_field(visual.id, "outline_depth_range", (0.02, 0.12))
```

The range is a pair, `(start, solid)`. Both are fractions of the distance
from the camera to the nearer of the two surfaces.

- A jump of **start** is where a line begins, with no strength at all.
- From **solid** up the line is drawn in full.
- Between the two the line fades in.
- `start == solid` gives a hard edge: every jump over it is drawn in full.

With the default, a branch 1% of the camera's distance in front of another
gets no line, one 3.5% in front gets a line at half strength, and one 6% or
more in front gets a solid one. So a deep overlap reads as a strong line and
a shallow one as a faint line.

- **To calm a busy picture**, raise both values: `(0.02, 0.12)` leaves
  about half the lines of the default in a dense tree seen from far away.
- **For lines on shallower overlaps**, lower the start. The lower it is,
  the sooner the steps of the voxel surface show (see the limits below).
- The values are relative to the camera's distance, so a line grows in as
  the camera moves toward a crossing and fades as it backs away.
- The range is per visual. It is kept, and has no effect, in the other
  modes.
- Set it through `add_labels...` or `update_visual_render_field`. A pair
  with a start outside `(0, 1]`, or a solid value below its start, is
  refused there.

In the outline control of a labels visual the mode is "Depth threshold",
and the range is the row below it, "Start" and "Solid".

### Limits

**Voxel steps.** The surface of a labels volume is made of voxel faces.
Where one voxel hides the next there is a real jump in depth, and nothing
can tell it from a small crossing. The steps are outlined when a voxel is
large on screen:

- zoomed in past the finest level of the data;
- for as long as a coarse level of a multiscale volume is on screen: while
  a view loads, lines appear on the coarse blocks and go when the fine
  level arrives;
- for anisotropic data seen along its long axis, sooner than the size of a
  voxel on screen suggests.

At the default range the steps show faintly from about 6 internal pixels a
voxel (3 on screen, see the thickness above) and plainly from about 13. A
larger start removes them: the start at which they begin is roughly 2.5
times the size of a voxel on screen over the height of the canvas, both in
the same pixels. On a surface seen at a slant they begin sooner.

**The line's thickness.** Step lines grow with the thickness of the line,
and the contrast band (`inner_thickness`) shows them sooner than the line
itself does. With large voxels, a thin line and no contrast band is
cleanest.

**2D and plane mode.** There is no jump in depth in a 2D view, or on a
render plane. There the mode draws what `"all_boundaries"` draws.

**A very wide depth range.** The test needs the depth buffer to tell the two
surfaces apart. With the far plane a few thousand times the near plane away
(the default) it has all the room it needs. With a surface more than about
50000 times the near plane's distance away, a start near 0.005 is inside the
noise of the depth buffer and lines appear on smooth surfaces. Keep the
canvas's `depth_range_3d` no wider than the scene needs.

**Something in front that is not picked.** The line is found from the depth
under each pixel, the label from the pick buffer. A visual drawn in front of
the labels with `pick_write=False` writes depth and no pick, so its shape is
outlined *on the labels*, in the labels' colour. Leave `pick_write` on for
anything drawn in front of labels in this mode.

**Two labels visuals holding the same label value** are one region to the
outline: there is no line between them in any labels mode. In
`"depth_threshold"` mode a jump in depth between them is drawn, with the
range of the nearer one.

**The camera.** The range is a fraction of the distance from the camera,
which means something with a perspective camera. cellier's 3D camera is
one.
