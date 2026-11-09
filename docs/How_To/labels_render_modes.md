# Choose how labels are drawn in 3D

A labels visual in a 3D view is drawn in one of several render modes. They
differ in how the surface of each label looks: made of voxel faces, shaded
smoothly, or a smooth surface of its own. Every mode works for labels in
memory and multiscale.

## The short version

```python
from cellier.visuals import InMemoryLabelsAppearance

visual = viewer.add_labels(
    store,
    InMemoryLabelsAppearance(render_mode="smooth_geometry"),
)
```

For a multiscale store, the same field is on `MultiscaleLabelsAppearance`:

```python
from cellier.visuals import MultiscaleLabelsAppearance

visual = viewer.add_labels_multiscale(
    store,
    MultiscaleLabelsAppearance(render_mode="smooth_iso"),
)
```

The mode only changes the 3D view. A 2D view of labels looks the same in
every mode.

## The modes

| `render_mode` | Surface | Shading | Cost |
|---|---|---|---|
| `"iso_categorical"` (default) | Voxel faces | Lit by the face normals | 1x |
| `"flat_categorical"` | Voxel faces | Unlit, the label's colour as it is | 1x |
| `"smooth_iso"` | Voxel faces | Lit by a normal that varies smoothly across the faces | about 1x |
| `"smooth_geometry"` | A smooth surface per label | Lit by the same smooth normal | several times |
| `"plane"` | On the visual's render planes | Unlit | see [Draw a volume on planes](render_planes.md) |

- **`"iso_categorical"`** shows the data exactly as it is: every voxel is a
  little cube, and each face is lit by which way it faces. Use it when the
  voxels themselves matter.
- **`"flat_categorical"`** draws the same faces in the label's flat colour.
  Shapes are hard to read from the colour alone; add an
  [outline](outlines.md) or ambient occlusion to give them depth.
- **`"smooth_iso"`** draws the same faces as `"iso_categorical"` -- the same
  pixels, at the same depth -- but lights them as if the surface were
  smooth. A round label looks round, while its outline against the
  background stays on the voxel grid. It costs about the same as
  `"iso_categorical"`.
- **`"smooth_geometry"`** replaces the voxel faces with a smooth surface for
  each label, so the outline and the depth are smooth too. See
  [Smooth geometry](#smooth-geometry) below.
- **`"plane"`** draws the labels on flat planes instead of as a volume.

The multiscale appearance has one mode more, `"gradient_debug"`, which
draws the surface normal as a colour to diagnose shading problems at brick
boundaries. It is left out of the render-mode control.

`"smooth_iso"` on a labels visual is not the image `"smooth_iso"` mode. The
image mode smooths an iso-surface of intensities at a threshold; the labels
mode smooths the shading of label surfaces and has no threshold.

## Changing the mode later

```python
controller = viewer.controller
controller.update_appearance_field(visual.id, "render_mode", "smooth_geometry")
```

A change of mode rebuilds the visual's shader, once.

In the dock, a labels visual's "Render mode" control lists every mode but
`"gradient_debug"`. Ask for it, and for the smooth-geometry level below,
with `FieldControls`:

```python
from cellier.convenience import FieldControls

viewer.add_labels(
    store,
    controls=[FieldControls(["visible", "opacity", "render_mode", "smooth_geometry_level"])],
)
```

## Smooth geometry

In `"smooth_geometry"` each label is turned into a smooth shape before it
is drawn. Place a sample at the centre of every voxel; at any point in the
volume, each label gets the share of the surrounding voxel centres that
carry it, weighted by how close they are (trilinear weights). A point
belongs to a label where its share is above a level, 0.5 by default.

What that does to the shape:

- **Flat sides stay where they were.** Halfway between a voxel of the label
  and a background voxel the share is exactly 0.5, so at the default level a
  flat side sits on the voxel faces, as in the other modes.
- **Convex corners and edges round off.** At the corner of a cube only one
  of the eight surrounding voxels carries the label, so the corner is cut.
- **Concave corners fill in.** Where seven of eight voxels carry the label,
  the eighth voxel's corner is filled.
- **A single voxel still draws**, as a small rounded blob: at a voxel's own
  centre its label has the whole share.
- **Touching labels are rounded one by one.** Where two labels meet, each
  is rounded on its own, which leaves a small groove along their shared
  edge. The colour changes along a smooth line.

The surface is found exactly, cell by cell between voxel centres, rather
than by stepping along each ray. Thin structures are not stepped over, and
on a multiscale visual `ray_steps_per_voxel` has no effect in this mode.

### The level: `smooth_geometry_level`

```python
InMemoryLabelsAppearance(render_mode="smooth_geometry", smooth_geometry_level=0.35)

controller.update_appearance_field(visual.id, "smooth_geometry_level", 0.4)
```

The share of the weight a label must carry for a point to belong to it, between
0.05 and 0.95.

- **0.5**, the default, keeps flat sides on the voxel faces.
- **Lower** grows every label. Voxels that touch only along an edge or at a
  corner join up, a one-voxel-thick sheet seen at a slant has fewer holes,
  and the grooves between touching labels narrow. Around 0.35 is a good
  start for thin structures.
- **Higher** shrinks every label, and thin parts break up first.

The level is sent straight to the GPU, so dragging its slider redraws
without rebuilding anything. It is kept, and has no effect, in the other
modes.

### Picking

A smooth surface is not on a voxel face, and where it fills in a concave
corner it lies in a background voxel. A pick on it reports a point in the
middle of the nearest voxel of the label that was drawn, rather than the
exact point on the surface, so the picked label is always the one under the
cursor.

### Cost

`"smooth_geometry"` does far more work per pixel than the other modes. As a
rough guide, a 1600 x 1600 frame that took about 3 ms in
`"iso_categorical"` took about 8 ms in `"smooth_geometry"`, in memory and
multiscale alike. `"smooth_iso"` costs about the same as
`"iso_categorical"`.

## With outlines, clipping planes and ambient occlusion

- **Outlines** work in every mode. The [depth outline](outlines.md#a-line-where-one-part-passes-in-front-of-another)
  outlines the steps of a voxel surface when the voxels are large on
  screen, and `"smooth_iso"` keeps those steps, since its faces are the
  voxel faces. A `"smooth_geometry"` surface has no voxel steps.
- **Clipping planes** cut every mode. Where a plane cuts through a label the
  cut face lies on the plane and is lit by the plane's normal. In
  `"smooth_geometry"`, within a voxel or two of a label's edge, the cut
  meets the label's own rounded surface instead.
- **Ambient occlusion** reads the normal each mode lights with, so in the
  two smooth modes it gets the smooth normal.

## Limits

- **Multiscale, coarse levels.** Every mode draws the level that is
  loaded. While a coarse level is on screen, a smooth surface is a smooth
  version of the coarse voxels: larger bumps that settle as the finer level
  arrives.
- **Multiscale, empty bricks.** Bricks with no label are skipped. Where a
  filled-in concave corner would reach past the face of a labelled brick
  into an empty one, that thin sliver is not drawn.
- **The edge of the volume.** Outside the volume counts as background, so
  in `"smooth_geometry"` a label that reaches the edge of the volume is
  closed by a surface on the box face, with its edges rounded.
