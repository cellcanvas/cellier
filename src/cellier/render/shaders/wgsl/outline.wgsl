// Screen-space outline composite.
//
// Runs as the fragment stage of a full-quad effect pass.  It reads the
// renderer's pick buffer, derives a per-pixel outline key, tests four
// neighbouring taps at the configured thickness, and composites a
// coloured outline over the rendered image.  One pass, no intermediate
// textures.
//
// The LUT entry's `kind` decides two independent things: what the key is,
// and where the colour comes from.
//
//     kind             key            colour
//     WHOLE_OBJECT     pick id        the visual's own slot
//     LABEL            label field    the key's own selection range
//     LABEL_ALL        label field    the visual's own slot
//
// LABEL_ALL is the fourth cell: every label in one volume banded in a
// single colour, boundaries between touching labels included.
//
// A visual whose entry carries a depth range also gets **depth edges**
// (has_depth_edges): a band where two pixels of the same key lie on
// surfaces a depth jump apart, one part of the visual passing in front of
// another.  See "Depth edges" below.
//
// Bindings are declared by the Python side (_outline.py); this file
// assumes:
//     0 u_outline   uniform
//     1 texSampler  sampler          (unused; the pass loads, never samples)
//     2 colorTex    texture_2d<f32>
//     3 pickTex     texture_2d<u32>
//     4 lutTex      texture_2d<u32>  rg16uint, 1024x1024
//     5 outlineIdTex texture_2d<u32>  r32uint, label keys (has_outline_id)
//     6 depthTex    texture_depth_2d  the frame's depth (has_depth_edges)
//
// Template vars: b_t_in, b_t_out, s_t_in, s_t_out, inner_t,
//                has_inward, has_outward, has_outline_id, has_depth_edges.

const KIND_NONE: u32 = 0u;
const KIND_LABEL: u32 = 2u;
const KIND_LABEL_ALL: u32 = 3u;
const PLACEMENT_INWARD: u32 = 0u;
const PLACEMENT_OUTWARD: u32 = 1u;

// Label keys and object ids share a numeric space, so label keys are
// tagged with the high bit before comparison.  global_id is 20 bits and a
// label key is at most 16 + 2^28, so the tag keeps the two disjoint and a
// label can never read as "same region" as a mesh that happens to share
// its number.
const LABEL_KEY_TAG: u32 = 0x80000000u;

// The range the label key's partition reserves for selected labels.
const MAX_LABEL_SLOT: u32 = 15u;

// strength scales the alpha of whatever is drawn for the hit.  It is 1 for
// an identity edge and between 0 and 1 for a depth edge.
struct OutlineEntry {
    slot: u32,
    kind: u32,
    placement: u32,
    strength: f32,
};

fn none_entry() -> OutlineEntry {
    return OutlineEntry(0u, KIND_NONE, PLACEMENT_INWARD, 1.0);
}

// Both label kinds key on the per-pixel label rather than on the object
// id.  They differ only in where the colour comes from.
fn is_label_kind(kind: u32) -> bool {
    return kind == KIND_LABEL || kind == KIND_LABEL_ALL;
}

// Clamp taps to the framebuffer.  A tap that runs off the edge then reads
// the centre's own key, so a region touching the screen border is not
// given a spurious outline along it.
fn clamp_texel(texel: vec2<i32>) -> vec2<i32> {
    let dims = vec2<i32>(textureDimensions(pickTex));
    return clamp(texel, vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
}

// Every pygfx and cellier shader begins its pick payload with
// pick_pack(u32(u_wobject.global_id), 20), so global_id is bits 0..19.
fn pick_id(texel: vec2<i32>) -> u32 {
    let pick = textureLoad(pickTex, clamp_texel(texel), 0);
    return pick.r | ((pick.g & 0xFu) << 16u);
}

// The entry byte is the low byte of r.  The rest of the texel (the upper
// byte of r, and g) is the depth range and is not read here.
fn lut_entry(object_id: u32) -> OutlineEntry {
    let x = i32(object_id & 1023u);
    let y = i32((object_id >> 10u) & 1023u);
    let value = textureLoad(lutTex, vec2<i32>(x, y), 0).r;
    return OutlineEntry(
        value & 0xFu, (value >> 4u) & 0x3u, (value >> 6u) & 0x1u, 1.0
    );
}

$$ if has_outline_id
fn label_key(texel: vec2<i32>) -> u32 {
    return textureLoad(outlineIdTex, clamp_texel(texel), 0).r;
}
$$ endif

// The key the edge test compares.  Whole-object visuals key on the pygfx
// object id; label visuals key on the per-pixel label field, which is what
// gives boundaries *between* labels inside one volume.
fn outline_key(texel: vec2<i32>) -> u32 {
$$ if has_outline_id
    if (is_label_kind(lut_entry(pick_id(texel)).kind)) {
        return LABEL_KEY_TAG | label_key(texel);
    }
$$ endif
    return pick_id(texel);
}

// The style for a pixel.  Identical to a plain LUT fetch for whole-object
// visuals; for KIND_LABEL the *slot* comes from the key's own range
// instead, because selection is per label, not per visual.  KIND_LABEL_ALL
// keeps the visual's own slot, exactly as a whole-object visual does, which
// is what makes every label in it one colour.
fn resolved_entry(texel: vec2<i32>) -> OutlineEntry {
    var entry = lut_entry(pick_id(texel));
$$ if has_outline_id
    if (entry.kind == KIND_LABEL) {
        let key = label_key(texel);
        // 1..15 is the selection range; anything above is a hashed
        // unselected label, anything at 0 is background.
        if (key >= 1u && key <= MAX_LABEL_SLOT) {
            entry.slot = key;
        } else {
            entry.slot = 0u;
        }
    } else if (entry.kind == KIND_LABEL_ALL) {
        // Only background is forced off.  A labels shader discards there,
        // so this is belt and braces rather than a path normally reached.
        if (label_key(texel) == 0u) {
            entry.slot = 0u;
        }
    }
$$ endif
    return entry;
}

$$ if has_depth_edges
// ---------------------------------------------------------------------
// Depth edges
//
// For two pixels of the same key, `near` and `far`, a thickness apart:
//
//     gap       z_far - z_near > start * z_near
//     carried   the far surface carried on in a straight line to where
//               the near pixel is: from `beyond`, one thickness past far,
//               or, where that is off the surface, from the pixel one
//               step past far.  Off the surface: outside the frame, or
//               another key.  With neither there is no edge.
//     in_front  z(carried) - z_near > start * z_near
//
// The pair is an edge when gap and in_front both hold.  Raw depth is
// linear across a plane on screen, so the far surface is carried on in raw
// depth.  The second condition is what keeps the rim of a wide surface,
// where the depth also changes fast, from being drawn as an edge.
//
// z is the distance from the camera along the view direction, and `start`
// and `solid` are the visual's depth range, fractions of that distance.
// The edge's strength rises from 0 at a jump of `start` to 1 at `solid`,
// on the smaller of the two jumps tested.
// ---------------------------------------------------------------------

fn in_frame(texel: vec2<i32>) -> bool {
    let dims = vec2<i32>(textureDimensions(pickTex));
    return texel.x >= 0 && texel.y >= 0 && texel.x < dims.x && texel.y < dims.y;
}

fn raw_depth(texel: vec2<i32>) -> f32 {
    return textureLoad(depthTex, clamp_texel(texel), 0);
}

// Distance from the camera plane of a raw depth.  Nothing drawn (the far
// plane) and anything behind the camera read as very far away.
fn view_z(raw: f32) -> f32 {
    let inv = u_outline.projection_transform_inv;
    let z = inv[2][2] * raw + inv[3][2];
    let w = inv[2][3] * raw + inv[3][3];
    let distance = -z / w;
    if (raw >= 1.0 || distance <= 0.0) {
        return 1.0e30;
    }
    return distance;
}

// (start, solid) of the visual drawn at a texel.  A start of 0 means the
// visual has no depth test.  g holds the start in 16 bits; the upper byte
// of r holds the ramp, 32 codes to a doubling of solid over start, 0 being
// a hard edge.
fn depth_range(texel: vec2<i32>) -> vec2<f32> {
    let id = pick_id(texel);
    let value = textureLoad(
        lutTex, vec2<i32>(i32(id & 1023u), i32((id >> 10u) & 1023u)), 0
    );
    let start = f32(value.g) / 65535.0;
    let ramp = (value.r >> 8u) & 0xFFu;
    return vec2<f32>(start, start * exp2(f32(ramp) / 32.0));
}

// The strength of a depth edge whose smaller jump is `jump`.
fn edge_strength(jump: f32, range: vec2<f32>) -> f32 {
    if (range.y > range.x) {
        return smoothstep(range.x, range.y, jump);
    }
    return 1.0;
}

// The strength of the depth edge between a near and a far pixel of one
// key, or -1 when the pair is not an edge.
//
//     far_texel  the far pixel
//     away       from the far pixel away from the near one, one thickness
//     t          the thickness, in pixels
//     z_near     the near pixel's distance
//     range      the range of the visual the edge is drawn for
fn depth_edge(
    far_texel: vec2<i32>,
    away: vec2<i32>,
    t: i32,
    key: u32,
    z_near: f32,
    range: vec2<f32>,
) -> f32 {
    let limit = range.x * z_near;
    let raw_far = raw_depth(far_texel);
    let gap = view_z(raw_far) - z_near;
    if (gap <= limit) {
        return -1.0;
    }
    var raw_carried: f32;
    let beyond = far_texel + away;
    if (in_frame(beyond) && outline_key(beyond) == key) {
        raw_carried = 2.0 * raw_far - raw_depth(beyond);
    } else {
        let one_past = far_texel + away / t;
        if (!in_frame(one_past) || outline_key(one_past) != key) {
            return -1.0;
        }
        raw_carried = raw_far + f32(t) * (raw_far - raw_depth(one_past));
    }
    let front = view_z(raw_carried) - z_near;
    if (front <= limit) {
        return -1.0;
    }
    return edge_strength(min(gap, front) / z_near, range);
}
$$ endif

// Resolve which outline style, if any, applies to this pixel.
//
// Inward placement takes its style from the *centre* pixel: the pixel is
// inside an outlined region and a tap t_in away has left it.  Outward
// placement takes its style from whichever *neighbour* is outlined, in
// the fixed scan order +x, -x, +y, -y, first hit wins.  Inward wins over
// outward: a pixel inside a region belongs to that region rather than to
// a halo cast onto it by a neighbour.
//
// The tests run on the raw key, never on the resolved slot, so two
// adjacent regions that both resolve to "not selected" stay
// distinguishable.
//
// A depth edge is directed like an identity edge, "the tap is much farther
// away" standing for "the tap has left the region".  Inward, the centre is
// the near pixel and the band lies on the nearer surface, with the
// centre's style and range.  Outward, the tap is the near pixel and the
// band lies on the farther surface, with the tap's.  An identity edge is
// returned at once; of the depth edges of the four taps the strongest is
// kept, because the first to fire need not be.
fn outline_hit(texel: vec2<i32>, t_in: i32, t_out: i32) -> OutlineEntry {
    let c_key = outline_key(texel);

$$ if has_inward
    let c_entry = resolved_entry(texel);
    if (c_entry.kind != KIND_NONE
        && c_entry.placement == PLACEMENT_INWARD
        && t_in > 0) {
        if (outline_key(texel + vec2<i32>(t_in, 0)) != c_key
            || outline_key(texel + vec2<i32>(-t_in, 0)) != c_key
            || outline_key(texel + vec2<i32>(0, t_in)) != c_key
            || outline_key(texel + vec2<i32>(0, -t_in)) != c_key) {
            return c_entry;
        }
$$ if has_depth_edges
        let c_range = depth_range(texel);
        if (c_range.x > 0.0) {
            let z_c = view_z(raw_depth(texel));
            var c_best = -1.0;
            var c_offsets = array<vec2<i32>, 4>(
                vec2<i32>(t_in, 0),
                vec2<i32>(-t_in, 0),
                vec2<i32>(0, t_in),
                vec2<i32>(0, -t_in)
            );
            for (var i: i32 = 0; i < 4; i = i + 1) {
                c_best = max(
                    c_best,
                    depth_edge(
                        texel + c_offsets[i], c_offsets[i], t_in, c_key, z_c, c_range
                    )
                );
            }
            if (c_best >= 0.0) {
                var c_hit = c_entry;
                c_hit.strength = c_best;
                return c_hit;
            }
        }
$$ endif
    }
$$ endif

$$ if has_outward
    if (t_out > 0) {
        var offsets = array<vec2<i32>, 4>(
            vec2<i32>(t_out, 0),
            vec2<i32>(-t_out, 0),
            vec2<i32>(0, t_out),
            vec2<i32>(0, -t_out)
        );
$$ if has_depth_edges
        var n_best = -1.0;
        var n_hit = none_entry();
$$ endif
        for (var i: i32 = 0; i < 4; i = i + 1) {
            let n_texel = texel + offsets[i];
            if (outline_key(n_texel) != c_key) {
                let n_entry = resolved_entry(n_texel);
                if (n_entry.kind != KIND_NONE
                    && n_entry.placement == PLACEMENT_OUTWARD) {
                    return n_entry;
                }
            }
$$ if has_depth_edges
            else if (c_key != 0u) {
                // The same key: the tap may be a nearer part of the region
                // the centre is in.
                let n_entry = resolved_entry(n_texel);
                let n_range = depth_range(n_texel);
                if (n_entry.kind != KIND_NONE
                    && n_entry.placement == PLACEMENT_OUTWARD
                    && n_range.x > 0.0) {
                    let strength = depth_edge(
                        texel,
                        -offsets[i],
                        t_out,
                        c_key,
                        view_z(raw_depth(n_texel)),
                        n_range
                    );
                    if (strength > n_best) {
                        n_best = strength;
                        n_hit = n_entry;
                    }
                }
            }
$$ endif
        }
$$ if has_depth_edges
        if (n_best >= 0.0) {
            n_hit.strength = n_best;
            return n_hit;
        }
$$ endif
    }
$$ endif

    return none_entry();
}

// Standard source-over.  An alpha of 1 replaces the fill outright, which
// is what the selection layer wants; the boundaries layer's translucent
// default blends instead.
fn alpha_over(src: vec4<f32>, dst: vec4<f32>) -> vec4<f32> {
    let a = src.a + dst.a * (1.0 - src.a);
    return vec4<f32>(mix(dst.rgb, src.rgb, src.a), a);
}

// A layer's colour at the strength of the hit it is drawn for.
fn at_strength(colour: vec4<f32>, hit: OutlineEntry) -> vec4<f32> {
$$ if has_depth_edges
    return vec4<f32>(colour.rgb, colour.a * hit.strength);
$$ else
    return colour;
$$ endif
}

@fragment
fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(varyings.position.xy);
    var out_color = textureLoad(colorTex, texel, 0);

    // Boundaries layer: every outlined region, thin, blended over the fill.
    if (u_outline.boundaries_enabled != 0) {
        let hit = outline_hit(texel, {{ b_t_in }}, {{ b_t_out }});
        if (hit.kind != KIND_NONE) {
            out_color = alpha_over(
                at_strength(u_outline.boundary_color, hit), out_color
            );
        }
    }

    // Selection layer: regions with a nonzero slot.  Wins over boundaries.
    if (u_outline.selection_enabled != 0) {
        let hit = outline_hit(texel, {{ s_t_in }}, {{ s_t_out }});
        if (hit.slot != 0u) {
            out_color = alpha_over(
                at_strength(u_outline.palette[hit.slot - 1u], hit), out_color
            );
        }
$$ if inner_t > 0
        else {
            // Contrast band, immediately inside the selection outline, so
            // a coloured outline stays legible over an arbitrary fill.
            let inner = outline_hit(
                texel, {{ s_t_in }} + {{ inner_t }}, {{ s_t_out }} + {{ inner_t }}
            );
            if (inner.slot != 0u) {
                out_color = alpha_over(
                    at_strength(u_outline.inner_color, inner), out_color
                );
            }
        }
$$ endif
    }

    return out_color;
}
