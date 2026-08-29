// Screen-space outline composite.
//
// Runs as the fragment stage of a full-quad effect pass.  It reads the
// renderer's pick buffer, derives a per-pixel outline key, tests four
// neighbouring taps at the configured thickness, and composites a
// coloured outline over the rendered image.  One pass, no intermediate
// textures.
//
// Bindings are declared by the Python side (_outline.py); this file
// assumes:
//     0 u_outline   uniform
//     1 texSampler  sampler          (unused; the pass loads, never samples)
//     2 colorTex    texture_2d<f32>
//     3 pickTex     texture_2d<u32>
//     4 lutTex      texture_2d<u32>  r8uint, 1024x1024
//     5 outlineIdTex texture_2d<u32>  r32uint, label keys (has_outline_id)
//
// Template vars: b_t_in, b_t_out, s_t_in, s_t_out, inner_t,
//                has_inward, has_outward, has_outline_id.

const KIND_NONE: u32 = 0u;
const KIND_LABEL: u32 = 2u;
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

struct OutlineEntry {
    slot: u32,
    kind: u32,
    placement: u32,
};

fn none_entry() -> OutlineEntry {
    return OutlineEntry(0u, KIND_NONE, PLACEMENT_INWARD);
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

fn lut_entry(object_id: u32) -> OutlineEntry {
    let x = i32(object_id & 1023u);
    let y = i32((object_id >> 10u) & 1023u);
    let value = textureLoad(lutTex, vec2<i32>(x, y), 0).r;
    return OutlineEntry(value & 0xFu, (value >> 4u) & 0x3u, (value >> 6u) & 0x1u);
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
    if (lut_entry(pick_id(texel)).kind == KIND_LABEL) {
        return LABEL_KEY_TAG | label_key(texel);
    }
$$ endif
    return pick_id(texel);
}

// The style for a pixel.  Identical to a plain LUT fetch for whole-object
// visuals; for labels the *slot* comes from the key's own range instead,
// because selection is per label, not per visual.
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
    }
$$ endif
    return entry;
}

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
        for (var i: i32 = 0; i < 4; i = i + 1) {
            let n_texel = texel + offsets[i];
            if (outline_key(n_texel) != c_key) {
                let n_entry = resolved_entry(n_texel);
                if (n_entry.kind != KIND_NONE
                    && n_entry.placement == PLACEMENT_OUTWARD) {
                    return n_entry;
                }
            }
        }
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

@fragment
fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(varyings.position.xy);
    var out_color = textureLoad(colorTex, texel, 0);

    // Boundaries layer: every outlined region, thin, blended over the fill.
    if (u_outline.boundaries_enabled != 0) {
        let hit = outline_hit(texel, {{ b_t_in }}, {{ b_t_out }});
        if (hit.kind != KIND_NONE) {
            out_color = alpha_over(u_outline.boundary_color, out_color);
        }
    }

    // Selection layer: regions with a nonzero slot.  Wins over boundaries.
    if (u_outline.selection_enabled != 0) {
        let hit = outline_hit(texel, {{ s_t_in }}, {{ s_t_out }});
        if (hit.slot != 0u) {
            out_color = alpha_over(u_outline.palette[hit.slot - 1u], out_color);
        }
$$ if inner_t > 0
        else {
            // Contrast band, immediately inside the selection outline, so
            // a coloured outline stays legible over an arbitrary fill.
            let inner = outline_hit(
                texel, {{ s_t_in }} + {{ inner_t }}, {{ s_t_out }} + {{ inner_t }}
            );
            if (inner.slot != 0u) {
                out_color = alpha_over(u_outline.inner_color, out_color);
            }
        }
$$ endif
    }

    return out_color;
}
