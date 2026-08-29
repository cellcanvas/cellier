{$ include 'pygfx.std.wgsl' $}
{$ include 'pygfx.image_common.wgsl' $}

// -- custom bindings injected by define_bindings --
// t_img        -- proxy texture (grid-dim), drives get_im_geometry()
// t_cache      -- texture_2d<i32>, integer label tile cache
// t_lut        -- texture_2d<f32>, float32 LUT (slot indices + level)
// u_lut_params -- LutParams uniform (auto-generated struct)
// u_block_scales -- BlockScales uniform (auto-generated struct)
// u_label_params -- LabelParams uniform (background_label, salt, n_entries)
// t_label_keys   -- texture_2d<i32>  (direct mode: sorted label IDs)
// t_label_colors -- texture_2d<f32>  (direct mode: RGBA per entry)
// t_paint_lut    -- texture_2d<f32>, .r=slot_index .g=presence (0 = no slot)
// t_paint_cache  -- texture_2d<f32>, .r=label_id_as_float32 .g=presence

// NOTE: Do NOT define struct LabelParams, LutParams, or BlockScales here.
// pygfx auto-generates them from numpy dtypes via structname=.

fn get_tile_scale(level: i32) -> vec2<f32> {
    switch level {
        case 1: { return vec2<f32>(u_block_scales.scale_1[0], u_block_scales.scale_1[1]); }
        case 2: { return vec2<f32>(u_block_scales.scale_2[0], u_block_scales.scale_2[1]); }
        case 3: { return vec2<f32>(u_block_scales.scale_3[0], u_block_scales.scale_3[1]); }
        case 4: { return vec2<f32>(u_block_scales.scale_4[0], u_block_scales.scale_4[1]); }
        case 5: { return vec2<f32>(u_block_scales.scale_5[0], u_block_scales.scale_5[1]); }
        case 6: { return vec2<f32>(u_block_scales.scale_6[0], u_block_scales.scale_6[1]); }
        case 7: { return vec2<f32>(u_block_scales.scale_7[0], u_block_scales.scale_7[1]); }
        case 8: { return vec2<f32>(u_block_scales.scale_8[0], u_block_scales.scale_8[1]); }
        case 9: { return vec2<f32>(u_block_scales.scale_9[0], u_block_scales.scale_9[1]); }
        default: { return vec2<f32>(0.0, 0.0); }
    }
}

// ── Label LUT indirection sample (returns integer label ID) ───────────────
fn sample_im_lut(texcoord: vec2<f32>) -> i32 {
    let block_size = vec2<f32>(u_lut_params.block_size_x, u_lut_params.block_size_y);
    let cache_size = vec2<i32>(i32(u_lut_params.cache_size_x), i32(u_lut_params.cache_size_y));
    let lut_size   = vec2<i32>(i32(u_lut_params.lut_size_x), i32(u_lut_params.lut_size_y));
    let vol_size   = vec2<f32>(u_lut_params.vol_size_x, u_lut_params.vol_size_y);
    let overlap    = u_lut_params.overlap;
    let padded_size = block_size + vec2<f32>(2.0 * overlap);

    let pos = clamp(texcoord * vol_size, vec2<f32>(0.0), vol_size - vec2<f32>(0.5));

    let tile_f   = floor(pos / block_size);
    let tile_idx = clamp(vec2<i32>(tile_f), vec2<i32>(0), lut_size - vec2<i32>(1));

    let lutv  = textureLoad(t_lut, tile_idx, 0);
    let level = i32(lutv.z);
    if (level == 0) {
        return u_label_params.background_label;
    }

    let tile_origin = vec2<f32>(lutv.x, lutv.y) * padded_size;

    let sj = get_tile_scale(level);
    let scaled_pos  = pos * sj;
    let within_tile = scaled_pos - floor(scaled_pos / block_size) * block_size;

    // Nearest-neighbor: round to texel, no +0.5 offset.
    let cache_pos_f = tile_origin + within_tile + vec2<f32>(overlap);
    let cache_pos   = clamp(vec2<i32>(round(cache_pos_f)), vec2<i32>(0), cache_size - vec2<i32>(1));
    return textureLoad(t_cache, cache_pos, 0).r;
}

// ── HSV → RGB ──────────────────────────────────────────────────────────────
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let hp = h * 6.0;
    let i  = floor(hp);
    let f  = hp - i;
    let p  = v * (1.0 - s);
    let q  = v * (1.0 - s * f);
    let t  = v * (1.0 - s * (1.0 - f));
    switch (i32(i) % 6) {
        case 0:  { return vec3<f32>(v, t, p); }
        case 1:  { return vec3<f32>(q, v, p); }
        case 2:  { return vec3<f32>(p, v, t); }
        case 3:  { return vec3<f32>(p, q, v); }
        case 4:  { return vec3<f32>(t, p, v); }
        default: { return vec3<f32>(v, p, q); }
    }
}

{$ include 'cellier.label_outline_key.wgsl' $}

fn random_label_color(label_id: i32, salt: u32) -> vec4<f32> {
    // Shared with the outline key, so a key collision is also a hue
    // collision; see cellier.label_outline_key.wgsl.
    let x = random_label_hash(label_id, salt);
    let hue  = f32(x & 0xFFFFu) / 65535.0;
    let sat  = 0.9 + 0.1 * f32((x >> 16u) & 0x3Fu) / 63.0;
    let val_ = 0.7 + 0.3 * f32((x >> 22u) & 0xFu)  / 15.0;
    return vec4<f32>(hsv_to_rgb(hue, sat, val_), 1.0);
}

$$ if colormap_mode == "direct"
fn direct_label_color(label_id: i32, n: i32) -> vec4<f32> {
    var lo = 0; var hi = n;
    for (var iter = 0; iter < 17; iter++) {
        if (lo >= hi) { break; }
        let mid = (lo + hi) / 2;
        let key = textureLoad(t_label_keys, vec2<i32>(mid, 0), 0).r;
        if (key == label_id) {
            return textureLoad(t_label_colors, vec2<i32>(mid, 0), 0);
        }
        if (key < label_id) { lo = mid + 1; } else { hi = mid; }
    }
    return vec4<f32>(0.0);
}
$$ endif

fn get_label_color(label_id: i32) -> vec4<f32> {
    $$ if colormap_mode == "random"
    return random_label_color(label_id, u_label_params.salt);
    $$ else
    return direct_label_color(label_id, i32(u_label_params.n_entries));
    $$ endif
}

// ── Paint overlay (level-0 only, label IDs stored as float32) ─────────────
fn sample_paint_label_lut(texcoord: vec2<f32>) -> vec2<f32> {
    // Returns (label_id_as_float32, presence).  presence < 0.5 means no paint.
    let block_size = vec2<f32>(u_lut_params.block_size_x, u_lut_params.block_size_y);
    let vol_size   = vec2<f32>(u_lut_params.vol_size_x, u_lut_params.vol_size_y);
    let lut_size   = vec2<i32>(i32(u_lut_params.lut_size_x), i32(u_lut_params.lut_size_y));

    let pos      = clamp(texcoord * vol_size, vec2<f32>(0.0), vol_size - vec2<f32>(0.5));
    let tile_f   = floor(pos / block_size);
    let tile_idx = clamp(vec2<i32>(tile_f), vec2<i32>(0), lut_size - vec2<i32>(1));

    let lutv = textureLoad(t_paint_lut, tile_idx, 0);
    if (lutv.y < 0.5) {
        return vec2<f32>(0.0, 0.0);
    }

    let slot   = i32(lutv.x);
    let bs     = i32(u_lut_params.block_size_x);
    let within = vec2<i32>(pos - tile_f * block_size);
    let coord  = vec2<i32>(within.x, slot * bs + within.y);
    let v      = textureLoad(t_paint_cache, coord, 0);
    return vec2<f32>(v.r, v.g);  // (label_id_as_float32, presence)
}

// ── Vertex shader ─────────────────────────────────────────────────────────

struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};

@vertex
fn vs_main(in: VertexInput) -> Varyings {
    var geo = get_im_geometry();
    let index = i32(in.vertex_index);
    let i0 = geo.indices[index];

    let data_pos  = vec4<f32>(geo.positions[i0], 1.0);
    let world_pos = u_wobject.world_transform * data_pos;
    let ndc_pos   = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

    var varyings: Varyings;
    varyings.position  = vec4<f32>(ndc_pos);
    varyings.world_pos = vec3<f32>(world_pos.xyz);
    varyings.texcoord  = vec2<f32>(geo.texcoords[i0]);
    return varyings;
}

// ── Fragment shader ───────────────────────────────────────────────────────

@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    var label_id: i32 = sample_im_lut(varyings.texcoord);

    // Paint overlay: substitute the painted label ID if this voxel has been
    // painted.  Runs before the background discard so painting with label 0
    // erases existing labels, and painting on a blank tile shows immediately.
    let paint = sample_paint_label_lut(varyings.texcoord);
    if (paint.y > 0.5) {
        label_id = i32(paint.x + 0.5);
    }

    if (label_id == u_label_params.background_label) { discard; }

    let color = get_label_color(label_id);
    if (color.a < 0.001) { discard; }

    var out: FragmentOutput;
    out.color = vec4<f32>(color.rgb, color.a * u_material.opacity);

    $$ if write_outline_id
    // Label key for the screen-space outline pass.  Zero-initialised
    // to 0 (= not outlined) on canvases without the outline_id target.
    out.outline_id = label_outline_key(label_id);
    $$ endif
    $$ if write_pick
    out.pick = (
        pick_pack(u32(u_wobject.global_id), 20) +
        pick_pack(u32(varyings.texcoord.x * 4194303.0), 22) +
        pick_pack(u32(varyings.texcoord.y * 4194303.0), 22)
    );
    $$ endif

    return out;
}
