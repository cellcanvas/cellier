{$ include 'pygfx.std.wgsl' $}
{$ include 'pygfx.image_common.wgsl' $}

// ── Custom bindings (bound by LabelImageShader.get_bindings) ─────────────
// t_img          : texture_2d<i32>  — label slice (r32sint)
// t_label_keys   : texture_2d<i32>  — sorted label IDs (direct mode only)
// t_label_colors : texture_2d<f32>  — RGBA per sorted entry (direct mode only)
// u_label_params : LabelParams uniform (background_label, salt, n_entries, _pad)
//
// NOTE: Do NOT define struct LabelParams here.
// pygfx auto-generates it from the numpy dtype via structname=.

// ── HSV → RGB ─────────────────────────────────────────────────────────────
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

// ── Hash-based random colormap ────────────────────────────────────────────
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
// ── Binary-search direct colormap ────────────────────────────────────────
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

// ── Vertex stage (standard pygfx image geometry) ─────────────────────────
struct VertexInput { @builtin(vertex_index) vertex_index: u32 };

@vertex
fn vs_main(in: VertexInput) -> Varyings {
    var geo = get_im_geometry();
    let i0  = geo.indices[i32(in.vertex_index)];
    let data_pos  = vec4<f32>(geo.positions[i0], 1.0);
    let world_pos = u_wobject.world_transform * data_pos;
    let ndc_pos   = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;
    var varyings: Varyings;
    varyings.position  = vec4<f32>(ndc_pos);
    varyings.world_pos = vec3<f32>(world_pos.xyz);
    varyings.texcoord  = vec2<f32>(geo.texcoords[i0]);
    return varyings;
}

// ── Fragment stage ────────────────────────────────────────────────────────
@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    let grid_size = vec2<i32>(textureDimensions(t_img));
    let texel     = vec2<i32>(floor(varyings.texcoord * vec2<f32>(grid_size)));
    let clamped   = clamp(texel, vec2<i32>(0), grid_size - vec2<i32>(1));
    let label_id  = textureLoad(t_img, clamped, 0).r;

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
