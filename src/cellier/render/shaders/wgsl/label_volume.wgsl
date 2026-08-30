{$ include 'pygfx.std.wgsl' $}

// ── Custom bindings ───────────────────────────────────────────────────────
// t_img          : texture_3d<i32>   — full label volume (r32sint)
// t_label_keys   : texture_2d<i32>   — direct-mode sorted keys
// t_label_colors : texture_2d<f32>   — direct-mode RGBA
// u_label_params : LabelParams uniform (background_label, salt, n_entries, _pad)
//
// NOTE: Do NOT define struct LabelParams here.
// pygfx auto-generates it from the numpy dtype via structname=.
//
// Local (object) space convention: voxel i is centered at i.
// The volume of shape (W, H, D) spans [-0.5, W-0.5] x [-0.5, H-0.5] x [-0.5, D-0.5].
// world_transform maps local space to world space.

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
{$ include 'cellier.view_normal.wgsl' $}

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

// ── Direct integer label fetch (no rounding; v is already a voxel cell) ───
fn load_label(v: vec3<i32>) -> i32 {
    return textureLoad(t_img, v, 0).r;
}

// ── Unit vector along an axis (0=x, 1=y, 2=z) ─────────────────────────────
fn unit_axis(a: i32) -> vec3<f32> {
    return vec3<f32>(
        select(0.0, 1.0, a == 0),
        select(0.0, 1.0, a == 1),
        select(0.0, 1.0, a == 2),
    );
}

// ── Ray-AABB intersection (slab test) ────────────────────────────────────
// Returns (t_near, t_far, entry_axis): entry_axis is the axis whose near slab
// won t_near, i.e. the box face the ray enters through. Ties break x > y > z.
struct AabbHit {
    t_near:     f32,
    t_far:      f32,
    entry_axis: i32,
};

fn intersect_aabb(
    ray_o: vec3<f32>, inv_d: vec3<f32>,
    box_min: vec3<f32>, box_max: vec3<f32>,
) -> AabbHit {
    let t0 = (box_min - ray_o) * inv_d;
    let t1 = (box_max - ray_o) * inv_d;
    let tnear_axis = min(t0, t1);
    let tfar_axis  = max(t0, t1);
    let t_near = max(max(tnear_axis.x, tnear_axis.y), tnear_axis.z);
    let t_far  = min(min(tfar_axis.x, tfar_axis.y), tfar_axis.z);
    var entry_axis = 0;
    var entry_best = tnear_axis.x;
    if (tnear_axis.y > entry_best) { entry_axis = 1; entry_best = tnear_axis.y; }
    if (tnear_axis.z > entry_best) { entry_axis = 2; entry_best = tnear_axis.z; }
    var hit: AabbHit;
    hit.t_near     = t_near;
    hit.t_far      = t_far;
    hit.entry_axis = entry_axis;
    return hit;
}

// ── Vertex shader ─────────────────────────────────────────────────────────
struct VertexInput { @builtin(vertex_index) vertex_index: u32 };

@vertex
fn vs_main(in: VertexInput) -> Varyings {
    // Build bounding-box corners from texture dimensions.
    // pygfx voxel convention: voxel i is centered at i; box spans [-0.5, N-0.5].
    let dims = vec3<f32>(textureDimensions(t_img));
    let lo   = vec3<f32>(-0.5);
    let hi   = dims - vec3<f32>(0.5);

    let corners = array<vec3<f32>, 8>(
        vec3<f32>(lo.x, lo.y, lo.z),
        vec3<f32>(hi.x, lo.y, lo.z),
        vec3<f32>(lo.x, hi.y, lo.z),
        vec3<f32>(hi.x, hi.y, lo.z),
        vec3<f32>(lo.x, lo.y, hi.z),
        vec3<f32>(hi.x, lo.y, hi.z),
        vec3<f32>(lo.x, hi.y, hi.z),
        vec3<f32>(hi.x, hi.y, hi.z),
    );
    // 12 triangles (36 indices), cull_mode = none.
    let face_idx = array<u32, 36>(
        0u,2u,1u, 1u,2u,3u,
        4u,5u,6u, 5u,7u,6u,
        0u,1u,4u, 1u,5u,4u,
        2u,6u,3u, 3u,6u,7u,
        0u,4u,2u, 2u,4u,6u,
        1u,3u,5u, 3u,7u,5u,
    );

    let local_pos = corners[face_idx[in.vertex_index]];
    let world_pos = u_wobject.world_transform * vec4<f32>(local_pos, 1.0);
    let ndc_pos   = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

    // Pass near/far clip-plane positions in local space for ray reconstruction.
    let ndc_to_local  = u_wobject.world_transform_inv * u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv;
    let cam_sign = sign(
        u_stdinfo.cam_transform[0][0] *
        u_stdinfo.cam_transform[1][1] *
        u_stdinfo.cam_transform[2][2]
    );

    var varyings: Varyings;
    varyings.position      = vec4<f32>(ndc_pos);
    varyings.world_pos     = vec3<f32>(world_pos.xyz);
    varyings.local_near    = vec4<f32>(ndc_to_local * vec4<f32>(ndc_pos.xy, -1.0 * cam_sign * ndc_pos.w, ndc_pos.w));
    varyings.local_far     = vec4<f32>(ndc_to_local * vec4<f32>(ndc_pos.xy,        cam_sign * ndc_pos.w, ndc_pos.w));
    return varyings;
}

// ── Fragment shader ───────────────────────────────────────────────────────
@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    let vol_dims  = vec3<i32>(textureDimensions(t_img));
    let vol_dimsf = vec3<f32>(vol_dims);

    // Reconstruct ray in local (voxel) space.
    let near_pos = varyings.local_near.xyz / varyings.local_near.w;
    let far_pos  = varyings.local_far.xyz  / varyings.local_far.w;
    let ray_dir  = normalize(far_pos - near_pos);

    // Ray-AABB intersection with local box [-0.5, dims-0.5].
    let lo = vec3<f32>(-0.5);
    let hi = vol_dimsf - vec3<f32>(0.5);
    let safe_dir    = select(ray_dir, vec3<f32>(1e-20), abs(ray_dir) < vec3<f32>(1e-20));
    let inv_dir     = 1.0 / safe_dir;
    let t_hit       = intersect_aabb(near_pos, inv_dir, lo, hi);
    if (t_hit.t_near > t_hit.t_far) { discard; }
    let t_start = max(t_hit.t_near, 0.0);
    let t_end   = t_hit.t_far;
    // True when the ray entered through a box face (vs. the near clip plane
    // sitting inside the volume, which clamps t_near < 0 to t_start = 0).
    let entered_box_face = t_hit.t_near >= 0.0;

    // ── Grid-anchored DDA (Amanatides-Woo) ───────────────────────────────
    // Walk the voxel grid cell-by-cell rather than sampling at a fixed cadence
    // along the ray. Voxel i owns [i-0.5, i+0.5); cell faces are at half
    // integers. Indexing the starting cell by round() and anchoring the next
    // face planes to those half integers makes the traversal a function of the
    // data grid alone, so the detected surface no longer slides with the
    // camera (no zoom/rotate edge wobble).
    let eps  = 1e-3;
    let P_in = near_pos + ray_dir * t_start;
    // Nudge inward so an entry exactly on a half-integer face snaps to the
    // first interior cell unambiguously (WGSL round() is ties-to-even).
    var v = clamp(
        vec3<i32>(round(P_in + safe_dir * eps)),
        vec3<i32>(0),
        vol_dims - vec3<i32>(1),
    );
    let step    = vec3<i32>(sign(safe_dir));
    // t to reach each axis's next cell face, and t to cross one full voxel.
    var t_max   = (vec3<f32>(v) + 0.5 * vec3<f32>(step) - near_pos) * inv_dir;
    let t_delta = abs(inv_dir);

    var surface_found = false;
    var surface_lid   = u_label_params.background_label;
    var last_axis     = -1;   // face crossed to enter the foreground cell
    var t_cross       = t_start;

    // Bounded by the longest possible grid path so a degenerate ray can't hang.
    let max_steps = vol_dims.x + vol_dims.y + vol_dims.z + 1;
    for (var i = 0; i < max_steps; i++) {
        let lid = load_label(v);
        if (lid != u_label_params.background_label) {
            surface_found = true;
            surface_lid   = lid;
            break;
        }
        // Advance across the nearest cell face.
        var a     = 0;
        var t_min = t_max.x;
        if (t_max.y < t_min) { a = 1; t_min = t_max.y; }
        if (t_max.z < t_min) { a = 2; t_min = t_max.z; }
        t_cross = t_min;
        if (t_cross > t_end) { break; }
        if (a == 0) {
            v.x += step.x; t_max.x += t_delta.x;
            if (v.x < 0 || v.x >= vol_dims.x) { break; }
        } else if (a == 1) {
            v.y += step.y; t_max.y += t_delta.y;
            if (v.y < 0 || v.y >= vol_dims.y) { break; }
        } else {
            v.z += step.z; t_max.z += t_delta.z;
            if (v.z < 0 || v.z >= vol_dims.z) { break; }
        }
        last_axis = a;
    }

    if (!surface_found) { discard; }

    let color = get_label_color(surface_lid);
    if (color.a < 0.001) { discard; }

    // Surface position and analytic, view-invariant face normal.
    // The normal is needed for shading in iso_categorical mode, and for the
    // ambient occlusion target in every mode -- a flat_categorical surface
    // is still a real voxel face.
    $$ set needs_normal = render_mode == "iso_categorical" or write_normal
    var surface_pos = vec3<f32>(0.0);
    $$ if needs_normal
    var obj_normal  = vec3<f32>(0.0, 1.0, 0.0);
    $$ endif
    if (last_axis < 0) {
        // Entry cell already foreground: the surface is the entry point itself.
        surface_pos = P_in;
        $$ if needs_normal
        if (entered_box_face) {
            // Boundary block clipped by the volume extent: use the box face.
            obj_normal = unit_axis(t_hit.entry_axis);
        } else {
            // Near plane slices the block interior: face the camera.
            obj_normal = normalize(-ray_dir);
        }
        $$ endif
    } else {
        // Exact crossing of the cell face between background and foreground.
        surface_pos = near_pos + ray_dir * t_cross;
        $$ if needs_normal
        let face_axis = unit_axis(last_axis);
        let face_sign = dot(vec3<f32>(step), face_axis);
        obj_normal    = -face_sign * face_axis;
        $$ endif
    }

    let world_surface = u_wobject.world_transform * vec4<f32>(surface_pos, 1.0);

    $$ if render_mode == "iso_categorical"
    let view_dir  = normalize(-ray_dir);
    let normal    = select(
        normalize(obj_normal),
        vec3<f32>(0.0, 1.0, 0.0),
        dot(obj_normal, obj_normal) < 1e-10,
    );
    let N         = select(-normal, normal, dot(normal, view_dir) > 0.0);
    let ambient   = 0.3;
    let diffuse   = max(dot(N, view_dir), 0.0);
    let lit_color = color.rgb * (ambient + diffuse);
    $$ else
    let lit_color = color.rgb;
    $$ endif

    let ndc_surface = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_surface;

    var out: FragmentOutput;
    out.color = vec4<f32>(lit_color, color.a * u_material.opacity);
    out.depth = ndc_surface.z / ndc_surface.w;

    $$ if write_normal
    // Ambient occlusion normal target.  Local space here *is* voxel space
    // (voxel i is centred at local i), so the face normal needs no scale
    // correction before the object-to-view transform.
    out.normal = pack_view_normal(
        obj_normal, (u_stdinfo.cam_transform * world_surface).xyz);
    $$ endif

    $$ if write_outline_id
    // Label key for the screen-space outline pass.  Zero-initialised
    // to 0 (= not outlined) on canvases without the outline_id target.
    out.outline_id = label_outline_key(surface_lid);
    $$ endif
    $$ if write_pick
    // Encode surface position in normalised [0, 1] voxel space.
    // Local convention: voxel i centred at i, so shift by +0.5 then divide
    // by texture dimensions to get [0, 1].
    let vol_dimsf_pick = vec3<f32>(textureDimensions(t_img));
    let pick_coord = (surface_pos + vec3<f32>(0.5)) / vol_dimsf_pick;
    out.pick = (
        pick_pack(u32(u_wobject.global_id), 20) +
        pick_pack(u32(clamp(pick_coord.x, 0.0, 1.0) * 16383.0), 14) +
        pick_pack(u32(clamp(pick_coord.y, 0.0, 1.0) * 16383.0), 14) +
        pick_pack(u32(clamp(pick_coord.z, 0.0, 1.0) * 16383.0), 14)
    );
    $$ endif

    return out;
}
