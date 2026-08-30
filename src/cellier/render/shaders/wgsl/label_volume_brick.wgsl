{$ include 'pygfx.std.wgsl' $}
// NOTE: do NOT include pygfx colormap, image_sample, or volume_common.
// Labels use textureLoad (nearest-neighbor on integer textures), not textureSample.

// ── Constants ──────────────────────────────────────────────────────────────
const BORDER: f32          = 2.0;     // ghost border (level-k voxels); must match Python overlap=2
                                      // smooth_iso normal kernel (±1 voxel) also requires BORDER ≥ 2
const MAX_BRICK_ITERS: u32 = 512u;
const STEPS_PER_BRICK: f32 = 24.0;

// NOTE: Do NOT define struct LabelParams, VOL_PARAMS, or BlockScales here.
// pygfx auto-generates them from numpy dtypes via structname=.

// ── Block-scale lookup ────────────────────────────────────────────────────
fn get_lod_scale(lut_w: u32) -> vec3<f32> {
    switch lut_w {
        case 1u:  { return u_block_scales.scale_1.xyz; }
        case 2u:  { return u_block_scales.scale_2.xyz; }
        case 3u:  { return u_block_scales.scale_3.xyz; }
        case 4u:  { return u_block_scales.scale_4.xyz; }
        case 5u:  { return u_block_scales.scale_5.xyz; }
        case 6u:  { return u_block_scales.scale_6.xyz; }
        case 7u:  { return u_block_scales.scale_7.xyz; }
        case 8u:  { return u_block_scales.scale_8.xyz; }
        case 9u:  { return u_block_scales.scale_9.xyz; }
        default:  { return u_block_scales.scale_1.xyz; }
    }
}

// ── Coordinate conversions ────────────────────────────────────────────────

fn norm_to_voxel(pos: vec3<f32>, norm_size: vec3<f32>, dataset_size: vec3<f32>) -> vec3<f32> {
    return ((pos / norm_size) + 0.5) * dataset_size;
}

fn voxel_to_norm(voxel: vec3<f32>, norm_size: vec3<f32>, dataset_size: vec3<f32>) -> vec3<f32> {
    return (voxel / dataset_size - 0.5) * norm_size;
}

// ── Ray-box intersection ──────────────────────────────────────────────────

fn intersect_box(
    ray_origin: vec3<f32>, inv_ray_dir: vec3<f32>,
    box_min: vec3<f32>,   box_max: vec3<f32>
) -> vec2<f32> {
    let t0 = (box_min - ray_origin) * inv_ray_dir;
    let t1 = (box_max - ray_origin) * inv_ray_dir;
    let t_near = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let t_far  = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    return vec2<f32>(t_near, t_far);
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
{$ include 'cellier.view_normal.wgsl' $}

// ── Random colormap ────────────────────────────────────────────────────────
fn random_label_color(label_id: i32, salt: u32) -> vec4<f32> {
    // Shared with the outline key, so a key collision is also a hue
    // collision; see cellier.label_outline_key.wgsl.
    let x = random_label_hash(label_id, salt);
    let hue  = f32(x & 0xFFFFu) / 65535.0;
    let sat  = 0.9 + 0.1 * f32((x >> 16u) & 0x3Fu) / 63.0;
    let val_ = 0.7 + 0.3 * f32((x >> 22u) & 0xFu)  / 15.0;
    return vec4<f32>(hsv_to_rgb(hue, sat, val_), 1.0);
}

// ── Direct colormap (only compiled in direct mode) ────────────────────────
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

// ── Atlas sampler (integer, nearest-neighbor) ─────────────────────────────
fn sample_atlas_label(
    voxel_pos:      vec3<f32>,
    lut_entry:      vec4<u32>,
    lod_scale:      vec3<f32>,
    brick_corner_k: vec3<f32>,
) -> i32 {
    let block_size  = vec3<f32>(u_vol_params.block_size_x,
                                u_vol_params.block_size_y,
                                u_vol_params.block_size_z);
    let cache_size  = vec3<i32>(i32(u_vol_params.cache_size_x),
                                i32(u_vol_params.cache_size_y),
                                i32(u_vol_params.cache_size_z));
    let padded_size = block_size + vec3<f32>(2.0 * BORDER);

    let tile_origin  = vec3<f32>(lut_entry.xyz) * padded_size;
    let voxel_k      = voxel_pos / lod_scale;
    // Extend into the ghost border region [-BORDER, block_size-1+BORDER] so that
    // samples near brick edges read true neighbor data from the padded tile
    // rather than clamping to the brick's last voxel.
    let pos_in_brick = clamp(voxel_k - brick_corner_k, vec3<f32>(-BORDER), block_size - vec3<f32>(1.0) + vec3<f32>(BORDER));

    // Nearest-neighbor: round, no +0.5 sub-voxel shift.
    let cache_pos = tile_origin + pos_in_brick + vec3<f32>(BORDER);
    let texel     = clamp(vec3<i32>(round(cache_pos)), vec3<i32>(0), cache_size - vec3<i32>(1));
    return textureLoad(t_cache, texel, 0).r;
}

// ── Brick context lookup ──────────────────────────────────────────────────

struct BrickContext {
    lut_entry:      vec4<u32>,
    lod_scale:      vec3<f32>,
    brick_corner_k: vec3<f32>,
    valid:          bool,
};

fn lookup_brick_context(voxel_pos: vec3<f32>) -> BrickContext {
    var ctx: BrickContext;
    let block_size = vec3<f32>(u_vol_params.block_size_x,
                               u_vol_params.block_size_y,
                               u_vol_params.block_size_z);
    let lut_size_i = vec3<i32>(i32(u_vol_params.lut_size_x),
                               i32(u_vol_params.lut_size_y),
                               i32(u_vol_params.lut_size_z));
    let safe_idx   = clamp(vec3<i32>(floor(voxel_pos / block_size)),
                           vec3<i32>(0),
                           lut_size_i - vec3<i32>(1));
    let lut_entry  = textureLoad(t_lut, safe_idx, 0);
    ctx.valid      = lut_entry.w > 0u;
    if (ctx.valid) {
        ctx.lut_entry      = lut_entry;
        ctx.lod_scale      = get_lod_scale(lut_entry.w);
        ctx.brick_corner_k = floor((voxel_pos / ctx.lod_scale) / block_size) * block_size;
    }
    return ctx;
}

// ── Brick setup ───────────────────────────────────────────────────────────

struct BrickInfo {
    lut_entry:      vec4<u32>,
    lod_scale:      vec3<f32>,
    brick_corner_k: vec3<f32>,
    t_end:          f32,
    num_steps:      u32,
    step_size:      f32,
    brick_max:      f32,
    valid:          bool,
};

fn setup_brick(
    ray_origin:  vec3<f32>,
    ray_dir:     vec3<f32>,
    inv_ray_dir: vec3<f32>,
    t:           f32,
    t_end_ray:   f32,
    norm_size:   vec3<f32>,
    dataset_size: vec3<f32>,
) -> BrickInfo {
    var info: BrickInfo;

    let block_size = vec3<f32>(u_vol_params.block_size_x,
                               u_vol_params.block_size_y,
                               u_vol_params.block_size_z);
    let lut_size_i = vec3<i32>(i32(u_vol_params.lut_size_x),
                               i32(u_vol_params.lut_size_y),
                               i32(u_vol_params.lut_size_z));

    let sample_pos = ray_origin + ray_dir * t;
    let voxel_pos  = clamp(
        norm_to_voxel(sample_pos, norm_size, dataset_size),
        vec3<f32>(0.0),
        dataset_size - vec3<f32>(0.001)
    );
    let brick_f   = floor(voxel_pos / block_size);
    let safe_idx  = clamp(vec3<i32>(brick_f),
                          vec3<i32>(0),
                          lut_size_i - vec3<i32>(1));

    let brick_min_v = brick_f * block_size;
    let brick_max_v = brick_min_v + block_size;
    let brick_min_n = voxel_to_norm(brick_min_v, norm_size, dataset_size);
    let brick_max_n = voxel_to_norm(brick_max_v, norm_size, dataset_size);

    let t_brick    = intersect_box(ray_origin, inv_ray_dir, brick_min_n, brick_max_n);
    info.t_end     = min(t_brick.y, t_end_ray);

    let lut_entry  = textureLoad(t_lut, safe_idx, 0);
    info.lut_entry = lut_entry;
    info.brick_max = textureLoad(t_brick_max, safe_idx, 0).r;
    info.valid     = lut_entry.w > 0u;

    if (info.valid) {
        info.lod_scale      = get_lod_scale(lut_entry.w);
        let voxel_k_bc      = voxel_pos / info.lod_scale;
        info.brick_corner_k = floor(voxel_k_bc / block_size) * block_size;
        let brick_len       = max(info.t_end - t, 1e-6);
        let brick_world_len = length(brick_max_n - brick_min_n);
        let max_scale  = max(info.lod_scale.x, max(info.lod_scale.y, info.lod_scale.z));
        info.num_steps = max(1u, u32((STEPS_PER_BRICK / max_scale) * brick_len / brick_world_len));
        info.step_size = brick_len / f32(info.num_steps);
    }

    return info;
}

// ── Object-space label surface normal (iso_categorical + gradient_debug) ─
$$ if render_mode == "iso_categorical" or render_mode == "gradient_debug"
fn label_object_normal(
    voxel_pos:      vec3<f32>,
    lut_entry:      vec4<u32>,
    lod_scale:      vec3<f32>,
    brick_corner_k: vec3<f32>,
    surface_label:  i32,
    dataset_size:   vec3<f32>,
) -> vec3<f32> {
    // Step by one level-k voxel in each axis direction (= lod_scale in dataset voxels).
    let s = lod_scale;
    let lo = vec3<f32>(0.0);
    let hi = dataset_size - vec3<f32>(0.001);

    // Re-fetch the LUT context for each neighbor independently.  When the
    // surface sits at or near a brick boundary, floating-point precision can
    // assign surface_brick_corner_k to the wrong brick, causing the fixed
    // context to read ghost-border data instead of the true neighbor voxel.
    // Using lookup_brick_context per sample is the same strategy already used
    // by the bisection loop.  lut_entry/lod_scale/brick_corner_k remain the
    // fallback for positions that land outside the loaded LUT (valid == false).
    let xp_pos = clamp(voxel_pos + vec3<f32>(s.x, 0.0, 0.0), lo, hi);
    let xp_ctx = lookup_brick_context(xp_pos);
    let xp = sample_atlas_label(xp_pos,
        select(lut_entry,      xp_ctx.lut_entry,      xp_ctx.valid),
        select(lod_scale,      xp_ctx.lod_scale,      xp_ctx.valid),
        select(brick_corner_k, xp_ctx.brick_corner_k, xp_ctx.valid));

    let xn_pos = clamp(voxel_pos - vec3<f32>(s.x, 0.0, 0.0), lo, hi);
    let xn_ctx = lookup_brick_context(xn_pos);
    let xn = sample_atlas_label(xn_pos,
        select(lut_entry,      xn_ctx.lut_entry,      xn_ctx.valid),
        select(lod_scale,      xn_ctx.lod_scale,      xn_ctx.valid),
        select(brick_corner_k, xn_ctx.brick_corner_k, xn_ctx.valid));

    let yp_pos = clamp(voxel_pos + vec3<f32>(0.0, s.y, 0.0), lo, hi);
    let yp_ctx = lookup_brick_context(yp_pos);
    let yp = sample_atlas_label(yp_pos,
        select(lut_entry,      yp_ctx.lut_entry,      yp_ctx.valid),
        select(lod_scale,      yp_ctx.lod_scale,      yp_ctx.valid),
        select(brick_corner_k, yp_ctx.brick_corner_k, yp_ctx.valid));

    let yn_pos = clamp(voxel_pos - vec3<f32>(0.0, s.y, 0.0), lo, hi);
    let yn_ctx = lookup_brick_context(yn_pos);
    let yn = sample_atlas_label(yn_pos,
        select(lut_entry,      yn_ctx.lut_entry,      yn_ctx.valid),
        select(lod_scale,      yn_ctx.lod_scale,      yn_ctx.valid),
        select(brick_corner_k, yn_ctx.brick_corner_k, yn_ctx.valid));

    let zp_pos = clamp(voxel_pos + vec3<f32>(0.0, 0.0, s.z), lo, hi);
    let zp_ctx = lookup_brick_context(zp_pos);
    let zp = sample_atlas_label(zp_pos,
        select(lut_entry,      zp_ctx.lut_entry,      zp_ctx.valid),
        select(lod_scale,      zp_ctx.lod_scale,      zp_ctx.valid),
        select(brick_corner_k, zp_ctx.brick_corner_k, zp_ctx.valid));

    let zn_pos = clamp(voxel_pos - vec3<f32>(0.0, 0.0, s.z), lo, hi);
    let zn_ctx = lookup_brick_context(zn_pos);
    let zn = sample_atlas_label(zn_pos,
        select(lut_entry,      zn_ctx.lut_entry,      zn_ctx.valid),
        select(lod_scale,      zn_ctx.lod_scale,      zn_ctx.valid),
        select(brick_corner_k, zn_ctx.brick_corner_k, zn_ctx.valid));

    // Central difference of indicator (1 = inside label, 0 = outside).
    // Divide by lod_scale: converts from level-k voxel-space gradient to
    // world-space gradient direction (larger voxels shrink that component).
    let gx = f32(i32(xp == surface_label) - i32(xn == surface_label));
    let gy = f32(i32(yp == surface_label) - i32(yn == surface_label));
    let gz = f32(i32(zp == surface_label) - i32(zn == surface_label));
    return vec3<f32>(gx, gy, gz) / lod_scale;
}
$$ endif

// ── Smooth iso helpers (only compiled in smooth_iso mode) ─────────────────
$$ if render_mode == "smooth_iso"

// soft_density — trilinear soft indicator for a single target label.
//
// Samples the 8 corners of the level-k voxel cube centred at voxel_pos
// (each corner at ±0.5 * lod_scale in dataset space) and returns the
// fraction of corners that carry target_label.  The 0.5 isosurface of this
// field sits between the last background and first foreground voxel, giving
// a sub-voxel surface position for the bisection loop to home in on.
//
// All 8 corners share the caller's brick context: ±0.5 level-k voxels is
// well within BORDER=2, so the padded atlas tile already holds correct data
// for every corner even when voxel_pos sits at a brick edge.
//
// Parameters
// ----------
// voxel_pos      : dataset-space centre of the query voxel (level-0 voxels)
// target_label   : label ID whose isosurface is being traced
// lut_entry      : atlas tile address (xyz) and LOD level (w) for this brick
// lod_scale      : scale factor from level-k voxels to level-0 voxels, per axis
// brick_corner_k : level-k coordinate of the brick's (0,0,0) corner
// dataset_size   : full dataset extent in level-0 voxels
fn soft_density(
    voxel_pos:      vec3<f32>,
    target_label:   i32,
    lut_entry:      vec4<u32>,
    lod_scale:      vec3<f32>,
    brick_corner_k: vec3<f32>,
    dataset_size:   vec3<f32>,
) -> f32 {
    let h  = lod_scale * 0.5;
    let lo = vec3<f32>(0.0);
    let hi = dataset_size - vec3<f32>(0.001);
    var count = 0.0;
    for (var dx = -1; dx <= 1; dx += 2) {
        for (var dy = -1; dy <= 1; dy += 2) {
            for (var dz = -1; dz <= 1; dz += 2) {
                let corner = clamp(
                    voxel_pos + vec3<f32>(f32(dx), f32(dy), f32(dz)) * h,
                    lo, hi,
                );
                count += f32(sample_atlas_label(corner, lut_entry, lod_scale, brick_corner_k) == target_label);
            }
        }
    }
    return count / 8.0;
}

// smooth_label_normal — 3×3×3 Sobel gradient of the binary indicator field.
//
// Evaluates f(p) = (label(p) == target_label) at the 26 non-centre positions
// of a 3×3×3 grid spaced one level-k voxel apart, then computes weighted
// central differences using a Gaussian Sobel kernel ([1, 2, 1] weights in
// each transverse axis) to produce a smooth surface normal.
//
// Compared to the 6-sample binary gradient in label_object_normal, this
// normal varies continuously across what was previously a flat voxel face,
// giving a curved shading appearance without displacing the surface geometry.
//
// Each of the 26 neighbours independently re-fetches its brick context via
// lookup_brick_context (same pattern as label_object_normal) so that samples
// crossing a brick boundary use the correct atlas tile.  The provided
// lut_entry/lod_scale/brick_corner_k serve as fallbacks for positions whose
// brick is not yet resident in the cache.  ±1 level-k voxel is within
// BORDER=2 so the padded tile holds the correct data regardless.
//
// Parameters
// ----------
// voxel_pos      : dataset-space position of the surface voxel (level-0 voxels)
// target_label   : label ID at the surface
// lut_entry      : fallback atlas tile address for the surface brick
// lod_scale      : scale factor from level-k voxels to level-0 voxels, per axis
// brick_corner_k : fallback level-k brick origin
// dataset_size   : full dataset extent in level-0 voxels
fn smooth_label_normal(
    voxel_pos:      vec3<f32>,
    target_label:   i32,
    lut_entry:      vec4<u32>,
    lod_scale:      vec3<f32>,
    brick_corner_k: vec3<f32>,
    dataset_size:   vec3<f32>,
) -> vec3<f32> {
    let s  = lod_scale;
    let lo = vec3<f32>(0.0);
    let hi = dataset_size - vec3<f32>(0.001);

    // Evaluate the binary indicator at the 26 non-centre neighbours and store
    // in a flat array indexed by (dx+1)*9 + (dy+1)*3 + (dz+1).
    var f: array<f32, 27>;
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) { continue; }
                let pos = clamp(
                    voxel_pos + vec3<f32>(f32(dx) * s.x, f32(dy) * s.y, f32(dz) * s.z),
                    lo, hi,
                );
                let ctx = lookup_brick_context(pos);
                let lbl = sample_atlas_label(
                    pos,
                    select(lut_entry,      ctx.lut_entry,      ctx.valid),
                    select(lod_scale,      ctx.lod_scale,      ctx.valid),
                    select(brick_corner_k, ctx.brick_corner_k, ctx.valid),
                );
                f[(dx + 1) * 9 + (dy + 1) * 3 + (dz + 1)] = f32(lbl == target_label);
            }
        }
    }

    // Compute Sobel gradient: weight 2 at offset 0, weight 1 at offset ±1
    // in each transverse axis.  The two nested loops iterate over the two
    // transverse axes (da, db) for each primary axis.
    var gx = 0.0; var gy = 0.0; var gz = 0.0;
    for (var da = -1; da <= 1; da++) {
        for (var db = -1; db <= 1; db++) {
            let w = select(1.0, 2.0, da == 0) * select(1.0, 2.0, db == 0);
            gx += w * (f[( 1 + 1) * 9 + (da + 1) * 3 + (db + 1)] - f[(-1 + 1) * 9 + (da + 1) * 3 + (db + 1)]);
            gy += w * (f[(da + 1) * 9 + ( 1 + 1) * 3 + (db + 1)] - f[(da + 1) * 9 + (-1 + 1) * 3 + (db + 1)]);
            gz += w * (f[(da + 1) * 9 + (db + 1) * 3 + ( 1 + 1)] - f[(da + 1) * 9 + (db + 1) * 3 + (-1 + 1)]);
        }
    }
    // Divide by lod_scale to convert the level-k voxel-space gradient to a
    // world-space direction (larger voxels shrink that gradient component).
    return vec3<f32>(gx, gy, gz) / lod_scale;
}

$$ endif

// ── Vertex shader ─────────────────────────────────────────────────────────

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
};

@vertex
fn vs_main(in: VertexInput) -> Varyings {
    let ns = vec3<f32>(u_vol_params.norm_size_x,
                       u_vol_params.norm_size_y,
                       u_vol_params.norm_size_z);
    let h = ns * 0.5;

    let corners = array<vec3<f32>, 8>(
        vec3<f32>(-h.x, -h.y, -h.z),
        vec3<f32>( h.x, -h.y, -h.z),
        vec3<f32>(-h.x,  h.y, -h.z),
        vec3<f32>( h.x,  h.y, -h.z),
        vec3<f32>(-h.x, -h.y,  h.z),
        vec3<f32>( h.x, -h.y,  h.z),
        vec3<f32>(-h.x,  h.y,  h.z),
        vec3<f32>( h.x,  h.y,  h.z),
    );

    let indices = array<u32, 36>(
        0u,2u,1u, 1u,2u,3u,
        4u,5u,6u, 5u,7u,6u,
        0u,1u,4u, 1u,5u,4u,
        2u,6u,3u, 3u,6u,7u,
        0u,4u,2u, 2u,4u,6u,
        1u,3u,5u, 3u,7u,5u,
    );

    let norm_pos  = corners[indices[in.vertex_index]];
    let world_pos = u_wobject.world_transform * vec4<f32>(norm_pos, 1.0);
    let ndc_pos   = u_stdinfo.projection_transform
                  * u_stdinfo.cam_transform
                  * world_pos;

    let ndc_to_norm = u_wobject.world_transform_inv
                    * u_stdinfo.cam_transform_inv
                    * u_stdinfo.projection_transform_inv;

    let cam_sign = sign(
        u_stdinfo.cam_transform[0][0] *
        u_stdinfo.cam_transform[1][1] *
        u_stdinfo.cam_transform[2][2]
    );

    var varyings: Varyings;
    varyings.position      = vec4<f32>(ndc_pos);
    varyings.world_pos     = vec3<f32>(world_pos.xyz);
    varyings.norm_near_pos = vec4<f32>(ndc_to_norm * vec4<f32>(ndc_pos.xy, -1.0 * cam_sign * ndc_pos.w, ndc_pos.w));
    varyings.norm_far_pos  = vec4<f32>(ndc_to_norm * vec4<f32>(ndc_pos.xy,        cam_sign * ndc_pos.w, ndc_pos.w));
    return varyings;
}


// ── Fragment shader ───────────────────────────────────────────────────────

@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    {$ include 'pygfx.clipping_planes.wgsl' $}

    var out: FragmentOutput;

    let norm_size    = vec3<f32>(u_vol_params.norm_size_x,
                                 u_vol_params.norm_size_y,
                                 u_vol_params.norm_size_z);
    let dataset_size = vec3<f32>(u_vol_params.dataset_size_x,
                                 u_vol_params.dataset_size_y,
                                 u_vol_params.dataset_size_z);
    let background_label = u_label_params.background_label;

    // ── Ray setup ──────────────────────────────────────────────────────────
    let far_pos  = varyings.norm_far_pos.xyz  / varyings.norm_far_pos.w;
    let near_pos = varyings.norm_near_pos.xyz / varyings.norm_near_pos.w;
    let ray_dir  = normalize(far_pos - near_pos);

    let safe_dir    = select(ray_dir, vec3<f32>(1e-10), abs(ray_dir) < vec3<f32>(1e-10));
    let inv_ray_dir = 1.0 / safe_dir;

    let half_ns = norm_size * 0.5;
    let hit     = intersect_box(near_pos, inv_ray_dir, -half_ns, half_ns);
    if (hit.x > hit.y) { discard; }
    var t_start = max(hit.x, 0.0);
    let t_end   = hit.y;
    if (t_start >= t_end) { discard; }

    let ray_origin = near_pos;

    var t             = t_start;
    var surface_found = false;
    // surface_pos: midpoint after bisection (used for depth).
    var surface_pos   = vec3<f32>(0.0);
    // surface_hi_p: last confirmed-foreground bisection position (used for depth).
    var surface_hi_p  = vec3<f32>(0.0);
    // surface_lid: label ID cached from bisection — avoids a fallible atlas
    // re-sample whose round() can snap to background at voxel boundaries.
    var surface_lid:  i32;
    var surface_lut_entry: vec4<u32>;
    var surface_lod_scale: vec3<f32>;
    var surface_brick_corner_k: vec3<f32>;
    var prev_t: f32 = t_start;

    for (var brick_iter = 0u; brick_iter < MAX_BRICK_ITERS; brick_iter++) {
        if (t >= t_end || surface_found) { break; }

        let brick = setup_brick(ray_origin, ray_dir, inv_ray_dir,
                                t, t_end, norm_size, dataset_size);

        if (!brick.valid) {
            // Set prev_t to the midpoint of the skipped region, NOT to
            // brick.t_end + epsilon.  The midpoint is block_size/2 LOD-k
            // voxels from the boundary — well outside the ghost-border
            // rounding range (BORDER = 2) — so sample_atlas_label cannot
            // snap it into the first foreground voxel of the next brick.
            prev_t = (t + brick.t_end) * 0.5;
            t = brick.t_end + 0.0001;
            continue;
        }

        // Skip bricks containing no labels (brick_max == 0.0 means all-background).
        if (brick.brick_max < 0.5) {
            // Same reasoning as the invalid-brick case above.
            prev_t = (t + brick.t_end) * 0.5;
            t = brick.t_end + 0.0001;
            continue;
        }

        surface_lut_entry      = brick.lut_entry;
        surface_lod_scale      = brick.lod_scale;
        surface_brick_corner_k = brick.brick_corner_k;

        // prev_t is intentionally carried forward unchanged here.  Fix 1
        // (per-midpoint lookup_brick_context) handles LOD boundaries during
        // bisection; clamping prev_t to the brick entry is unnecessary and
        // causes bisection to lose its background anchor when the label
        // surface aligns with a brick boundary.  See: ISO brick boundary
        // fix notes.

        var t_sample = t;
        for (var i = 0u; i < brick.num_steps; i++) {
            if (t_sample > brick.t_end) { break; }

            let pos    = ray_origin + ray_dir * t_sample;
            let voxel  = clamp(
                norm_to_voxel(pos, norm_size, dataset_size),
                vec3<f32>(0.0),
                dataset_size - vec3<f32>(0.001)
            );
            let lid = sample_atlas_label(voxel, brick.lut_entry,
                                         brick.lod_scale, brick.brick_corner_k);

            if (lid != background_label) {
                // 10-step bisection for sub-voxel surface refinement.
                // norm space: lo_p = previous sample position, hi_p = current.
                //
                // Fix 1: re-fetch the LUT at each midpoint so the correct LOD
                // context is used when the midpoint crosses a LOD boundary.
                // hi_ctx tracks the LOD context of the foreground side and is
                // written back to surface_* after bisection so the final color
                // sample uses the context valid at surface_hi_p.
                var lo_p_norm = ray_origin + ray_dir * prev_t;
                var hi_p_norm = pos;
                var lid_hi:   i32 = lid;
                var hi_ctx: BrickContext;
                hi_ctx.lut_entry      = surface_lut_entry;
                hi_ctx.lod_scale      = surface_lod_scale;
                hi_ctx.brick_corner_k = surface_brick_corner_k;
                hi_ctx.valid          = true;
                for (var r = 0; r < 10; r++) {
                    let mid_norm  = (lo_p_norm + hi_p_norm) * 0.5;
                    let mid_voxel = clamp(
                        norm_to_voxel(mid_norm, norm_size, dataset_size),
                        vec3<f32>(0.0),
                        dataset_size - vec3<f32>(0.001)
                    );
                    // Re-fetch LOD context at the midpoint; fall back to hi_ctx
                    // if the brick is not yet resident in the cache.
                    let mid_ctx = lookup_brick_context(mid_voxel);
                    let use_lut_entry      = select(hi_ctx.lut_entry,      mid_ctx.lut_entry,      mid_ctx.valid);
                    let use_lod_scale      = select(hi_ctx.lod_scale,      mid_ctx.lod_scale,      mid_ctx.valid);
                    let use_brick_corner_k = select(hi_ctx.brick_corner_k, mid_ctx.brick_corner_k, mid_ctx.valid);
                    // smooth_iso: test the 0.5 contour of the soft density field
                    // for the specific label found at the coarse step (lid).
                    // Other modes: hard nearest-neighbor test for any non-background label.
                    $$ if render_mode == "smooth_iso"
                    let is_fg = soft_density(
                        mid_voxel, lid,
                        use_lut_entry, use_lod_scale, use_brick_corner_k,
                        dataset_size,
                    ) > 0.5;
                    $$ else
                    let mid_id = sample_atlas_label(
                        mid_voxel,
                        use_lut_entry,
                        use_lod_scale,
                        use_brick_corner_k,
                    );
                    let is_fg = (mid_id != background_label);
                    $$ endif
                    if (is_fg) {
                        hi_p_norm             = mid_norm;
                        $$ if render_mode == "smooth_iso"
                        // lid_hi stays as lid; the target label is fixed throughout bisection.
                        $$ else
                        lid_hi                = mid_id;
                        $$ endif
                        hi_ctx.lut_entry      = use_lut_entry;
                        hi_ctx.lod_scale      = use_lod_scale;
                        hi_ctx.brick_corner_k = use_brick_corner_k;
                    } else {
                        lo_p_norm = mid_norm;
                    }
                }
                // Write hi_ctx back: the final color sample must use the LOD
                // context that is valid at surface_hi_p, not the initial hit.
                surface_lut_entry      = hi_ctx.lut_entry;
                surface_lod_scale      = hi_ctx.lod_scale;
                surface_brick_corner_k = hi_ctx.brick_corner_k;
                surface_lid   = lid_hi;
                surface_pos   = (lo_p_norm + hi_p_norm) * 0.5;
                surface_hi_p  = hi_p_norm;
                surface_found = true;
                break;
            }
            prev_t = t_sample;
            t_sample += brick.step_size;
        }
        if (!surface_found) {
            t = brick.t_end + 0.0001;
        }
    }

    // Compute world_surface before the discard so the depth write uses
    // the correct surface position for all surviving pixels.
    let surface_or_exit = select(
        ray_origin + ray_dir * t_end,
        surface_pos,
        surface_found
    );
    let world_surface = u_wobject.world_transform * vec4<f32>(surface_or_exit, 1.0);

    // Deferred discards.
    if (!surface_found) { discard; }

    // Use the label ID cached during bisection — avoids a fallible re-sample
    // whose round() can snap to background at voxel boundaries.
    let color = get_label_color(surface_lid);
    if (color.a < 0.001) { discard; }

    // The surface normal is needed for shading in the iso modes, for the
    // debug visualisation, and for the ambient occlusion target in every
    // mode -- a flat_categorical surface is still a real label boundary.
    $$ set needs_normal = render_mode in ["iso_categorical", "gradient_debug", "smooth_iso"] or write_normal
    $$ if needs_normal
    let surface_voxel = clamp(
        norm_to_voxel(surface_hi_p, norm_size, dataset_size),
        vec3<f32>(0.0),
        dataset_size - vec3<f32>(0.001),
    );
    $$ if render_mode == "smooth_iso"
    // 3x3x3 Sobel normal.
    let obj_normal = smooth_label_normal(
        surface_voxel, surface_lid,
        surface_lut_entry, surface_lod_scale, surface_brick_corner_k,
        dataset_size,
    );
    $$ else
    let obj_normal = label_object_normal(
        surface_voxel,
        surface_lut_entry, surface_lod_scale, surface_brick_corner_k,
        surface_lid, dataset_size,
    );
    $$ endif
    $$ endif

    $$ if render_mode == "iso_categorical" or render_mode == "smooth_iso"
    // Shade with ambient + diffuse using the object-space gradient normal.
    let view_dir = normalize(-ray_dir);
    let normal   = select(
        normalize(obj_normal),
        vec3<f32>(0.0, 1.0, 0.0),
        dot(obj_normal, obj_normal) < 1e-10,
    );
    let N        = select(-normal, normal, dot(normal, view_dir) > 0.0);
    let ambient  = 0.3;
    let diffuse  = max(dot(N, view_dir), 0.0);
    let lit_color = color.rgb * (ambient + diffuse);
    $$ elif render_mode == "gradient_debug"
    // Visualize the normalised gradient magnitude as RGB (abs of each component).
    // Zero-length gradient (interior / degenerate surface) renders as black.
    // Brick-boundary artifacts show up as colour discontinuities.
    let lit_color = select(
        vec3<f32>(0.0),
        abs(normalize(obj_normal)),
        dot(obj_normal, obj_normal) > 1e-10,
    );
    $$ else
    let lit_color = color.rgb;
    $$ endif

    let ndc_surface = u_stdinfo.projection_transform
                    * u_stdinfo.cam_transform
                    * world_surface;

    out.color = vec4<f32>(lit_color, color.a * u_material.opacity);
    out.depth = ndc_surface.z / ndc_surface.w;

    $$ if write_normal
    // Ambient occlusion normal target.  ``obj_normal`` is a direction in
    // voxel space, so it needs the per-axis index-per-unit factor before it
    // is a direction in the object's own (normalized) space -- without it a
    // z-anisotropic dataset writes normals tilted toward z.
    out.normal = pack_view_normal(
        obj_normal * (dataset_size / norm_size),
        (u_stdinfo.cam_transform * world_surface).xyz,
    );
    $$ endif

    $$ if write_outline_id
    // Label key for the screen-space outline pass.  Zero-initialised
    // to 0 (= not outlined) on canvases without the outline_id target.
    out.outline_id = label_outline_key(surface_lid);
    $$ endif
    $$ if write_pick
    // surface_pos is in centred normalised object space; encode as [0, 1].
    let pick_coord = (surface_pos / norm_size) + vec3<f32>(0.5);
    out.pick = (
        pick_pack(u32(u_wobject.global_id), 20) +
        pick_pack(u32(clamp(pick_coord.x, 0.0, 1.0) * 16383.0), 14) +
        pick_pack(u32(clamp(pick_coord.y, 0.0, 1.0) * 16383.0), 14) +
        pick_pack(u32(clamp(pick_coord.z, 0.0, 1.0) * 16383.0), 14)
    );
    $$ endif

    return out;
}
