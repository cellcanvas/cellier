// A MaskedImage's reads.
//
// Included by multiscale_volume_brick.wgsl when `masked` is set, after the
// value side's copy of the lookup (value_get_lod_scale, value_sample_atlas,
// ...), which _masked_volume_brick.py makes from the mask's by renaming.
//
// Coordinates: p is a CENTRED level-0 voxel position (voxel i centred on
// p = i) of the side named; u = (p - t_k) / s_k is level k's (level_mapping).
// Requires u_material.mask_to_value, mask_hi_1..9, value_hi_1..9,
// surface_start, surface_end, projection_steps_per_voxel, projection_cap.

// -- per-level bounds --------------------------------------------------------

// The last voxel centre of each level, in its own voxels, shader order.
fn mask_level_hi(level: u32) -> vec3<f32> {
    switch level {
        case 1u: { return u_material.mask_hi_1.xyz; }
        case 2u: { return u_material.mask_hi_2.xyz; }
        case 3u: { return u_material.mask_hi_3.xyz; }
        case 4u: { return u_material.mask_hi_4.xyz; }
        case 5u: { return u_material.mask_hi_5.xyz; }
        case 6u: { return u_material.mask_hi_6.xyz; }
        case 7u: { return u_material.mask_hi_7.xyz; }
        case 8u: { return u_material.mask_hi_8.xyz; }
        default: { return u_material.mask_hi_9.xyz; }
    }
}

fn value_level_hi(level: u32) -> vec3<f32> {
    switch level {
        case 1u: { return u_material.value_hi_1.xyz; }
        case 2u: { return u_material.value_hi_2.xyz; }
        case 3u: { return u_material.value_hi_3.xyz; }
        case 4u: { return u_material.value_hi_4.xyz; }
        case 5u: { return u_material.value_hi_5.xyz; }
        case 6u: { return u_material.value_hi_6.xyz; }
        case 7u: { return u_material.value_hi_7.xyz; }
        case 8u: { return u_material.value_hi_8.xyz; }
        default: { return u_material.value_hi_9.xyz; }
    }
}

// -- the mask's 3D read past a level's end ------------------------------------

// The bounds of the level last read, in level-k texel units: u in
// [0, hi] is voxel_k = u + t / s in [t / s, hi + t / s].  Looked up only when
// the level changes, so a march pays one compare a sample, not two switches.
var<private> extent_level: u32 = 0u;
var<private> extent_lo_k: vec3<f32>;
var<private> extent_hi_k: vec3<f32>;

// A coarse level made by halving an odd extent is shorter than the finest, so
// its far end would read the brick's zero padding where the data are absent.
// Clamped, a level's last voxel repeats past its end.
fn mask_extent_clamp(voxel_k: vec3<f32>, level: u32) -> vec3<f32> {
    if (level != extent_level) {
        extent_level = level;
        extent_lo_k = get_level_offset(i32(level)) / get_lod_scale(level);
        extent_hi_k = extent_lo_k + mask_level_hi(level);
    }
    return clamp(voxel_k, extent_lo_k, extent_hi_k);
}

// -- the value's lookup --------------------------------------------------------

struct MaskedSample {
    value: f32,
    found: bool,
};

// The LUT cell and brick of one value position, cached between samples so
// that the LUT and level lookups run only when a sample leaves the cell.
struct ValueCell {
    cell:   vec3<i32>,
    lut:    vec4<u32>,
    scale:  vec3<f32>,
    corner: vec3<f32>,
    valid:  bool,
};

fn value_block_size() -> vec3<f32> {
    return vec3<f32>(u_value_vol_params.block_size_x,
                     u_value_vol_params.block_size_y,
                     u_value_vol_params.block_size_z);
}

// The LUT cell of value level-0 position p, as the LUT indexes it.
fn value_cell_index(p: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor((p + vec3<f32>(0.5)) / value_block_size()));
}

fn value_cell_at(p: vec3<f32>) -> ValueCell {
    var c: ValueCell;
    c.cell = value_cell_index(p);
    c.lut = textureLoad(t_value_lut, c.cell, 0);
    c.valid = c.lut.w > 0u;
    if (c.valid) {
        c.scale = value_get_lod_scale(c.lut.w);
        c.corner = value_placed_corner_k(
            value_brick_corner_from_cell(c.cell, c.lut.w),
            i32(c.lut.w), c.scale, 0.0);
    }
    return c;
}

// Within the value image's voxel extent.
fn value_inside(p: vec3<f32>) -> bool {
    let size = vec3<f32>(u_value_vol_params.dataset_size_x,
                         u_value_vol_params.dataset_size_y,
                         u_value_vol_params.dataset_size_z);
    return all(p >= vec3<f32>(-0.5)) && all(p < size - vec3<f32>(0.5));
}

// Level-0 position p, read at level k: the snapped axes (snap = 1) moved
// half up to the nearest level-k voxel, as round_world_to_voxel does, and,
// with do_clamp, every axis clamped to the level's voxel centres.  The move
// is at most half a level voxel, inside the brick's ghost border.
fn tile_read_pos(p: vec3<f32>, scale: vec3<f32>, offset: vec3<f32>,
                 hi: vec3<f32>, snap: vec3<f32>, do_clamp: bool) -> vec3<f32> {
    var u = (p - offset) / scale;
    u = mix(u, floor(u + vec3<f32>(0.5)), snap);
    if (do_clamp) { u = clamp(u, vec3<f32>(0.0), hi); }
    return u * scale + offset;
}

// The value is clamped to its level's voxel centres with no fade, so no
// colour comes from no data.
fn value_read_pos(p: vec3<f32>, c: ValueCell, snap: vec3<f32>) -> vec3<f32> {
    return tile_read_pos(p, c.scale, value_get_level_offset(i32(c.lut.w)),
                         value_level_hi(c.lut.w), snap, true);
}

// The value at level-0 position p, `found` false off the image or where its
// LUT cell has no brick.
fn value_at_voxel_snapped(p: vec3<f32>, snap: vec3<f32>) -> MaskedSample {
    var out: MaskedSample;
    out.found = false;
    if (!value_inside(p)) { return out; }
    let c = value_cell_at(p);
    if (!c.valid) { return out; }
    out.value = value_sample_atlas(value_read_pos(p, c, snap), c.lut, c.scale, c.corner);
    out.found = true;
    return out;
}

fn value_at_voxel(p: vec3<f32>) -> MaskedSample {
    return value_at_voxel_snapped(p, vec3<f32>(0.0));
}

// -- 3D: the value along the surface normal ---------------------------------

// The max, mean or min of the value along the world direction `inward_world`
// from the surface point pos_norm (the mask node's normalized space), over
// [surface_start, surface_end] world units.  An empty interval reads the one
// point.  Steps at projection_steps_per_voxel per value voxel of the level the
// LUT gives at the surface, at most projection_cap samples (past it the step
// widens); a sample sits at the middle of its step.  `found` is false when no
// sample counted.
fn masked_project(pos_norm: vec3<f32>, inward_world: vec3<f32>) -> MaskedSample {
    var out: MaskedSample;
    out.found = false;
    let dir_norm = (u_wobject.world_transform_inv * vec4<f32>(inward_world, 0.0)).xyz;
    let p0 = (u_material.mask_to_value * vec4<f32>(pos_norm, 1.0)).xyz;
    let dp = (u_material.mask_to_value * vec4<f32>(dir_norm, 0.0)).xyz;
    let s0 = u_material.surface_start;
    let s1 = u_material.surface_end;
    if (s1 <= s0) { return value_at_voxel(p0 + dp * s0); }

    var scale = value_get_lod_scale(1u);
    if (value_inside(p0)) {
        let first = value_cell_at(p0);
        if (first.valid) { scale = first.scale; }
    }
    let voxels = length(dp / scale) * (s1 - s0);
    let n = clamp(u32(ceil(u_material.projection_steps_per_voxel * voxels)),
                  1u, u32(u_material.projection_cap));
    let ds = (s1 - s0) / f32(n);

    var acc_max: f32 = -3.0e38;
    var acc_min: f32 = 3.0e38;
    var acc_sum: f32 = 0.0;
    var acc_n: u32 = 0u;
    var cur: ValueCell;
    var have_cell = false;
    for (var i = 0u; i < n; i++) {
        let p = p0 + dp * (s0 + (f32(i) + 0.5) * ds);
        if (!value_inside(p)) { continue; }
        let cell = value_cell_index(p);
        if (!have_cell || any(cell != cur.cell)) {
            cur = value_cell_at(p);
            have_cell = true;
        }
        if (!cur.valid) { continue; }
        let v = value_sample_atlas(value_read_pos(p, cur, vec3<f32>(0.0)),
                                   cur.lut, cur.scale, cur.corner);
        acc_max = max(acc_max, v);
        acc_min = min(acc_min, v);
        acc_sum += v;
        acc_n += 1u;
    }
    if (acc_n == 0u) { return out; }
    out.found = true;
    $$ if surface_op == 'max'
    out.value = acc_max;
    $$ elif surface_op == 'min'
    out.value = acc_min;
    $$ else
    out.value = acc_sum / f32(acc_n);
    $$ endif
    return out;
}

$$ if render_mode == 'plane'
// -- 2D: the band round the mask's contour (7) --------------------------------
//
// Positions are centred level-0 voxels of the mask (p = norm_to_voxel(...)).
// Requires u_material.mask_snap, mask_edge_fade, surface_min_width_px,
// band_rays, band_cap, and t_mask_range (_cell_ranges.py).

// An aligned plane reads the mask as a 2D tile reads it: the
// sliced axis snapped, every axis clamped to the level's voxel centres, and
// the tile's fade at the data's edge.  A tilted one reads it at its true
// depth, as the plain route does.
fn mask_plane_aligned() -> bool {
    return u_material.mask_edge_fade > 0.5;
}

fn mask_read_pos(p: vec3<f32>, level: u32, scale: vec3<f32>) -> vec3<f32> {
    return tile_read_pos(p, scale, get_level_offset(i32(level)), mask_level_hi(level),
                         u_material.mask_snap.xyz, mask_plane_aligned());
}

// The tile path's edges (image_block.wgsl, sample_im_lut): the level-0
// position is clamped to [-0.5, N0 - 1] before it is mapped to level k, and
// the tile's overlap texels past the data are zero.  So a linear tile fades
// to zero over u in [-0.5, 0) and over u in (hi, (N0 - 1 - t) / s] at level
// k.  The read position is clamped to [0, hi] and this fade multiplies it;
// bilinear is separable, so the fade is one factor per in-plane axis.
fn mask_edge_fade(p: vec3<f32>, level: u32, scale: vec3<f32>) -> f32 {
    if (!mask_plane_aligned()) { return 1.0; }
    let n0 = vec3<f32>(u_vol_params.dataset_size_x, u_vol_params.dataset_size_y,
                       u_vol_params.dataset_size_z);
    let pc = clamp(p, vec3<f32>(-0.5), n0 - vec3<f32>(1.0));
    let u = (pc - get_level_offset(i32(level))) / scale;
    let past = max(-u, u - mask_level_hi(level));
    let f = vec3<f32>(1.0) - clamp(past, vec3<f32>(0.0), vec3<f32>(1.0));
    let g = mix(f, vec3<f32>(1.0), u_material.mask_snap.xyz);
    return g.x * g.y * g.z;
}

// A LUT cell of the mask, cached along a search ray as ValueCell is.
struct MaskCell {
    cell:   vec3<i32>,
    lut:    vec4<u32>,
    scale:  vec3<f32>,
    corner: vec3<f32>,
    valid:  bool,
};

// The LUT cell of level-0 voxel p, as setup_brick indexes it.
fn mask_cell_of(p: vec3<f32>) -> vec3<i32> {
    let block = vec3<f32>(u_vol_params.block_size_x, u_vol_params.block_size_y,
                          u_vol_params.block_size_z);
    let n0 = vec3<f32>(u_vol_params.dataset_size_x, u_vol_params.dataset_size_y,
                       u_vol_params.dataset_size_z);
    let lut_size = vec3<i32>(i32(u_vol_params.lut_size_x), i32(u_vol_params.lut_size_y),
                             i32(u_vol_params.lut_size_z));
    let index = clamp(p + vec3<f32>(0.5), vec3<f32>(0.0), n0 - vec3<f32>(0.001));
    return clamp(vec3<i32>(floor(index / block)), vec3<i32>(0),
                 lut_size - vec3<i32>(1));
}

fn mask_cell_at(cell: vec3<i32>) -> MaskCell {
    var c: MaskCell;
    c.cell = cell;
    c.lut = textureLoad(t_lut, cell, 0);
    c.valid = c.lut.w > 0u;
    if (c.valid) {
        c.scale = get_lod_scale(c.lut.w);
        c.corner = placed_corner_k(brick_corner_from_cell(cell, c.lut.w),
                                   i32(c.lut.w), c.scale, 0.0);
    }
    return c;
}

// The mask at p as the plane branch reads it, zero off the data's edge (the
// mask's true edge, where the tile rule fades: not an unloaded cell).
fn mask_cell_read(p: vec3<f32>, c: MaskCell) -> f32 {
    let n0 = vec3<f32>(u_vol_params.dataset_size_x, u_vol_params.dataset_size_y,
                       u_vol_params.dataset_size_z);
    if (any(p < vec3<f32>(-1.0)) || any(p > n0)) { return 0.0; }
    return sample_atlas(mask_read_pos(p, c.lut.w, c.scale), c.lut, c.scale, c.corner)
        * mask_edge_fade(p, c.lut.w, c.scale);
}

// Whether every read of cell c is on the pixel's side of the threshold: its
// brick's range, taken at the level the LUT draws it at.
fn mask_cell_skips(c: MaskCell, inside: bool) -> bool {
    let range = textureLoad(t_mask_range, c.cell, 0);
    if (range.z != f32(c.lut.w)) { return false; }
    let iso = u_material.threshold;
    // Parenthesised: naga >= wgpu 0.32 reads `y < iso, range.x >` as a
    // template list.
    return select((range.y < iso), (range.x >= iso), inside);
}

// The read at p is the data's own: within the level's voxel centres on every
// axis that is not snapped, where the edge fade is one.  The ranges leave the
// edge out, so a skip stays where this holds.
fn mask_reads_data(p: vec3<f32>, c: MaskCell) -> bool {
    let level = i32(c.lut.w);
    let u = (p - get_level_offset(level)) / c.scale;
    let free = u_material.mask_snap.xyz < vec3<f32>(0.5);
    let outside = (u < vec3<f32>(0.0)) | (u > mask_level_hi(c.lut.w));
    return !any(outside & free);
}

// Where a ray from p0 along dv (voxels per world unit) leaves LUT cell
// `cell`, in world units along it.
fn mask_cell_exit(p0: vec3<f32>, dv: vec3<f32>, cell: vec3<i32>) -> f32 {
    let block = vec3<f32>(u_vol_params.block_size_x, u_vol_params.block_size_y,
                          u_vol_params.block_size_z);
    let lo = vec3<f32>(cell) * block - vec3<f32>(0.5);
    let hi = lo + block;
    var t = 3.0e38;
    for (var a = 0; a < 3; a++) {
        if (dv[a] > 1e-12) { t = min(t, (hi[a] - p0[a]) / dv[a]); }
        if (dv[a] < -1e-12) { t = min(t, (lo[a] - p0[a]) / dv[a]); }
    }
    return t;
}

// The world size of one pixel at this fragment, from the camera.
fn pixel_world_size(pos_norm: vec3<f32>) -> f32 {
    let world = u_wobject.world_transform * vec4<f32>(pos_norm, 1.0);
    let clip = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world;
    let ndc = clip.xyz / clip.w;
    let step = vec3<f32>(2.0 / u_stdinfo.physical_size.x, 0.0, 0.0);
    let back = u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv
        * vec4<f32>(ndc + step, 1.0);
    return length(back.xyz / back.w - world.xyz / world.w);
}

// The placed band [surface_start, surface_end] along the inward normal,
// widened about its centre to surface_min_width_px pixels.
fn mask_band_limits(px: f32) -> vec2<f32> {
    var a = u_material.surface_start;
    var b = u_material.surface_end;
    let width = u_material.surface_min_width_px * px;
    if (b - a < width) {
        let c = 0.5 * (a + b);
        a = c - 0.5 * width;
        b = c + 0.5 * width;
    }
    return vec2<f32>(a, b);
}

// The pixel's share of the band, d its signed distance (world units).
fn mask_band_coverage(d: f32, px: f32) -> f32 {
    let band = mask_band_limits(px);
    let lo = max(band.x, d - 0.5 * px);
    let hi = min(band.y, d + 0.5 * px);
    return clamp((hi - lo) / px, 0.0, 1.0);
}

// Signed in-plane distance to the contour, world units, positive inside: the
// nearest sign change of m - iso along band_rays rays in the plane, one
// drawn-level voxel a step (band_cap steps a ray at most, past which the step
// widens), placed by linear interpolation.  Each ray stops at the best found
// so far.  Only the pixel's side is searched, as far as the placed band and
// half a pixel reach.  No crossing: +-3e38 (+-2e38 when the per-pixel skip
// found none), outside every band.  A step into a cell with no brick reads
// nothing and goes on, its last read kept: a distance can be over-estimated,
// never under.
fn mask_band_search(p0: vec3<f32>, m0: f32, lod_scale: vec3<f32>,
                    norm_size: vec3<f32>, dataset_size: vec3<f32>,
                    normal_world: vec3<f32>, px: f32) -> f32 {
    let iso = u_material.threshold;
    let f0 = m0 - iso;
    let inside = f0 >= 0.0;
    let band = mask_band_limits(px);
    let reach = select(max(-band.x, 0.0), max(band.y, 0.0), inside) + 0.5 * px;
    let none = select(-3.0e38, 3.0e38, inside);
    let n = normalize(normal_world);
    var helper = vec3<f32>(1.0, 0.0, 0.0);
    if (abs(n.x) > 0.9) { helper = vec3<f32>(0.0, 1.0, 0.0); }
    let e1 = normalize(cross(n, helper));
    let e2 = cross(n, e1);
    let inv = u_wobject.world_transform_inv;
    let inv3 = mat3x3<f32>(inv[0].xyz, inv[1].xyz, inv[2].xyz);
    let to_voxel = dataset_size / norm_size;
    let n_rays = u32(u_material.band_rays);
    let cap = u_material.band_cap;
    var best = reach;
    var found = false;

    // Per pixel: the reach as a box in level-0 voxels.  Within the
    // pixel's cell grown by its clearance, on the data's own reads, it holds
    // no crossing and no ray is cast.
    var cur = mask_cell_at(mask_cell_of(p0));
    if (cur.valid && mask_cell_skips(cur, inside)) {
        let clearance = vec3<f32>(textureLoad(t_mask_range, cur.cell, 0).w);
        let a1 = (inv3 * e1) * to_voxel;
        let a2 = (inv3 * e2) * to_voxel;
        let extent = reach * sqrt(a1 * a1 + a2 * a2);
        let block = vec3<f32>(u_vol_params.block_size_x, u_vol_params.block_size_y,
                              u_vol_params.block_size_z);
        let lo = vec3<f32>(cur.cell) * block - vec3<f32>(0.5);
        let in_cell = all(p0 - extent >= lo - clearance)
            && all(p0 + extent < lo + block + clearance);
        if (in_cell && mask_reads_data(p0 - extent, cur)
            && mask_reads_data(p0 + extent, cur)) {
            // 2e38, not 3e38, so debug_mode 'band_skip' can count it.
            return select(-2.0e38, 2.0e38, inside);
        }
    }

    for (var r = 0u; r < n_rays; r++) {
        let angle = 6.2831853 * (f32(r) + 0.5) / f32(n_rays);
        let dir_world = cos(angle) * e1 + sin(angle) * e2;
        // Level-0 voxels per world unit along this ray.
        let dv = (inv3 * dir_world) * to_voxel;
        var step = 1.0 / max(length(dv / lod_scale), 1e-6);
        var n_steps = ceil(best / step);
        if (n_steps > cap) {
            n_steps = cap;
            step = best / cap;
        }
        var prev = f0;
        for (var k = 1u; k <= u32(n_steps); k++) {
            let t = f32(k) * step;
            if (t - step >= best) { break; }
            let p = p0 + dv * t;
            let cell = mask_cell_of(p);
            if (any(cell != cur.cell)) {
                cur = mask_cell_at(cell);
                // Per ray: a cell wholly on the pixel's side holds no
                // crossing, so go on from its last step.
                if (cur.valid && mask_cell_skips(cur, inside)) {
                    let last = min(u32(floor(mask_cell_exit(p0, dv, cell) / step)),
                                   u32(n_steps));
                    let p_last = p0 + dv * (f32(last) * step);
                    // The box of data reads is convex: both ends in it, all in it.
                    if (last > k && mask_reads_data(p, cur)
                        && mask_reads_data(p_last, cur)) {
                        k = last;
                        prev = mask_cell_read(p_last, cur) - iso;
                        continue;
                    }
                }
            }
            // A cell with no brick is unknown, not zero: go on past it.
            if (!cur.valid) { continue; }
            let f = mask_cell_read(p, cur) - iso;
            if ((f >= 0.0) != inside) {
                // A crossing past the reach is no crossing: d = reach would
                // leave a coverage of zero only up to rounding.
                let frac = clamp(prev / (prev - f), 0.0, 1.0);
                let dist = t - step + frac * step;
                if (dist < best) {
                    best = dist;
                    found = true;
                }
                break;
            }
            prev = f;
        }
    }
    if (!found) { return none; }
    return select(-best, best, inside);
}

struct MaskedPlane {
    value:    f32,
    coverage: f32,
    found:    bool,
};

// A plane pixel of a MaskedImage: the mask read as a 2D image of it reads,
// its distance to the contour, the pixel's share of the band, and the value
// under it (its own value: no max, mean or min in 2D).  `found` is false
// where nothing is drawn: outside the band, or where the value has no data.
fn masked_plane_band(plane_pos: vec3<f32>, brick: BrickInfo, normal_world: vec3<f32>,
                     norm_size: vec3<f32>, dataset_size: vec3<f32>) -> MaskedPlane {
    var out: MaskedPlane;
    out.found = false;
    let p0 = norm_to_voxel(plane_pos, norm_size, dataset_size);
    let level = brick.lut_entry.w;
    let m0 = sample_atlas(mask_read_pos(p0, level, brick.lod_scale), brick.lut_entry,
                          brick.lod_scale, brick.brick_corner_k)
        * mask_edge_fade(p0, level, brick.lod_scale);
    let px = max(pixel_world_size(plane_pos), 1e-6);
    let d = mask_band_search(p0, m0, brick.lod_scale, norm_size, dataset_size,
                             normal_world, px);
    $$ if debug_mode == 'band_skip'
    // Debug: 1 where the per-pixel skip cast no ray, 0 elsewhere.
    out.value = select(0.0, 1.0, abs(d) == 2.0e38);
    out.coverage = 1.0;
    out.found = true;
    return out;
    $$ endif
    out.coverage = mask_band_coverage(d, px);
    if (out.coverage <= 0.0) { return out; }
    let v = value_at_voxel_snapped(
        (u_material.mask_to_value * vec4<f32>(plane_pos, 1.0)).xyz,
        u_material.value_snap.xyz);
    if (!v.found) { return out; }
    out.value = v.value;
    out.found = true;
    return out;
}
$$ endif
