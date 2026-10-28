// Slab projection: one value per ray from the samples inside a slab
// (oblique 2D design v3, 4.2 and 5.3.8).
//
// Included by the multiscale image brick shader when it is built with a
// ``slab_op`` other than "none" (the image's ``slab_projection``).  Labels
// never project, so the labels shader has no slab mode.
//
// The slab is a window along the ray: the caller gives the anchor ``t_anchor``
// (where the ray meets the slab's centre plane), the half-open window in
// slab depth ``[-h, +h)``, and the rate ``depth_per_t`` at which the slab
// depth grows along the ray.  The depth is measured along an explicit
// direction (D10), never derived from a cross product in pygfx order.  The
// include takes the window rather than the scene's thickness, so a 3D render
// plane can use it later with the view ray's own interval (10.2).
//
// Requires from the including shader: ``setup_brick``, ``sample_atlas``,
// ``norm_to_voxel``, ``index_to_norm``, ``BrickInfo`` and ``u_wobject``.

struct SlabAccumulator {
    vmax: f32,
    vmin: f32,
    sum: f32,
    count: f32,
    t_max: f32,
    t_min: f32,
};

fn slab_begin(t_anchor: f32) -> SlabAccumulator {
    return SlabAccumulator(-3.0e38, 3.0e38, 0.0, 0.0, t_anchor, t_anchor);
}

fn slab_sample(acc: ptr<function, SlabAccumulator>, value: f32, t: f32) {
    if (value > (*acc).vmax) {
        (*acc).vmax = value;
        (*acc).t_max = t;
    }
    if (value < (*acc).vmin) {
        (*acc).vmin = value;
        (*acc).t_min = t;
    }
    (*acc).sum = (*acc).sum + value;
    (*acc).count = (*acc).count + 1.0;
}

// ``(value, t of the pick)``: the winning sample for max and min (O12), the
// anchor (the centre plane) for the mean.
fn slab_end(acc: SlabAccumulator, t_anchor: f32) -> vec2<f32> {
    $$ if slab_op == 'max'
    return vec2<f32>(acc.vmax, acc.t_max);
    $$ elif slab_op == 'min'
    return vec2<f32>(acc.vmin, acc.t_min);
    $$ else
    return vec2<f32>(acc.sum / max(acc.count, 1.0), t_anchor);
    $$ endif
}

struct SlabResult {
    found: bool,
    value: f32,
    t_pick: f32,
};

// The slab depth of a point in the node's normalized space.
fn slab_depth_of(norm_pos: vec3<f32>, direction: vec3<f32>, depth_origin: f32) -> f32 {
    let world = (u_wobject.world_transform * vec4<f32>(norm_pos, 1.0)).xyz;
    return dot(world, direction) - depth_origin;
}

// Reduce the samples of one ray inside the slab (the loop of K4).
//
// Samples are one ray step apart -- the Euclidean length of one voxel of the
// anchor's drawn level along the ray, over ``steps_per_voxel`` (K6) --
// anchored on the centre plane, or, for a ray along one data axis
// (``along_axis``), shifted onto that axis's voxel centres so the reduction
// is exact.  A sample counts when it lies in the box-and-clip interval
// ``[t_lo, t_hi]`` and is resident (O9), and, with nearest sampling, when
// the centre of the voxel it reads is inside the window; with linear
// sampling, when the sample itself is.  The brick is looked up again only
// when a sample leaves the current one.  A window that holds no counted
// sample because it is thinner than one voxel of the drawn level (a coarse
// level while moving) reads the voxel the anchor is in instead.
fn slab_reduce(
    near_pos: vec3<f32>,
    ray_dir: vec3<f32>,
    inv_ray_dir: vec3<f32>,
    t_anchor: f32,
    t_lo: f32,
    t_hi: f32,
    half_extent: f32,
    depth_per_t: f32,
    direction: vec3<f32>,
    steps_per_voxel: f32,
    along_axis: bool,
    nearest: bool,
    norm_size: vec3<f32>,
    dataset_size: vec3<f32>,
) -> SlabResult {
    var result = SlabResult(false, 0.0, t_anchor);
    let anchor_pos = near_pos + ray_dir * t_anchor;
    let depth_origin = slab_depth_of(anchor_pos, direction, 0.0);
    // The anchor's level sets the step: a brick there, or the nearest one
    // the ray reaches inside the data.
    var anchor_brick = setup_brick(
        near_pos, ray_dir, inv_ray_dir, clamp(t_anchor, t_lo, t_hi), t_hi,
        norm_size, dataset_size);
    if (!anchor_brick.valid) {
        anchor_brick = setup_brick(
            near_pos, ray_dir, inv_ray_dir, t_lo, t_hi, norm_size, dataset_size);
    }
    var lod_scale = vec3<f32>(1.0);
    if (anchor_brick.valid) { lod_scale = anchor_brick.lod_scale; }
    let du_dt = ray_dir * (dataset_size / norm_size) / lod_scale;
    let voxels_per_t = length(du_dt);
    if (voxels_per_t < 1e-12 || abs(depth_per_t) < 1e-20) {
        return result;
    }
    let dt = 1.0 / (voxels_per_t * max(steps_per_voxel, 1e-3));
    let sgn = select(-1.0, 1.0, depth_per_t >= 0.0);
    var t0 = t_anchor;
    if (along_axis && anchor_brick.valid) {
        // Onto the voxel centres of the axis the ray runs along.
        let a = abs(du_dt);
        var k = 0;
        if (a.y > a.x && a.y >= a.z) { k = 1; }
        if (a.z > a.x && a.z > a.y) { k = 2; }
        let w = norm_to_voxel(near_pos + ray_dir * t0, norm_size, dataset_size)
            / lod_scale - anchor_brick.brick_corner_k;
        let frac = w[k] - round(w[k]);
        t0 = t0 - frac / du_dt[k];
    }
    // The window in steps, a step either side for the centre rule.
    let span = half_extent / abs(depth_per_t);
    let j_lo = i32(floor((t_anchor - sgn * span - t0) * sgn / dt)) - 1;
    let j_hi = i32(ceil((t_anchor + sgn * span - t0) * sgn / dt)) + 1;
    // Samples on the window's faces sit there up to float error: a
    // thousandth of a step decides them, keeping [-h, +h) half open.
    let face_eps = 1e-3 * dt * abs(depth_per_t);
    var acc = slab_begin(t_anchor);
    var cur = anchor_brick;
    var cur_t0 = t_lo;
    var cur_set = false;
    for (var j = j_lo; j <= j_hi && j < j_lo + 8192; j = j + 1) {
        let tj = t0 + sgn * f32(j) * dt;
        if (tj < t_lo || tj > t_hi) { continue; }
        let pos = near_pos + ray_dir * tj;
        if (!cur_set || tj >= cur.t_end || tj < cur_t0) {
            cur = setup_brick(near_pos, ray_dir, inv_ray_dir, tj, t_hi, norm_size, dataset_size);
            cur_t0 = tj;
            cur_set = true;
        }
        if (!cur.valid) { continue; }
        let voxel = norm_to_voxel(pos, norm_size, dataset_size);
        var depth = (tj - t_anchor) * depth_per_t;
        if (nearest) {
            // The centre of the voxel this sample reads, at its level.
            let w = voxel / cur.lod_scale - cur.brick_corner_k;
            let centre = (round(w) + cur.brick_corner_k) * cur.lod_scale;
            let centre_norm = index_to_norm(centre + vec3<f32>(0.5), norm_size, dataset_size);
            depth = slab_depth_of(centre_norm, direction, depth_origin);
        }
        if (depth < -half_extent - face_eps || depth >= half_extent - face_eps) {
            continue;
        }
        let v = sample_atlas(voxel, cur.lut_entry, cur.lod_scale, cur.brick_corner_k);
        slab_sample(&acc, v, tj);
    }
    if (acc.count < 0.5) {
        // A slab thinner than one voxel of the drawn level can hold no
        // voxel centre: it reads the one voxel its centre plane is in (4.2,
        // level of detail), if that point is inside the data and resident.
        if (t_anchor < t_lo || t_anchor > t_hi) { return result; }
        let at = setup_brick(
            near_pos, ray_dir, inv_ray_dir, t_anchor, t_hi, norm_size, dataset_size);
        if (!at.valid) { return result; }
        let voxel = norm_to_voxel(anchor_pos, norm_size, dataset_size);
        let v = sample_atlas(voxel, at.lut_entry, at.lod_scale, at.brick_corner_k);
        return SlabResult(true, v, t_anchor);
    }
    let reduced = slab_end(acc, t_anchor);
    return SlabResult(true, reduced.x, reduced.y);
}
