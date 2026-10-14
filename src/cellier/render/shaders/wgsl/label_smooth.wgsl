// -- Smooth label surfaces: smooth_iso and smooth_geometry --------------
//
// Shared by the in-memory and multiscale labels shaders.
//
// Both modes work on a smooth field built per label.  Place a sample at
// every voxel centre; the field of label L at a point p is the trilinear
// weight of the voxel centres around p that carry L.  Inside one "dual
// cell" -- the cube between eight neighbouring voxel centres -- the eight
// corner labels are fixed and each label's field is a trilinear function.
//
// * smooth_iso keeps the voxel-face geometry and only shades with
//   label_smooth_normal, a normal that varies continuously over the faces.
// * smooth_geometry draws the surface where some label's field first
//   exceeds the level u_material.smooth_level (label_smooth_march), and
//   shades it with the same normal.  At the default level, 0.5, at most one
//   label can exceed it, so the surfaces partition space into smooth
//   per-label regions: flat sides stay on the voxel faces, convex corners
//   round off and concave corners fill in.  A voxel's own centre always has
//   weight 1, so a single voxel still draws, as a small rounded blob.  A
//   lower level grows every label; a higher one shrinks it.
//
// All positions here are in "cell space": the voxel with integer index n
// has its centre at n.  The including shader maps its own coordinates to
// cell space and provides
//
//     fn label_smooth_load(idx: vec3<i32>) -> i32
//
// returning the label of the voxel at index idx, including indices just
// outside the region it owns (up to two voxels; the normal reads a 4x4x4
// block).  It also provides u_label_params.background_label and, for
// smooth_geometry, u_material.smooth_level.

// Bisection steps on a cell's cubic; 2^-14 of a cell is far below a pixel.
const LABEL_SMOOTH_BISECT: i32 = 14;

// Outward surface normal of label `label` at cell-space position p.
//
// The Sobel gradient of the indicator (label == `label`) at each of the
// eight voxel centres around p, blended with p's trilinear weights.  The
// eight 3x3x3 Sobel stencils together cover the 4x4x4 block floor(p) - 1
// .. floor(p) + 2, read once.  The blend is what makes the normal vary
// continuously across a voxel face; a Sobel at the nearest voxel alone is
// constant over each voxel and shades in patches.
//
// The result is a gradient (a covector) in cell space and is not
// normalised; zero where the label has no gradient.
fn label_smooth_normal(p: vec3<f32>, label: i32) -> vec3<f32> {
    let base = vec3<i32>(floor(p));
    let fr   = p - floor(p);
    var g: array<f32, 64>;
    for (var z = 0; z < 4; z++) {
        for (var y = 0; y < 4; y++) {
            for (var x = 0; x < 4; x++) {
                let lab = label_smooth_load(base + vec3<i32>(x - 1, y - 1, z - 1));
                g[x + 4 * y + 16 * z] = f32(lab == label);
            }
        }
    }
    var grad = vec3<f32>(0.0);
    for (var c = 0; c < 8; c++) {
        let cx = c & 1;
        let cy = (c >> 1u) & 1;
        let cz = (c >> 2u) & 1;
        let ix = cx + 1;
        let iy = cy + 1;
        let iz = cz + 1;
        var s = vec3<f32>(0.0);
        // Sobel: central difference along each axis, [1, 2, 1] weights
        // across the two transverse axes.
        for (var a = -1; a <= 1; a++) {
            for (var b = -1; b <= 1; b++) {
                let w = select(1.0, 2.0, a == 0) * select(1.0, 2.0, b == 0);
                s.x += w * (g[(ix + 1) + 4 * (iy + a) + 16 * (iz + b)]
                          - g[(ix - 1) + 4 * (iy + a) + 16 * (iz + b)]);
                s.y += w * (g[(ix + a) + 4 * (iy + 1) + 16 * (iz + b)]
                          - g[(ix + a) + 4 * (iy - 1) + 16 * (iz + b)]);
                s.z += w * (g[(ix + a) + 4 * (iy + b) + 16 * (iz + 1)]
                          - g[(ix + a) + 4 * (iy + b) + 16 * (iz - 1)]);
            }
        }
        let wx = select(1.0 - fr.x, fr.x, cx == 1);
        let wy = select(1.0 - fr.y, fr.y, cy == 1);
        let wz = select(1.0 - fr.z, fr.z, cz == 1);
        grad += (wx * wy * wz) * s;
    }
    // The indicator rises into the label; the outward normal is opposite.
    return -grad;
}

// A cell-space point inside a voxel that carries `label`, near p.
//
// The voxel is the corner of p's dual cell carrying `label` with the most
// trilinear weight, and p is clamped into the middle half of it.  A
// smooth_geometry hit can sit in a background voxel where a concave corner
// is filled in, and a hit on a flat side sits exactly on a voxel face; a
// pick reported here always decodes to a voxel of the label that was drawn.
fn label_smooth_pick_point(p: vec3<f32>, label: i32) -> vec3<f32> {
    let base = vec3<i32>(floor(p));
    let fr   = p - floor(p);
    var best_w = -1.0;
    var best = vec3<f32>(floor(p + vec3<f32>(0.5)));
    for (var c = 0; c < 8; c++) {
        let off = vec3<i32>(c & 1, (c >> 1u) & 1, (c >> 2u) & 1);
        if (label_smooth_load(base + off) != label) { continue; }
        let w = select(1.0 - fr.x, fr.x, off.x == 1)
              * select(1.0 - fr.y, fr.y, off.y == 1)
              * select(1.0 - fr.z, fr.z, off.z == 1);
        if (w > best_w) {
            best_w = w;
            best = vec3<f32>(base + off);
        }
    }
    return clamp(p, best - vec3<f32>(0.25), best + vec3<f32>(0.25));
}

// -- smooth_geometry: first crossing along a ray ------------------------

struct LabelSmoothHit {
    found:    bool,
    t:        f32,
    label:    i32,
    // The field was already above the level where the march began.
    at_start: bool,
};

// Value at s of the cubic with coefficients k (k.x + k.y s + k.z s^2 + k.w s^3).
fn label_smooth_cubic(k: vec4<f32>, s: f32) -> f32 {
    return k.x + s * (k.y + s * (k.z + s * k.w));
}

// The first s in (0, len] where cubic k rises above zero, or -1.
//
// k(0) <= 0 is assumed.  The roots of the derivative split [0, len] into
// at most three monotonic pieces; the first piece whose end is above zero
// holds the crossing, which bisection then pins down.
fn label_smooth_first_rise(k: vec4<f32>, len: f32) -> f32 {
    // Derivative 3 k.w s^2 + 2 k.z s + k.y.
    let qa = 3.0 * k.w;
    let qb = 2.0 * k.z;
    let qc = k.y;
    var ends = array<f32, 3>(len, len, len);
    var n_ends = 0;
    if (abs(qa) > 1e-12) {
        let disc = qb * qb - 4.0 * qa * qc;
        if (disc > 0.0) {
            let r = sqrt(disc);
            let s1 = (-qb - r) / (2.0 * qa);
            let s2 = (-qb + r) / (2.0 * qa);
            let lo = min(s1, s2);
            let hi = max(s1, s2);
            if (lo > 0.0 && lo < len) { ends[n_ends] = lo; n_ends++; }
            if (hi > 0.0 && hi < len) { ends[n_ends] = hi; n_ends++; }
        }
    } else if (abs(qb) > 1e-12) {
        let s1 = -qc / qb;
        if (s1 > 0.0 && s1 < len) { ends[n_ends] = s1; n_ends++; }
    }
    ends[n_ends] = len;
    n_ends++;

    var start = 0.0;
    for (var i = 0; i < n_ends; i++) {
        let end = ends[i];
        if (label_smooth_cubic(k, end) > 0.0) {
            var lo = start;
            var hi = end;
            for (var r = 0; r < LABEL_SMOOTH_BISECT; r++) {
                let mid = 0.5 * (lo + hi);
                if (label_smooth_cubic(k, mid) > 0.0) { hi = mid; } else { lo = mid; }
            }
            return hi;
        }
        start = end;
    }
    return -1.0;
}

// Whether any corner of dual cell k carries a label.  Unrolled, with no
// array, so the empty-space walk stays in registers.
fn label_smooth_cell_has_label(k: vec3<i32>) -> bool {
    let bg = u_label_params.background_label;
    return label_smooth_load(k) != bg
        || label_smooth_load(k + vec3<i32>(1, 0, 0)) != bg
        || label_smooth_load(k + vec3<i32>(0, 1, 0)) != bg
        || label_smooth_load(k + vec3<i32>(1, 1, 0)) != bg
        || label_smooth_load(k + vec3<i32>(0, 0, 1)) != bg
        || label_smooth_load(k + vec3<i32>(1, 0, 1)) != bg
        || label_smooth_load(k + vec3<i32>(0, 1, 1)) != bg
        || label_smooth_load(k + vec3<i32>(1, 1, 1)) != bg;
}

// Walk the dual cells a ray crosses from t0 to t1 and return the first
// point where some label's field exceeds u_material.smooth_level.
//
// The ray in cell space is c(t) = c0 + dc * t; dc need not be unit length.
// In each dual cell [k, k + 1]^3 the field of each label along the ray is a
// cubic in t (a product of three linear factors per corner), so the
// crossing is found exactly per cell rather than by sampling, and a thin
// neck between two voxels that touch only along an edge is not stepped
// over.  max_cells bounds the walk.
fn label_smooth_march(
    c0: vec3<f32>, dc: vec3<f32>, t0: f32, t1: f32, max_cells: i32,
) -> LabelSmoothHit {
    var hit: LabelSmoothHit;
    hit.found = false;
    hit.at_start = false;
    hit.t = t1;
    hit.label = u_label_params.background_label;
    if (t0 >= t1) { return hit; }

    let bg = u_label_params.background_label;
    let level = u_material.smooth_level;
    let safe_dc = select(dc, vec3<f32>(1e-20), abs(dc) < vec3<f32>(1e-20));
    let inv_dc  = 1.0 / safe_dc;
    let step    = vec3<i32>(sign(safe_dc));
    let c_start = c0 + dc * t0;
    // Nudge along the ray so a start exactly on a cell face picks the cell
    // the ray goes into.
    var k = vec3<i32>(floor(c_start + sign(safe_dc) * 1e-4));
    let next_face = vec3<f32>(k) + select(vec3<f32>(0.0), vec3<f32>(1.0), step > vec3<i32>(0));
    var t_max   = (next_face - c0) * inv_dc;
    let t_delta = abs(inv_dc);

    var t_cell = t0;
    for (var i = 0; i < max_cells; i++) {
        let t_exit = min(min(min(t_max.x, t_max.y), t_max.z), t1);
        if (label_smooth_cell_has_label(k)) {
            // Corner c of cell k is voxel k + (c & 1, c >> 1 & 1, c >> 2 & 1).
            var lab: array<i32, 8>;
            for (var c = 0; c < 8; c++) {
                lab[c] = label_smooth_load(
                    k + vec3<i32>(c & 1, (c >> 1u) & 1, (c >> 2u) & 1));
            }
            // Ray inside the cell: p(s) = p0 + dc * s, s in [0, len].
            let p0  = c0 + dc * t_cell - vec3<f32>(k);
            let len = max(t_exit - t_cell, 0.0);
            // Each corner's weight is a product of one linear factor per
            // axis: p for the +1 corner, 1 - p for the 0 corner.  Expand it
            // to cubic coefficients once per corner.
            var corner_k: array<vec4<f32>, 8>;
            for (var c = 0; c < 8; c++) {
                let ux = select(vec2<f32>(1.0 - p0.x, -dc.x), vec2<f32>(p0.x, dc.x), (c & 1) == 1);
                let uy = select(vec2<f32>(1.0 - p0.y, -dc.y), vec2<f32>(p0.y, dc.y), ((c >> 1u) & 1) == 1);
                let uz = select(vec2<f32>(1.0 - p0.z, -dc.z), vec2<f32>(p0.z, dc.z), ((c >> 2u) & 1) == 1);
                corner_k[c] = vec4<f32>(
                    ux.x * uy.x * uz.x,
                    ux.y * uy.x * uz.x + ux.x * uy.y * uz.x + ux.x * uy.x * uz.y,
                    ux.y * uy.y * uz.x + ux.y * uy.x * uz.y + ux.x * uy.y * uz.y,
                    ux.y * uy.y * uz.y,
                );
            }
            // The first crossing of every distinct label in the cell; the
            // earliest wins.
            var best_s = len + 1.0;
            var best_label = bg;
            for (var a = 0; a < 8; a++) {
                let la = lab[a];
                if (la == bg) { continue; }
                var seen = false;
                for (var b = 0; b < a; b++) { seen = seen || lab[b] == la; }
                if (seen) { continue; }
                var kl = vec4<f32>(-level, 0.0, 0.0, 0.0);
                for (var b = a; b < 8; b++) {
                    if (lab[b] == la) { kl += corner_k[b]; }
                }
                var s_hit = -1.0;
                if (kl.x > 0.0) {
                    s_hit = 0.0;
                } else if (len > 0.0) {
                    s_hit = label_smooth_first_rise(kl, len);
                }
                if (s_hit >= 0.0 && s_hit < best_s) {
                    best_s = s_hit;
                    best_label = la;
                }
            }
            if (best_s <= len) {
                hit.found = true;
                hit.t = t_cell + best_s;
                hit.label = best_label;
                hit.at_start = i == 0 && best_s == 0.0;
                return hit;
            }
        }
        if (t_exit >= t1) { break; }

        // Step to the next cell.
        if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
            t_cell = t_max.x; k.x += step.x; t_max.x += t_delta.x;
        } else if (t_max.y <= t_max.z) {
            t_cell = t_max.y; k.y += step.y; t_max.y += t_delta.y;
        } else {
            t_cell = t_max.z; k.z += step.z; t_max.z += t_delta.z;
        }
    }
    return hit;
}
