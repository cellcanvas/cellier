// Clipping planes for ray-marched volumes (clipping planes design v2, 4.3).
//
// pygfx's own snippet tests one position per fragment.  For a volume that
// position is on the back face of the proxy box, so it drops whole rays
// instead of cutting them.  Here the ray's sampled interval is clipped
// instead: once per ray, exactly, because the kept region (an intersection
// of half-spaces) is convex.
//
// Uses pygfx's ``clipping_planes`` uniform and ``n_clipping_planes``
// template variable.  A plane is ``(a, b, c, d)`` in world space; a point
// is kept where ``dot(abc, p) >= d``.

// The world normal of the plane that set the ray's start, or zero when the
// proxy box or the near plane did.  A hit that stays at that start is on a
// cut face and is shaded with this normal.
var<private> clip_start_normal: vec3<f32>;

// Clip ``origin + t * direction`` (node-local) for ``t`` in ``[t0, t1]``.
// Returns the clipped ``(t0, t1)``; the caller discards when the first is
// not before the second.
fn clip_ray_interval(
    origin: vec3<f32>, direction: vec3<f32>, t0: f32, t1: f32,
) -> vec2<f32> {
    var lo = t0;
    var hi = t1;
    clip_start_normal = vec3<f32>(0.0);
    $$ if n_clipping_planes
    let o = (u_wobject.world_transform * vec4<f32>(origin, 1.0)).xyz;
    let d = (u_wobject.world_transform * vec4<f32>(direction, 0.0)).xyz;
    for (var i = 0; i < {{ n_clipping_planes }}; i = i + 1) {
        let plane = u_material.clipping_planes[i];
        let a = dot(o, plane.xyz) - plane.w;
        let b = dot(d, plane.xyz);
        if (b > 0.0) {
            // The ray enters the kept side here.
            let t_enter = -a / b;
            if (t_enter > lo) {
                lo = t_enter;
                clip_start_normal = plane.xyz;
            }
        } else if (b < 0.0) {
            hi = min(hi, -a / b);
        } else if (a < 0.0) {
            // Parallel to the plane and on the clipped side.
            hi = lo - 1.0;
        }
    }
    $$ endif
    return vec2<f32>(lo, hi);
}

// Whether a clipping plane, not the box or the near plane, set the start.
fn clip_start_on_plane() -> bool {
    return dot(clip_start_normal, clip_start_normal) > 0.0;
}

// ``clip_start_normal`` as a covector in the node's local space.
fn clip_start_normal_local() -> vec3<f32> {
    return transpose(mat3x3<f32>(
        u_wobject.world_transform[0].xyz,
        u_wobject.world_transform[1].xyz,
        u_wobject.world_transform[2].xyz,
    )) * clip_start_normal;
}
