// Render planes: the ray-plane search of the "plane" render mode (plane
// rendering design v3, 5.1).
//
// Included by the four volume shaders, and only when they are built for
// ``render_mode == "plane"``: a shader built for any other mode contains
// none of this.  Each shader samples its own data at the hit; what they
// share is the search below.
//
// Requires from the including shader: ``u_render_planes`` (bound in plane
// mode only) and ``u_wobject``.  A plane is a frame in the scene's world
// space, pygfx ``(x, y, z)``: an origin, two in-plane unit vectors and an
// extent ``(min_0, max_0, min_1, max_1)`` measured from the origin along
// them.  An unbounded side is ``+-1e30``, never infinity.

struct PlaneHit {
    hit: bool,
    // The ray parameter of the hit.
    t: f32,
    // The plane's world normal, ``axis_0 x axis_1``.
    normal: vec3<f32>,
}

// The nearest plane the world-space ray ``ray_o + t * ray_d`` meets inside
// its extent, for ``t`` in ``[t_after, t_end]``; with ``inclusive`` false
// a hit at ``t_after`` itself does not count, which is how a caller walks
// from one plane to the next along a ray.
//
// A ray is affine in its parameter, so ``t`` is the same number in the
// node's local space and in world space: the caller passes the interval it
// has already cut to the data box and to the clipping planes.
fn plane_nearest_hit(
    ray_o: vec3<f32>, ray_d: vec3<f32>, t_after: f32, t_end: f32, inclusive: bool,
) -> PlaneHit {
    var result = PlaneHit(false, t_end, vec3<f32>(0.0));
    for (var i = 0; i < u_render_planes.count; i = i + 1) {
        let origin = u_render_planes.origin[i].xyz;
        let axis_0 = u_render_planes.axis_0[i].xyz;
        let axis_1 = u_render_planes.axis_1[i].xyz;
        let extent = u_render_planes.extent[i];
        let normal = cross(axis_0, axis_1);
        let denom = dot(ray_d, normal);
        if (abs(denom) < 1e-20) { continue; }
        let t_i = dot(origin - ray_o, normal) / denom;
        if (t_i < t_after || t_i > result.t) { continue; }
        if (!inclusive && t_i <= t_after) { continue; }
        let rel = ray_o + t_i * ray_d - origin;
        let c0 = dot(rel, axis_0);
        let c1 = dot(rel, axis_1);
        if (c0 < extent.x || c0 > extent.y || c1 < extent.z || c1 > extent.w) {
            continue;
        }
        result = PlaneHit(true, t_i, normal);
    }
    return result;
}

// A plane's world normal as a covector in the node's local space, which is
// what ``pack_view_normal`` takes.  That function faces the result to the
// viewer, so the two sides of a plane are shaded alike.
fn plane_normal_local(world_normal: vec3<f32>) -> vec3<f32> {
    return transpose(mat3x3<f32>(
        u_wobject.world_transform[0].xyz,
        u_wobject.world_transform[1].xyz,
        u_wobject.world_transform[2].xyz,
    )) * world_normal;
}
