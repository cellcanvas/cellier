// -- View-space surface normal for the ambient occlusion pass --------------
//
// Included by every cellier volume shader.  Produces the vec4 written to
// the `normal` render target, which the ambient occlusion pass prefers
// over reconstructing a normal from depth.
//
// Why the target exists: reconstructing a normal by differencing depth
// works on meshes (0.66 deg median error) and fails on raymarched
// isosurfaces (34 deg, and worse the closer the user looks), because the
// isosurface of a trilinearly interpolated field carries about a quarter
// voxel of fixed world-space roughness.  The gradient these shaders
// already compute for shading measures 1.31 deg, so it is written here.
//
// Two conventions the occlusion pass depends on:
//
// * **The target is zero where nothing wrote it.**  WGSL zero-initialises
//   `var out: FragmentOutput;`, and a real normal is unit length, so the
//   pass picks per pixel between the written normal and reconstruction
//   with no lookup table and no template var.  Never write a zero or
//   near-zero vector here.
// * **The normal is in view space and faces the camera.**  That is what
//   `reconstruct_view_normal` produces, so the two sources have to agree
//   or half the frame would occlude inside-out.
//
// Requires from the including shader: `u_wobject` and `u_stdinfo`.

// Transform an object-space (local) normal to view space.
//
// The object-to-world step uses the **inverse transpose** rather than the
// matrix itself.  cellier routinely renders z-anisotropic data, where the
// two differ and using the matrix tilts every normal away from the surface
// it belongs to.  `world_transform_inv` is already in the wobject uniform,
// so the inverse transpose is one transpose of its upper-left 3x3.
//
// The world-to-view step uses the camera matrix directly: a camera
// transform is rigid, so there its inverse transpose is the same thing.
fn local_normal_to_view(local_normal: vec3<f32>) -> vec3<f32> {
    let inv = u_wobject.world_transform_inv;
    let normal_matrix = transpose(mat3x3<f32>(inv[0].xyz, inv[1].xyz, inv[2].xyz));
    let world_normal = normal_matrix * local_normal;
    let cam = u_stdinfo.cam_transform;
    let view_normal = mat3x3<f32>(cam[0].xyz, cam[1].xyz, cam[2].xyz) * world_normal;
    return view_normal;
}

// Pack an object-space normal for the `normal` render target.
//
// Flips the result toward the camera before writing.  In view space the
// camera sits at the origin looking down -z, so the direction from a
// surface point back to the camera is -normalize(view_pos); a normal
// pointing away from it would make the occlusion pass sample a hemisphere
// buried inside the geometry.
//
// `view_pos` is the surface position in view space.  A degenerate normal
// is written as exactly zero, which reads downstream as "nothing wrote
// this pixel" and falls back to depth reconstruction -- the right
// behaviour, since there is no usable normal to hand over.
fn pack_view_normal(local_normal: vec3<f32>, view_pos: vec3<f32>) -> vec4<f32> {
    let view_normal = local_normal_to_view(local_normal);
    let length_sq = dot(view_normal, view_normal);
    if (length_sq < 1e-12) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let unit = view_normal * inverseSqrt(length_sq);
    let to_camera = -normalize(view_pos);
    let faced = select(-unit, unit, dot(unit, to_camera) > 0.0);
    return vec4<f32>(faced, 1.0);
}
