// Winding of a ray-marched volume's proxy box.
//
// The box's pipeline culls front faces: every fragment marches the whole
// ray from the near plane, so one face per pixel is all that may draw, and
// the back faces are the ones still on screen when the camera is inside
// the box.  Which faces are "front" is decided by screen-space winding,
// and a mirroring world or camera transform reverses it, which would cull
// the back faces instead (and draw nothing from inside the box).
//
// Maps a triangle-list vertex index to the index to read, reversing each
// triangle's order under a mirroring transform so that the outward faces
// stay counter-clockwise.  Same rule as pygfx's mesh shader.  Needs
// ``u_wobject`` and ``u_stdinfo``.
fn box_winding_index(vertex_index: u32) -> u32 {
    let mirrored = determinant(u_wobject.world_transform)
                 * determinant(u_stdinfo.cam_transform) < 0.0;
    let corner = vertex_index % 3u;
    return select(vertex_index, vertex_index + 2u - 2u * corner, mirrored);
}
