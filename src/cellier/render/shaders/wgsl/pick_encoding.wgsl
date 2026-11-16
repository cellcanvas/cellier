// The pick position of a brick-shader volume.
//
// The pick texel is 64 bits; the low 20 hold the object id, and the other 44
// hold the position as three fields whose widths follow the volume's axis
// lengths (cellier.render._pick_layout.pick_bit_layout, written into
// u_vol_params by build_vol_params_buffer).  An axis of n voxels gets
// bit_length(n - 1) bits to name the voxel and frac bits past it, so one step
// is 2^-frac level-0 voxels.  A step of a power-of-two fraction of a voxel
// lies inside one voxel, so storing floor(v * 2^frac) and reading back the
// step's centre never leaves the voxel v is in.  frac is negative only when
// the volume needs more than 44 bits; a step then spans 2^-frac voxels.
//
// Needs u_vol_params and pygfx.std.wgsl's pick_pack.

fn pick_frac_bits() -> vec3<i32> {
    return vec3<i32>(u_vol_params.pick_frac_bits_x,
                     u_vol_params.pick_frac_bits_y,
                     u_vol_params.pick_frac_bits_z);
}

fn pick_widths() -> vec3<i32> {
    return vec3<i32>(u_vol_params.pick_widths_x,
                     u_vol_params.pick_widths_y,
                     u_vol_params.pick_widths_z);
}

// One encoded step on each axis, in level-0 voxels.
fn pick_step_voxels() -> vec3<f32> {
    return ldexp(vec3<f32>(1.0), -pick_frac_bits());
}

// Pack a continuous level-0 voxel position v (voxel j spans [j, j + 1)) into
// the three position fields.  Call after pick_pack(global_id, 20).  ldexp
// scales by a power of two exactly, and a field is at most 24 bits, so the
// floor is exact in f32.  The clamp only acts on a point at or past the far
// face (or below zero), which the decoder reads as the nearest voxel inside.
fn pick_pack_voxel(v: vec3<f32>) -> vec4<u32> {
    let w   = pick_widths();
    let top = vec3<f32>((vec3<u32>(1u) << vec3<u32>(w)) - vec3<u32>(1u));
    let q   = vec3<u32>(clamp(floor(ldexp(v, pick_frac_bits())), vec3<f32>(0.0), top));
    return pick_pack(q.x, w.x) + pick_pack(q.y, w.y) + pick_pack(q.z, w.z);
}
