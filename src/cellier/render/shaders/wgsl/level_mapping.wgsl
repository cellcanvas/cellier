// Level placement: the translation half of each level's level -> data map.
//
// Plan v2 D1 (level-0 centre coordinates, cellier.render._level_mapping):
//
//     p = s_k * u + t_k        u = (p - t_k) / s_k
//
// where p is the data (level-0 voxel) coordinate with voxel i centred on
// p = i, and u the level-k coordinate with level voxel i centred on u = i.
// A shader that starts from an edge-convention position pos (voxel i covers
// [i, i + 1)) converts once, p = pos - 0.5.  Linear sampling reads texel
// coordinate u + 0.5; nearest sampling reads voxel floor(u + 0.5).
//
// offset_k is t_k in level-0 voxels, shader order (x, y, z); the 2D buffer
// fills x and y only.  s_k comes from each shader's own get_*_scale (the 2D
// buffer stores 1 / s_k, the 3D buffer s_k).
//
// Requires u_block_scales.offset_1 .. offset_9.

fn get_level_offset(level: i32) -> vec3<f32> {
    switch level {
        case 1: { return u_block_scales.offset_1.xyz; }
        case 2: { return u_block_scales.offset_2.xyz; }
        case 3: { return u_block_scales.offset_3.xyz; }
        case 4: { return u_block_scales.offset_4.xyz; }
        case 5: { return u_block_scales.offset_5.xyz; }
        case 6: { return u_block_scales.offset_6.xyz; }
        case 7: { return u_block_scales.offset_7.xyz; }
        case 8: { return u_block_scales.offset_8.xyz; }
        case 9: { return u_block_scales.offset_9.xyz; }
        default: { return vec3<f32>(0.0); }
    }
}

// A brick's corner in level-k voxels, moved by the level's placement so that
// a per-sample lookup needs no offset: (p - t) / s - corner = p / s -
// (corner + t / s).  `shift` is extra level-0 voxels folded in the same way
// (the 3D labels shader passes 0.5, its index -> centred conversion).  The
// 3D shaders compute this once per brick; t is constant within a brick.
fn placed_corner_k(corner_k: vec3<f32>, level: i32, scale: vec3<f32>, shift: f32) -> vec3<f32> {
    return corner_k + (get_level_offset(level) + vec3<f32>(shift)) / scale;
}
