{$ include 'pygfx.std.wgsl' $}

// -- colormap include (conditional on template var) --
$$ if colormap_dim
{$ include 'pygfx.colormap.wgsl' $}
$$ endif

{$ include 'pygfx.image_common.wgsl' $}

// -- custom bindings injected by define_bindings --
// t_img / s_img     -- proxy texture (grid-dim), drives get_im_geometry()
// t_cache / s_cache -- tile cache texture
// t_lut             -- texture_2d<f32>, float32 LUT
// u_lut_params      -- LutParams uniform (auto-generated struct)
// u_block_scales    -- BlockScales uniform (auto-generated struct)

// NOTE: Do NOT define struct LutParams or struct BlockScales here.
// pygfx auto-generates them from the numpy dtype via structname=.
// Writing them manually causes a "redefinition" compilation error.

fn get_tile_scale(level: i32) -> vec2<f32> {
    // Return the per-level scale factor for within-tile coordinate remapping.
    // Level 0 is reserved (black). Level 1 = finest = scale 1.0.
    // Level k = scale 1/2^(k-1).
    switch level {
        case 1: { return vec2<f32>(u_block_scales.scale_1[0], u_block_scales.scale_1[1]); }
        case 2: { return vec2<f32>(u_block_scales.scale_2[0], u_block_scales.scale_2[1]); }
        case 3: { return vec2<f32>(u_block_scales.scale_3[0], u_block_scales.scale_3[1]); }
        case 4: { return vec2<f32>(u_block_scales.scale_4[0], u_block_scales.scale_4[1]); }
        case 5: { return vec2<f32>(u_block_scales.scale_5[0], u_block_scales.scale_5[1]); }
        case 6: { return vec2<f32>(u_block_scales.scale_6[0], u_block_scales.scale_6[1]); }
        case 7: { return vec2<f32>(u_block_scales.scale_7[0], u_block_scales.scale_7[1]); }
        case 8: { return vec2<f32>(u_block_scales.scale_8[0], u_block_scales.scale_8[1]); }
        case 9: { return vec2<f32>(u_block_scales.scale_9[0], u_block_scales.scale_9[1]); }
        default: { return vec2<f32>(0.0, 0.0); }
    }
}

{$ include 'cellier.tile_rule.wgsl' $}
{$ include 'cellier.level_mapping.wgsl' $}

fn sample_im_lut(texcoord: vec2<f32>) -> vec4<f32> {
    let block_size = vec2<f32>(u_lut_params.block_size_x, u_lut_params.block_size_y);
    let cache_size = vec2<f32>(u_lut_params.cache_size_x, u_lut_params.cache_size_y);
    let lut_size   = vec2<i32>(i32(u_lut_params.lut_size_x), i32(u_lut_params.lut_size_y));
    let vol_size   = vec2<f32>(u_lut_params.vol_size_x, u_lut_params.vol_size_y);
    let overlap    = u_lut_params.overlap;
    let padded_size = block_size + vec2<f32>(2.0 * overlap);

    // Position in level-0 voxel coordinates (x=W, y=H).
    let pos = clamp(texcoord * vol_size, vec2<f32>(0.0), vol_size - vec2<f32>(0.5));

    // Which tile does this pixel fall into?
    let tile_f   = floor(pos / block_size);
    // LUT index: clamp to valid range. lut_size is (gW, gH).
    let tile_idx = clamp(vec2<i32>(tile_f), vec2<i32>(0), lut_size - vec2<i32>(1));

    // LUT lookup: returns (cache_tile_x, cache_tile_y, level, 0).
    let lutv = textureLoad(t_lut, tile_idx, 0);

    // If level is 0, this tile has no data -- return black.
    let level = i32(lutv.z);
    if (level == 0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Cache slot origin in cache texture coordinates (voxel units).
    let tile_origin = vec2<f32>(lutv.x, lutv.y) * padded_size;

    // Level coordinate (cellier.level_mapping.wgsl): p = pos - 0.5 is the
    // centred data position, u = (p - t) / s the level-k one, and linear
    // sampling reads texel coordinate u + 0.5.  The tile corner comes from the
    // LUT cell (tile_corner_from_cell), the same rule that wrote the LUT; only
    // the offset inside the tile uses the float scale.  Clamp to the padded
    // tile, less half a texel for linear filtering, so a sample just outside
    // its tile repeats the edge texel rather than blending into the
    // neighbouring atlas slot.
    let sj = get_tile_scale(level);  // 1 / s
    let u  = (pos - vec2<f32>(0.5) - get_level_offset(level).xy) * sj;
    let corner_k = tile_corner_from_cell(tile_idx, level);
    let within_tile = clamp(u - corner_k + vec2<f32>(0.5),
                            vec2<f32>(0.5 - overlap),
                            block_size - vec2<f32>(0.5) + vec2<f32>(overlap));

    // Final cache sample coordinate (normalised).
    let cache_pos   = tile_origin + within_tile + vec2<f32>(overlap);
    let cache_coord = cache_pos / cache_size;

    return textureSample(t_cache, s_cache, cache_coord);
}


// -- vertex stage (matches standard pygfx image.wgsl) --

struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};

@vertex
fn vs_main(in: VertexInput) -> Varyings {
    var geo = get_im_geometry();

    // Select what face we are at
    let index = i32(in.vertex_index);
    let i0 = geo.indices[index];

    // Sample position, and convert to world pos, and then to ndc
    let data_pos = vec4<f32>(geo.positions[i0], 1.0);
    let world_pos = u_wobject.world_transform * data_pos;
    let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

    var varyings: Varyings;
    varyings.position = vec4<f32>(ndc_pos);
    varyings.world_pos = vec3<f32>(world_pos.xyz);
    varyings.texcoord = vec2<f32>(geo.texcoords[i0]);
    return varyings;
}


// -- fragment stage --

@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    // Sample through the LUT indirection.
    let base = sample_im_lut(varyings.texcoord);
    let raw = vec4<f32>(base.r, 0.0, 0.0, 1.0);

    // Apply clim + colormap (standard pygfx machinery).
    let color = sampled_value_to_color(raw);

    // Move to physical colorspace (linear photon count)
    $$ if colorspace == 'srgb' or colorspace.startswith('yuv')
        let physical_color = srgb2physical(color.rgb);
    $$ else
        let physical_color = color.rgb;
    $$ endif
    let opacity = color.a * u_material.opacity;
    let out_color = vec4<f32>(physical_color, opacity);

    var out: FragmentOutput;
    out.color = out_color;

    $$ if write_pick
    out.pick = (
        pick_pack(u32(u_wobject.global_id), 20) +
        pick_pack(u32(varyings.texcoord.x * 4194303.0), 22) +
        pick_pack(u32(varyings.texcoord.y * 4194303.0), 22)
    );
    $$ endif

    return out;
}
