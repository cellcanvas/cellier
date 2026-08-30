// Screen-space ambient occlusion.
//
// One file, three fragment stages, selected by the `stage` template var.
// Each stage is compiled on its own, because the three inner passes bind
// different textures and a helper referring to a binding the pass does
// not declare is a compile error rather than dead code.
//
//   stage == "compute"    depthTex (+ normalTex, pickTex, lutTex) -> aoTex
//   stage == "blur"       aoTex                                   -> aoBlurTex
//   stage == "composite"  colorTex + aoTex                        -> targetTex
//
// Bindings are assigned by the Python side (_ssao.py); this file assumes
//     0 u_effect    uniform
//     1 texSampler  sampler
//     2 depthTex    texture_depth_2d   (compute)
//     2 aoTex       texture_2d<f32>    (blur)
//     2 colorTex    texture_2d<f32>    (composite)
//     3 normalTex   texture_2d<f32>    (compute, has_normal_target)
//     3 aoTex       texture_2d<f32>    (composite)
//     4 pickTex     texture_2d<u32>    (compute, has_exclusions)
//     5 lutTex      texture_2d<u32>    (compute, has_exclusions)
//
// Template vars: stage, n_samples, blur_radius, has_normal_target,
//                has_exclusions.
//
// Sign conventions, since they are easy to get backwards: pygfx cameras
// are right-handed and look down -z, so view-space z is negative and a
// *larger* z is *nearer* the camera.

$$ if stage == "compute"

// ---------------------------------------------------------------------------
// Depth -> view space
//
// to_view_pos and reconstruct_view_normal below are copied verbatim from
// pygfx's NormalPass.wgsl (pygfx/renderers/wgpu/engine/effectpasses.py),
// which implements atyuwen's 5-tap reconstruction:
//     https://atyuwen.github.io/posts/normal-reconstruction/
// They are copied rather than imported because upstream keeps them inside
// a class attribute string that also carries a debug fs_main; slicing
// that string apart at import time would be brittle in exactly the way
// this project already regrets elsewhere.  tests/render/test_ssao.py
// asserts the upstream functions still exist with the same signatures, so
// a divergence is visible rather than silent.
// ---------------------------------------------------------------------------

fn to_view_pos(uv: vec2<f32>, depth: f32, projection_transform_inv: mat4x4<f32>) -> vec3<f32> {
    // map uv to NDC, note flipping Y
    // take depth directly from depth buffer (non-linear)
    let ndc = vec4<f32>(
        uv.x * 2.0 - 1.0,
        1.0 - uv.y * 2.0,
        depth,
        1.0);
    // convert to view space
    let h = projection_transform_inv * ndc;
    return h.xyz / h.w;
}

fn reconstruct_view_normal(uv: vec2<f32>, width: i32, height: i32, projection_transform_inv: mat4x4<f32>) -> vec3<f32> {
    // cast width and height to f32 once
    let _width = f32(width);
    let _height = f32(height);

    // sample raw (non-linear) depth buffer at current pixel
    // and at 2 pixels in each direction horizontally and vertically
    let texIndex = vec2<i32>(uv * vec2<f32>(_width, _height));
    let depth = textureLoad(depthTex, texIndex, 0);
    let H = vec4<f32>(
        textureLoad(depthTex, texIndex + vec2<i32>(-1, 0), 0), // left 1
        textureLoad(depthTex, texIndex + vec2<i32>(1, 0), 0), // right 1
        textureLoad(depthTex, texIndex + vec2<i32>(-2, 0), 0), // left 2
        textureLoad(depthTex, texIndex + vec2<i32>(2, 0), 0)); // right 2
    let V = vec4<f32>(
        textureLoad(depthTex, texIndex + vec2<i32>(0, -1), 0), // up 1 (+Y is down)
        textureLoad(depthTex, texIndex + vec2<i32>(0, 1), 0), // down 1
        textureLoad(depthTex, texIndex + vec2<i32>(0, -2), 0), // up 2
        textureLoad(depthTex, texIndex + vec2<i32>(0, 2), 0)); // down 2

    // extrapolate the depth from each pair towards the center pixel
    // and compute the absolute difference between that and the actual depth
    let he = abs((2 * H.xy - H.zw) - depth);
    let ve = abs((2 * V.xy - V.zw) - depth);

    // get view space position of center pixel and the four direct neighbors
    let view_space_pos = to_view_pos(uv, depth, projection_transform_inv);
    let view_space_pos_l = to_view_pos(uv + vec2<f32>(-1.0 / _width, 0.0), H.x, projection_transform_inv);
    let view_space_pos_r = to_view_pos(uv + vec2<f32>( 1.0 / _width, 0.0), H.y, projection_transform_inv);
    let view_space_pos_u = to_view_pos(uv + vec2<f32>( 0.0, -1.0 / _height), V.x, projection_transform_inv);
    let view_space_pos_d = to_view_pos(uv + vec2<f32>( 0.0, 1.0 / _height), V.y, projection_transform_inv);

    // get the difference vectors, note that they are oriented
    // through the center pixel and continue outwards
    let l = view_space_pos - view_space_pos_l; // left to center
    let r = view_space_pos_r - view_space_pos; // center to right
    let d = view_space_pos - view_space_pos_d; // down to center
    let u = view_space_pos_u - view_space_pos; // center to up

    // pick the horizontal difference vector (left or right)
    // based on which has the smallest depth extrapolation error
    let hDeriv = select(r, l, he.x < he.y);
    let vDeriv = select(d, u, ve.x < ve.y); // repeat for vertical

    // take the cross product of the horizontal and vertical difference vectors
    // and normalize to get the surface normal in view space
    let view_normal = normalize(cross(hDeriv, vDeriv));

    return view_normal;
}

// ---------------------------------------------------------------------------
// Per-pixel kernel rotation
//
// Neither reference article's 4x4 noise texture is needed: a small integer
// hash of (pixel, frame index) gives a rotation angle directly, which
// removes a texture, a binding and the noiseScale tiling arithmetic.  The
// frame term is what lets TemporalAccumulationPass -- which runs after
// this pass -- average the noise away once the camera settles.
// ---------------------------------------------------------------------------

fn hash_angle(texel: vec2<i32>, frame: i32) -> f32 {
    var h: u32 = (u32(texel.x) * 73856093u) ^ (u32(texel.y) * 19349663u);
    h = h ^ (u32(frame) * 83492791u);
    h = h ^ (h >> 13u);
    h = h * 1274126177u;
    h = h ^ (h >> 16u);
    return f32(h & 0xFFFFFFu) * (6.28318530718 / 16777216.0);
}

$$ if has_exclusions
// Every pygfx and cellier shader begins its pick payload with
// pick_pack(u32(u_wobject.global_id), 20), so global_id is bits 0..19.
fn pick_id(texel: vec2<i32>) -> u32 {
    let pick = textureLoad(pickTex, texel, 0);
    return pick.r | ((pick.g & 0xFu) << 16u);
}

// Bit 7 of the shared visual LUT entry is the "excluded from AO" flag.
fn ao_excluded(object_id: u32) -> bool {
    let x = i32(object_id & 1023u);
    let y = i32((object_id >> 10u) & 1023u);
    let value = textureLoad(lutTex, vec2<i32>(x, y), 0).r;
    return (value & 128u) != 0u;
}
$$ endif

fn ssao_at(texel: vec2<i32>, uv: vec2<f32>) -> f32 {
    let depth = textureLoad(depthTex, texel, 0);

    // Background.  The far plane carries no surface, and without this
    // early-out every silhouette grows a dark halo into empty space.
    if (depth >= 1.0) {
        return 1.0;
    }

    $$ if has_exclusions
    if (ao_excluded(pick_id(texel))) {
        return 1.0;
    }
    $$ endif

    let origin = to_view_pos(uv, depth, u_effect.projection_transform_inv);

    // Hybrid normal.  A shader that never wrote the normal target leaves
    // an exactly zero vector there, and a real normal is unit length, so
    // the choice needs no lookup table and no template var.
    $$ if has_normal_target
    let written = textureLoad(normalTex, texel, 0).xyz;
    let normal = select(
        reconstruct_view_normal(
            uv, u_effect.width, u_effect.height, u_effect.projection_transform_inv),
        normalize(written),
        dot(written, written) > 0.25);
    $$ else
    let normal = reconstruct_view_normal(
        uv, u_effect.width, u_effect.height, u_effect.projection_transform_inv);
    $$ endif

    // Gram-Schmidt against a rotated in-plane vector gives the TBN basis.
    let angle     = hash_angle(texel, u_effect.frame_index);
    let rv        = vec3<f32>(cos(angle), sin(angle), 0.0);
    let tangent   = normalize(rv - normal * dot(rv, normal));
    let bitangent = cross(normal, tangent);
    let tbn       = mat3x3<f32>(tangent, bitangent, normal);

    let dims  = vec2<f32>(f32(u_effect.width), f32(u_effect.height));
    let limit = vec2<i32>(u_effect.width - 1, u_effect.height - 1);

    var occlusion = 0.0;
    for (var i = 0; i < {{ n_samples }}; i += 1) {
        let sample_pos = origin + (tbn * u_effect.kernel[i].xyz) * u_effect.radius;

        // Project the sample point back to a texture coordinate.
        let clip = u_effect.projection_transform * vec4<f32>(sample_pos, 1.0);
        if (clip.w <= 0.0) {
            continue;
        }
        let ndc  = clip.xyz / clip.w;
        // The Y flip is the inverse of the one to_view_pos applies.
        // Getting it wrong produces AO that looks almost right and is
        // vertically mirrored.
        let s_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (s_uv.x < 0.0 || s_uv.x > 1.0 || s_uv.y < 0.0 || s_uv.y > 1.0) {
            continue;
        }

        let s_texel = clamp(vec2<i32>(s_uv * dims), vec2<i32>(0, 0), limit);
        let s_depth = textureLoad(depthTex, s_texel, 0);
        if (s_depth >= 1.0) {
            continue;  // background occludes nothing
        }

        let scene_z = to_view_pos(
            s_uv, s_depth, u_effect.projection_transform_inv).z;

        // Range check: discard comparisons between surfaces that are not
        // actually neighbours, which is what stops a foreground object
        // casting occlusion onto a distant background.
        let range = smoothstep(0.0, 1.0,
                               u_effect.radius / max(abs(origin.z - scene_z), 1e-8));
        // "The visible surface in that direction is nearer than the
        // sample point", i.e. the sample is inside geometry.
        let hit   = select(0.0, 1.0, scene_z >= sample_pos.z + u_effect.bias);
        occlusion += hit * range;
    }

    return clamp(1.0 - occlusion / f32({{ n_samples }}), 0.0, 1.0);
}

@fragment
fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(varyings.position.xy);
    let uv    = (vec2<f32>(texel) + vec2<f32>(0.5))
              / vec2<f32>(f32(u_effect.width), f32(u_effect.height));
    return vec4<f32>(ssao_at(texel, uv), 0.0, 0.0, 1.0);
}

$$ elif stage == "blur"

// A single (2r+1)^2 box over the AO texture.  Deliberately not separable:
// at r = 2 two passes and a third private texture cost more than the taps
// they save.
@fragment
fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(varyings.position.xy);
    let limit = vec2<i32>(textureDimensions(aoTex)) - vec2<i32>(1, 1);

    var total = 0.0;
    var count = 0.0;
    for (var dy = -{{ blur_radius }}; dy <= {{ blur_radius }}; dy += 1) {
        for (var dx = -{{ blur_radius }}; dx <= {{ blur_radius }}; dx += 1) {
            let t = clamp(texel + vec2<i32>(dx, dy), vec2<i32>(0, 0), limit);
            total += textureLoad(aoTex, t, 0).r;
            count += 1.0;
        }
    }
    return vec4<f32>(total / count, 0.0, 0.0, 1.0);
}

$$ else

// Multiply the occlusion into the composited colour.
//
// Both reference articles apply AO to the ambient term inside the
// lighting pass; an effect pass sees only the final colour, so that is
// not available here.  For an unlit isosurface -- cellier's default --
// the multiply is not an approximation of shading, it *is* the shading.
// `strength` and `power` make it a look control rather than a claim
// about lighting.
@fragment
fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(varyings.position.xy);
    let ao    = textureLoad(aoTex, texel, 0).r;
    let color = textureLoad(colorTex, texel, 0);
    let factor = mix(1.0, pow(ao, u_effect.power), u_effect.strength);
    return vec4<f32>(color.rgb * factor, color.a);
}

$$ endif
