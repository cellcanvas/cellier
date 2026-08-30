// In-memory image volume raycasting: mip, minip and iso.
//
// Adapted from pygfx's ``volume_ray.wgsl`` (pygfx 0.17).  cellier owns a
// copy for three reasons, in descending order of how much they matter:
//
// 1. **The iso branch can write the ``normal`` render target.**  The
//    ambient occlusion pass reconstructs a normal from depth for meshes
//    (0.66 deg median error) and cannot do so usefully for a raymarched
//    isosurface (34 deg, worsening as the user zooms in).  The gradient
//    this shader already computes for shading measures 1.31 deg, so it is
//    written out.  A volume rendered through pygfx's own shader has no way
//    to do that, which would leave in-memory volumes -- the very case the
//    feature exists for -- looking speckled.
// 2. **The iso branch's depth is fixed.**  Upstream builds the surface
//    position with a 3x3 matrix and so writes depth as if the volume sat at
//    the origin, which makes a translated iso volume win depth tests it
//    should lose.  ``scripts/pygfx_iso_depth_bug.py`` demonstrates it.  The
//    mip and minip branches upstream already use the full 4x4; only iso was
//    wrong.
// 3. **The normal uses the inverse transpose.**  Upstream rotates the
//    gradient by the world matrix itself.  For the anisotropic voxel
//    spacing cellier renders routinely the two differ, and the matrix tilts
//    every normal away from the surface it belongs to.
//
// Everything else -- the ray setup, the three raycast bodies, the lighting
// selection between scene lights and the builtin simple model -- is kept as
// close to upstream as the fixes allow, so a pygfx bump is a readable diff.

{# Includes #}
{$ include 'pygfx.std.wgsl' $}
$$ if colormap_dim
    {$ include 'pygfx.colormap.wgsl' $}
$$ endif
{$ include 'pygfx.volume_common.wgsl' $}
{$ include 'cellier.view_normal.wgsl' $}


struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};


struct RenderOutput {
    color: vec4<f32>,   // The final color for the current ray
    coord: vec3<f32>,   // The texture coord (for picking info) for the current ray
    depth: f32,         // The depth to write for the current ray
    // Surface normal in the volume's own (data) space, or exactly zero when
    // this mode has no surface.  Zero is meaningful downstream: the ambient
    // occlusion pass treats an unwritten normal target as "reconstruct from
    // depth instead", and a projection mode has no surface to hand it.
    local_normal: vec3<f32>,
};


$$ if mode in ['iso']
    // This mode uses lights

    $$ if num_point_lights > 0 or num_spot_lights > 0 or num_dir_lights > 0
        {$ include 'pygfx.light_phong.wgsl' $}

        struct ReflectedLight {
            direct_diffuse: vec3<f32>,
            direct_specular: vec3<f32>,
            indirect_diffuse: vec3<f32>,
            indirect_specular: vec3<f32>,
        };

        fn calculate_light(physical_albedo: vec3f, world_pos: vec3f, surface_normal: vec3f, view_dir: vec3f) -> vec3f {

            // Apply lighting
            var reflected_light: ReflectedLight = ReflectedLight(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
            var geometry: GeometricContext;
            geometry.position = world_pos;
            geometry.normal = surface_normal;
            geometry.view_dir = view_dir;

            var material: BlinnPhongMaterial;
            material.diffuse_color = physical_albedo;
            material.specular_color = srgb2physical(vec3f(0.2863));
            material.specular_shininess = u_material.shininess;
            material.specular_strength = 1.0;

            {$ include 'pygfx.light_punctual.wgsl' $}

            let ambient_color = u_ambient_light.color.rgb;  // already physical
            let irradiance = ambient_color;
            RE_IndirectDiffuse( irradiance, geometry, material, &reflected_light );

            var physical_color = reflected_light.direct_diffuse + reflected_light.direct_specular + reflected_light.indirect_diffuse + reflected_light.indirect_specular;
            physical_color += srgb2physical(u_material.emissive_color.rgb);

            return physical_color;
        }
    $$ else
        // No non-ambient lights in the scene: fall back to the builtin simple
        // model, which is what cellier's multiscale brick shader always uses.
        {$ include 'pygfx.light_phong_simple.wgsl' $}
        fn calculate_light(physical_albedo: vec3f, world_pos: vec3f, surface_normal: vec3f, view_dir: vec3f) -> vec3f {
            return lighting_phong(surface_normal, view_dir, physical_albedo);
        }
    $$ endif

$$endif


@vertex
fn vs_main(in: VertexInput) -> Varyings {

    // Our geometry is implicitly defined by the volume dimensions.
    var geo = get_vol_geometry();

    // Select what face we're at
    let index = i32(in.vertex_index);
    let i0 = geo.indices[index];

    // Sample position, and convert to world pos, and then to ndc
    let data_pos = vec4<f32>(nonlinear_transform(geo.positions[i0]), 1.0);
    let world_pos = u_wobject.world_transform * data_pos;
    let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

    // Prepare inverse matrix
    let ndc_to_data = u_wobject.world_transform_inv * u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv;

    var varyings: Varyings;

    varyings.position = vec4<f32>(ndc_pos);
    varyings.world_pos = vec3<f32>(world_pos.xyz);

    // The position on the face of the cube. We can say that it's the back
    // face, because we cull the front faces.  These positions are in data
    // positions (voxels) rather than texcoords (0..1), because distances
    // make more sense in that space.
    varyings.data_back_pos = vec4<f32>(data_pos);

    // Take care to take into account of the camera flipping any axii
    let cam_sign = sign(determinant(u_stdinfo.cam_transform));

    // NDC positions for the near and far clipping planes, transformed back
    // to data coordinates.  The view vector built from these is resistant to
    // perspective transforms, and makes a camera inside the volume render
    // only the part in front of it.
    let ndc_pos1 = vec4<f32>(ndc_pos.xy, -1.0 * cam_sign * ndc_pos.w, ndc_pos.w);
    let ndc_pos2 = vec4<f32>(ndc_pos.xy, cam_sign * ndc_pos.w, ndc_pos.w);
    varyings.data_near_pos = vec4<f32>(ndc_to_data * ndc_pos1);
    varyings.data_far_pos = vec4<f32>(ndc_to_data * ndc_pos2);

    return varyings;
}


@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {

    // clipping planes
    {$ include 'pygfx.clipping_planes.wgsl' $}

    // Get size of the volume
    let sizef = vec3<f32>(textureDimensions(t_img));

    // Step size as a float in voxels, between ~0.1 and 0.8, scaled by the
    // (square root of the) volume size.
    let relative_step_size = clamp(sqrt(max(sizef.x, max(sizef.y, sizef.z))) / 20.0, 0.1, 0.8);

    // Positions in data coordinates
    let back_pos = varyings.data_back_pos.xyz / varyings.data_back_pos.w;
    let far_pos = varyings.data_far_pos.xyz / varyings.data_far_pos.w;
    let near_pos = varyings.data_near_pos.xyz / varyings.data_near_pos.w;

    // Unit vector pointing in the view direction through this fragment.
    let view_ray = normalize(far_pos - near_pos);

    // The (signed) distance from back_pos to the first voxel to sample.
    var dist = dot(near_pos - back_pos, view_ray);
    dist = max(dist, min((-0.5 - back_pos.x) / view_ray.x, (sizef.x - 0.5 - back_pos.x) / view_ray.x));
    dist = max(dist, min((-0.5 - back_pos.y) / view_ray.y, (sizef.y - 0.5 - back_pos.y) / view_ray.y));
    dist = max(dist, min((-0.5 - back_pos.z) / view_ray.z, (sizef.z - 0.5 - back_pos.z) / view_ray.z));

    let front_pos = back_pos + view_ray * dist;

    let nsteps = i32(-dist / relative_step_size + 0.5);
    if( nsteps < 1 ) { discard; }

    // Starting position and step vector in texture coordinates.
    let start_coord = (front_pos + vec3<f32>(0.5, 0.5, 0.5)) / sizef;
    let step_coord = ((back_pos - front_pos) / sizef) / f32(nsteps);

    // Render
    let render_out: RenderOutput = raycast(sizef, nsteps, start_coord, step_coord);

    do_alpha_test(render_out.color.a);

    // Create fragment output.
    var out: FragmentOutput;
    out.color = render_out.color;
    out.depth = render_out.depth;

    $$ if write_normal
    // Ambient occlusion normal target.  ``local_normal`` is a gradient in
    // data (voxel) space, which is the volume's own object space, so
    // pack_view_normal's inverse-transpose handles the rest.  The mip and
    // minip modes return exactly zero here, which reads downstream as
    // "nothing wrote this pixel": a projection mode has no surface, and
    // handing the occlusion pass a normal derived from the depth of an
    // extremum sample would be worse than handing it nothing.
    if (dot(render_out.local_normal, render_out.local_normal) > 0.0) {
        let surface_data_pos = render_out.coord * sizef - vec3<f32>(0.5);
        let surface_view_pos = u_stdinfo.cam_transform
                             * u_wobject.world_transform
                             * vec4<f32>(surface_data_pos, 1.0);
        out.normal = pack_view_normal(render_out.local_normal, surface_view_pos.xyz);
    }
    $$ endif

    $$ if write_pick
    // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
    out.pick = (
        pick_pack(u32(u_wobject.global_id), 20) +
        pick_pack(u32(render_out.coord.x * 16383.0), 14) +
        pick_pack(u32(render_out.coord.y * 16383.0), 14) +
        pick_pack(u32(render_out.coord.z * 16383.0), 14)
    );
    $$ endif
    return out;
}


// ---- The different supported render modes ----

$$ if mode == 'mip'
    // raycasting function for MIP rendering.
    fn raycast(sizef: vec3<f32>, nsteps: i32, start_coord: vec3<f32>, step_coord: vec3<f32>) -> RenderOutput {
        let nstepsf = f32(nsteps);

        // Primary loop: find the approximate location of the maximum.
        var the_ref = -999999.0;
        var the_coord = start_coord;
        var the_value : vec4<f32>;
        for (var iter=0.0; iter<nstepsf; iter=iter+1.0) {
            let coord = start_coord + iter * step_coord;
            let value = sample_vol(coord, sizef);
            let reff = value.r;
            if (reff > the_ref) {
                the_ref = reff;
                the_coord = coord;
                the_value = value;
            }
        }

        // Secondary loop: close in on a more accurate position.
        var substep_coord = step_coord;
        for (var iter2=0; iter2<4; iter2=iter2+1) {
            substep_coord = substep_coord * 0.5;
            let coord1 = the_coord - substep_coord;
            let coord2 = the_coord + substep_coord;
            let value1 = sample_vol(coord1, sizef);
            let value2 = sample_vol(coord2, sizef);
            let ref1 = value1.r;
            let ref2 = value2.r;
            if (ref1 >= the_ref) {  // deliberate larger-equal
                the_ref = ref1;
                the_coord = coord1;
                the_value = value1;
            } else if (ref2 > the_ref) {
                the_ref = ref2;
                the_coord = coord2;
                the_value = value2;
            }
        }

        let color = sampled_value_to_color(the_value);
        $$ if colorspace == 'srgb'
            let physical_color = srgb2physical(color.rgb);
        $$ else
            let physical_color = color.rgb;
        $$ endif

        let opacity = color.a * u_material.opacity;
        let out_color = vec4<f32>(physical_color, opacity);

        let data_pos = the_coord * sizef - vec3<f32>(0.5, 0.5, 0.5);
        let world_pos = u_wobject.world_transform * vec4<f32>(data_pos, 1.0);
        let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

        var out: RenderOutput;
        out.color = out_color;
        out.coord = the_coord;
        out.depth = ndc_pos.z / ndc_pos.w;
        out.local_normal = vec3<f32>(0.0);
        return out;
    }

$$ elif mode == 'minip'
    // raycasting function for MINimum Intensity Projection rendering.
    fn raycast(sizef: vec3<f32>, nsteps: i32, start_coord: vec3<f32>, step_coord: vec3<f32>) -> RenderOutput {
        let nstepsf = f32(nsteps);

        // Primary loop: find the approximate location of the minimum.
        var the_ref = 999999.0;
        var the_coord = start_coord;
        var the_value : vec4<f32>;
        for (var iter=0.0; iter<nstepsf; iter=iter+1.0) {
            let coord = start_coord + iter * step_coord;
            let value = sample_vol(coord, sizef);
            let reff = value.r;
            if (reff < the_ref) {
                the_ref = reff;
                the_coord = coord;
                the_value = value;
            }
        }

        // Secondary loop: close in on a more accurate position.
        var substep_coord = step_coord;
        for (var iter2=0; iter2<4; iter2=iter2+1) {
            substep_coord = substep_coord * 0.5;
            let coord1 = the_coord - substep_coord;
            let coord2 = the_coord + substep_coord;
            let value1 = sample_vol(coord1, sizef);
            let value2 = sample_vol(coord2, sizef);
            let ref1 = value1.r;
            let ref2 = value2.r;
            if (ref1 <= the_ref) {  // deliberate less-equal
                the_ref = ref1;
                the_coord = coord1;
                the_value = value1;
            } else if (ref2 < the_ref) {
                the_ref = ref2;
                the_coord = coord2;
                the_value = value2;
            }
        }

        let color = sampled_value_to_color(the_value);
        $$ if colorspace == 'srgb'
            let physical_color = srgb2physical(color.rgb);
        $$ else
            let physical_color = color.rgb;
        $$ endif

        let opacity = color.a * u_material.opacity;
        let out_color = vec4<f32>(physical_color, opacity);

        let data_pos = the_coord * sizef - vec3<f32>(0.5, 0.5, 0.5);
        let world_pos = u_wobject.world_transform * vec4<f32>(data_pos, 1.0);
        let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

        var out: RenderOutput;
        out.color = out_color;
        out.coord = the_coord;
        out.depth = ndc_pos.z / ndc_pos.w;
        out.local_normal = vec3<f32>(0.0);
        return out;
    }

$$ elif mode == 'iso'
    fn raycast(sizef: vec3<f32>, nsteps: i32, start_coord: vec3<f32>, step_coord: vec3<f32>) -> RenderOutput {
        let nstepsf = f32(nsteps);

        // Primary loop: find the approximate location of the surface.
        let iso_threshold = u_material.threshold;
        let actual_step_coord = u_material.step_size * step_coord;
        var surface_found = false;
        var the_coord = start_coord;
        var the_value : vec4<f32>;
        for (var iter=0.0; iter<nstepsf; iter=iter+1) {
            let coord = start_coord + iter * actual_step_coord;
            let value = sample_vol(coord, sizef);
            let reff = value.r;
            if (reff > iso_threshold) {
                the_coord = coord;
                the_value = value;
                surface_found = true;
                break;
            }
        }

        if surface_found {
            // Take smaller steps back to make sure the surface was found.
            let substep_coord = -1 * u_material.substep_size * step_coord;
            let substep_start_coord = the_coord;
            let max_iter = 1 / u_material.substep_size;
            for (var iter=1.0; iter<max_iter; iter=iter+1) {
                let coord = substep_start_coord + iter * substep_coord;
                let value = sample_vol(coord, sizef);
                let reff = value.r;
                if (reff < iso_threshold){
                    break;
                }
                the_coord = coord;
                the_value = value;
            }
        }
        else {
            discard;
        }

        let color = sampled_value_to_color(the_value);
        $$ if colorspace == 'srgb'
            let physical_albedo = srgb2physical(color.rgb);
        $$ else
            let physical_albedo = color.rgb;
        $$ endif

        // Central-difference gradient, one voxel either side per axis.  The
        // offsets are in texture coordinates but the step is a whole voxel,
        // so the result is a direction in *data* space -- which is the
        // volume's own object space, and what the normal target wants.
        var local_normal : vec3<f32>;
        var positive_value : vec4<f32>;
        var negative_value : vec4<f32>;
        let gradient_coord = 1.0 / sizef;

        negative_value = sample_vol(the_coord + vec3<f32>(-gradient_coord[0],0.0,0.0), sizef);
        positive_value = sample_vol(the_coord + vec3<f32>(gradient_coord[0],0.0,0.0), sizef);
        local_normal[0] = positive_value.r - negative_value.r;

        negative_value = sample_vol(the_coord + vec3(0.0,-gradient_coord[1],0.0), sizef);
        positive_value = sample_vol(the_coord + vec3(0.0,gradient_coord[1],0.0), sizef);
        local_normal[1] = positive_value.r - negative_value.r;

        negative_value = sample_vol(the_coord + vec3(0.0,0.0,-gradient_coord[2]), sizef);
        positive_value = sample_vol(the_coord + vec3(0.0,0.0,gradient_coord[2]), sizef);
        local_normal[2] = positive_value.r - negative_value.r;

        // The surface position, in data space and then in world space.
        //
        // FIX vs pygfx: upstream builds world_pos with a 3x3 matrix here,
        // dropping the translation, so a translated iso volume writes depth
        // as if it sat at the origin and wins depth tests it should lose --
        // and lights it from the wrong place too.  The full 4x4 is what the
        // mip and minip branches already use.
        let data_pos = the_coord * sizef - vec3<f32>(0.5, 0.5, 0.5);
        let world_pos4 = u_wobject.world_transform * vec4<f32>(data_pos, 1.0);
        let world_pos = world_pos4.xyz;
        let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos4;

        // Project the normal to world space for shading.
        //
        // FIX vs pygfx: by the inverse transpose rather than the matrix.
        // For the anisotropic voxel spacing cellier renders routinely the
        // two differ, and the matrix tilts every normal.
        let inv = u_wobject.world_transform_inv;
        let normal_matrix = transpose(mat3x3<f32>(inv[0].xyz, inv[1].xyz, inv[2].xyz));
        let world_normal = normalize(normal_matrix * local_normal);

        // Project the step direction to world space for the view vector.
        let world_matrix = mat3x3<f32>(
            u_wobject.world_transform[0].xyz,
            u_wobject.world_transform[1].xyz,
            u_wobject.world_transform[2].xyz);
        let view_dir = normalize(world_matrix * (-step_coord));

        // Flip the normal toward the viewer; see pygfx/issues/#105.
        let is_front = dot(world_normal, view_dir) > 0.0;
        let reoriented_normal = select(-world_normal, world_normal, is_front);

        let physical_color = calculate_light(physical_albedo, world_pos, reoriented_normal, view_dir);
        let opacity = color.a * u_material.opacity;
        let out_color = vec4<f32>(physical_color, opacity);

        var out: RenderOutput;
        out.color = out_color;
        out.coord = the_coord;
        out.depth = ndc_pos.z / ndc_pos.w;
        out.local_normal = local_normal;
        return out;
    }

$$ else
    fn raycast(sizef: vec3<f32>, nsteps: i32, start_coord: vec3<f32>, step_coord: vec3<f32>) -> RenderOutput {
        {{ mode }}__is_not_a_valid_render_mode();
        var out: RenderOutput;
        return out;
    }
$$ endif
