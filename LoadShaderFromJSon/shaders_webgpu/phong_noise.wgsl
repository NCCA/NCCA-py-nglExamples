// Ported from shaders/PhongVertex.glsl + shaders/PhongFragment.glsl.
//
// The OpenGL original computes its per-fragment direction vectors
// (lightDir/halfVector/vPosition) in eye space, from a separate MV
// uniform. This repo's WebGPU demos do their Phong lighting in world
// space instead (see ShadedGrid/PhongGridShader.wgsl), so there's no MV
// here -- world_position (M * vertex) stands in for eye-space position,
// and light.position/transform.viewer_pos are read as world-space points
// directly. Everything else -- which vectors get computed per vertex and
// interpolated, the pointLight() lambert/specular split, and the
// six-octave noise fractal sum in the fragment stage -- is unchanged.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) fragment_normal: vec3<f32>,
    @location(1) uv: vec3<f32>,
    @location(2) light_dir: vec3<f32>,
    @location(3) half_vector: vec3<f32>,
    @location(4) eye_direction: vec3<f32>,
    @location(5) v_position: vec3<f32>,
};

@vertex
fn vertex_main(
    @location(0) in_vert: vec3<f32>,
    @location(1) in_normal: vec3<f32>,
    @location(2) in_uv: vec2<f32>,
) -> VertexOutput {
    var output: VertexOutput;

    output.fragment_normal = normalize((transform.normal_matrix * vec4<f32>(in_normal, 0.0)).xyz);
    output.position = transform.mvp * vec4<f32>(in_vert, 1.0);

    let world_position = (transform.m * vec4<f32>(in_vert, 1.0)).xyz;
    output.v_position = world_position;
    output.eye_direction = normalize(transform.viewer_pos - world_position);

    let light_dir = normalize(light.position - world_position);
    output.light_dir = light_dir;
    output.half_vector = normalize(output.eye_direction + light_dir);

    output.uv = vec3<f32>(in_uv / params.repeat, params.repeat);

    return output;
}

// @brief a function to compute point light values
fn pointLight(n: vec3<f32>, light_dir: vec3<f32>, half_vector: vec3<f32>) -> vec4<f32> {
    let L = normalize(light_dir);
    let lambertTerm = dot(n, L);
    var diffuse = vec4<f32>(0.0);
    var ambient = vec4<f32>(0.0);
    var specular = vec4<f32>(0.0);
    if (lambertTerm > 0.0) {
        diffuse = material.diffuse * light.diffuse * lambertTerm;
        ambient = material.ambient * light.ambient;
        let halfV = normalize(half_vector);
        let ndothv = max(dot(n, halfV), 0.0);
        specular = material.specular * light.specular * pow(ndothv, material.shininess);
    }
    return ambient + diffuse + specular;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(input.fragment_normal);
    let uv = input.uv;
    let time = params.time;

    // Perturb the texcoords with three components of noise
    let uvw = uv + 0.1 * vec3<f32>(
        snoise(uv + vec3<f32>(0.0, 0.0, time)),
        snoise(uv + vec3<f32>(43.0, 17.0, time)),
        snoise(uv + vec3<f32>(-17.0, -43.0, time))
    );
    // Six components of noise in a fractal sum
    var n_val = snoise(uvw - vec3<f32>(0.0, 0.0, time));
    n_val = n_val + 0.5 * snoise(uvw * 2.0 - vec3<f32>(0.0, 0.0, time * 1.4));
    n_val = n_val + 0.25 * snoise(uvw * 4.0 - vec3<f32>(0.0, 0.0, time * 2.0));
    n_val = n_val + 0.125 * snoise(uvw * 8.0 - vec3<f32>(0.0, 0.0, time * 2.8));
    n_val = n_val + 0.0625 * snoise(uvw * 16.0 - vec3<f32>(0.0, 0.0, time * 4.0));
    n_val = n_val + 0.03125 * snoise(uvw * 32.0 - vec3<f32>(0.0, 0.0, time * 5.6));
    n_val = n_val * 0.7;

    // A "hot" colormap -- cheesy but effective
    let lit = pointLight(n, input.light_dir, input.half_vector);
    var colour = lit * vec4<f32>(vec3<f32>(1.0, 0.5, 0.0) + vec3<f32>(n_val, n_val, n_val), 1.0);
    colour.a = 1.0;
    return colour;
}
