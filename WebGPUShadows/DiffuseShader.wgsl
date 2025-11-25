@group(0) @binding(0) var<storage, read> mesh_uniforms: array<VertexUniforms>;
@group(0) @binding(1) var<storage, read> light_uniforms : array<LightUniforms>;
@group(0) @binding(2) var shadow_map: texture_depth_2d;
@group(0) @binding(3) var shadow_sampler: sampler_comparison;


struct VertexUniforms
{
    MVP : mat4x4<f32>,
    model_view : mat4x4<f32>,
    normal_matrix : mat4x4<f32>,
    colour : vec4<f32>,
    lightMVP : mat4x4<f32>
};


struct LightUniforms
{
    light_pos : vec4<f32>,
    light_diffuse : vec4<f32>
};


struct VertexInput
{
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>
};

struct VertexOutput
{
    @builtin(position) position : vec4<f32>,
    @location(0) normal : vec3<f32>,
    @location(1) uv : vec2<f32>,
    @location(2) frag_pos : vec3<f32>,
    @location(3) colour : vec4<f32>,
    @location(4) shadow_pos : vec4<f32>
};

fn extract_mat3x3(mat: mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        mat[0].xyz,
        mat[1].xyz,
        mat[2].xyz
    );
}

@vertex
fn vertex_main(input : VertexInput,@builtin(instance_index) instanceIdx: u32) -> VertexOutput
{
    var output : VertexOutput;

    let uniforms = mesh_uniforms[instanceIdx];

    output.position = uniforms.MVP * vec4<f32>(input.position, 1.0);
    output.normal = extract_mat3x3(uniforms.normal_matrix) * input.normal;
    output.uv = input.uv;
    output.frag_pos = (uniforms.model_view * vec4<f32>(input.position, 1.0)).xyz;
    output.colour = uniforms.colour;
    output.shadow_pos = uniforms.lightMVP * vec4<f32>(input.position, 1.0);

    return output;
}

struct FragmentInput
{
    @location(0) normal : vec3<f32>,
    @location(1) uv : vec2<f32>,
    @location(2) frag_pos : vec3<f32>,
    @location(3) colour : vec4<f32>,
    @location(4) shadow_pos : vec4<f32>

};

struct FragmentOutput
{
    @location(0) colour : vec4<f32>
};

@fragment
fn fragment_main(input : FragmentInput) -> FragmentOutput
{
    var output : FragmentOutput;
    let num_lights = arrayLength(&light_uniforms);

    // Perspective divide
    let shadow_coords = input.shadow_pos.xyz / input.shadow_pos.w;
    // transform from clip space to texture space
    let shadow_uv = shadow_coords.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    let shadow_depth = shadow_coords.z;
    let light0 = light_uniforms[0];

    let L0 = normalize(light0.light_pos.xyz - input.frag_pos);

    // Sample the shadow map
    var shadow_factor = 1.0;
    // Only sample if the fragment is within the shadow map's frustum
    if (shadow_uv.x >= 0.0 && shadow_uv.x <= 1.0 && shadow_uv.y >= 0.0 && shadow_uv.y <= 1.0 && shadow_depth <=1.0)
    {
        let bias= max(0.02 * (1.0 - dot(normalize(input.normal), L0)), 0.002);
        // use a small bias to avoid shadow acne
        shadow_factor = textureSampleCompare(shadow_map, shadow_sampler, shadow_uv, shadow_depth - bias);
    }


    // Start with a simple ambient light term
    var final_colour : vec3<f32> = input.colour.rgb * 0.01;

    // We are only shadowing the first light
    let distance0 = length(light0.light_pos.xyz - input.frag_pos);
    let attenuation0 = 1.0 / (distance0 * distance0 + 1.0);
    let radiance0 = light0.light_diffuse.rgb * attenuation0;
    let diffuse0 = max(dot(normalize(input.normal), L0), 0.0);
    final_colour += input.colour.rgb * radiance0 * diffuse0 * shadow_factor;


    for (var i: u32 = 1u; i < num_lights; i = i + 1u)
    {
        let light = light_uniforms[i];
        let L = normalize(light.light_pos.xyz - input.frag_pos);
        let distance = length(light.light_pos.xyz - input.frag_pos);
        let attenuation = 1.0 / (distance * distance + 1.0);
        let radiance = light.light_diffuse.rgb * attenuation;
        let diffuse = max(dot(normalize(input.normal), L), 0.0);
        final_colour += input.colour.rgb * radiance * diffuse;
    }
    output.colour = vec4<f32>(final_colour, input.colour.a);
    return output;
}
