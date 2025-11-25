@group(0) @binding(0) var<storage, read> mesh_uniforms: array<VertexUniforms>;
@group(0) @binding(1) var<storage, read> light_uniforms : array<LightUniforms>;

@group(0) @binding(2) var<uniform> global_uniforms : GlobalUniforms;

struct GlobalUniforms
{
    view : mat4x4<f32>,
    projection : mat4x4<f32>,
    camera_pos : vec4<f32>
};


struct VertexUniforms
{

    model : mat4x4<f32>,
    normal_matrix : mat4x4<f32>,
    colour : vec4<f32>
};


struct LightUniforms
{
    light_pos : vec4<f32>,
    light_diffuse : vec4<f32>,
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
    @location(3) colour : vec4<f32>
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

    let MVP = global_uniforms.projection * global_uniforms.view * uniforms.model;


    output.position = MVP * vec4<f32>(input.position, 1.0);
    output.normal = extract_mat3x3(uniforms.normal_matrix) * input.normal;
    output.uv = input.uv;
    output.frag_pos = (uniforms.model * vec4<f32>(input.position, 1.0)).xyz;
    output.colour = uniforms.colour;

    return output;
}

struct FragmentInput
{
    @location(0) normal : vec3<f32>,
    @location(1) uv : vec2<f32>,
    @location(2) frag_pos : vec3<f32>,
    @location(3) colour : vec4<f32>

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

    // Start with a simple ambient light term
    var final_colour : vec3<f32> = input.colour.rgb * 0.01;

    for (var i: u32 = 0u; i < num_lights; i = i + 1u)
    {
        let light = light_uniforms[i];
        let L = normalize(light.light_pos.xyz - input.frag_pos);
        let V = normalize(global_uniforms.camera_pos.xyz - input.frag_pos);
        let H = normalize(L + V);
        let diffuse = max(dot(normalize(input.normal), H), 0.0);
        final_colour += input.colour.rgb * light.light_diffuse.rgb * diffuse;
    }
    output.colour = vec4<f32>(final_colour, input.colour.a);
    return output;
}
