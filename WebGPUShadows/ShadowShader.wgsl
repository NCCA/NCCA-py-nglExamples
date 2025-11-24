@group(0) @binding(0) var<storage, read> mesh_uniforms: array<VertexUniforms>;

struct VertexUniforms
{
    MVP : mat4x4<f32>,
    model_view : mat4x4<f32>,
    normal_matrix : mat4x4<f32>,
    colour : vec4<f32>,
    lightMVP : mat4x4<f32>
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
};


@vertex
fn vertex_main(input : VertexInput,@builtin(instance_index) instanceIdx: u32) -> VertexOutput
{
    var output : VertexOutput;
    let uniforms = mesh_uniforms[instanceIdx];
    output.position = uniforms.lightMVP * vec4<f32>(input.position, 1.0);
    return output;
}

@fragment
fn fragment_main()
{
    // We don't need to do anything here as we are only writing to the depth buffer
}
