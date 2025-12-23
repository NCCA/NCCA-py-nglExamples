@group(0) @binding(0) var<uniform> uniforms : Uniforms;
struct Uniforms
{
    projection_matrix : mat4x4<f32>,
};

struct VertexIn {
    @location(0) position: vec2<f32>,
    @location(1) colour: vec3<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) fragColour: vec3<f32>,
};

@vertex
fn vertex_main(input: VertexIn) -> VertexOut {
    var output: VertexOut;
    output.position = uniforms.projection_matrix * vec4<f32>(input.position, 0.0, 1.0);
    output.fragColour = input.colour;
    return output;
}
@fragment
fn fragment_main(@location(0) fragColour: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(fragColour, 1.0); // Simple colour output
}
