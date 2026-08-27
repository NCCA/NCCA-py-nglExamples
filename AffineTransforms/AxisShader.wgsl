// Unlit vertex colour, matching the OpenGL gizmo's DefaultShader.COLOUR.
struct Uniforms {
    mvp: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) colour: vec3<f32>,
};

@vertex
fn vertex_main(
    @location(0) in_vert: vec3<f32>,
    @location(1) in_colour: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = u.mvp * vec4<f32>(in_vert, 1.0);
    out.colour = in_colour;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.colour, 1.0);
}
