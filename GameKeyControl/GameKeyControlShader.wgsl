struct Uniforms {
    mvp: mat4x4<f32>,
    colour: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vertex_main(
    @location(0) in_vert: vec3<f32>,
    @location(1) in_normal: vec3<f32>,
    @location(2) in_uv: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = u.mvp * vec4<f32>(in_vert, 1.0);
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return u.colour;
}
