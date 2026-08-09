struct Uniforms {
    mvp: mat4x4<f32>,
    colour: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vertex_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.mvp * vec4<f32>(in.position, 1.0);
    return out;
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    return uniforms.colour;
}
