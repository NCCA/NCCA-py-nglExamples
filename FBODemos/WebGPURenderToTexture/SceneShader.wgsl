// Second pass shader: draw geometry textured with the teapot render target.

struct Uniforms {
    mvp : mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var t_diffuse : texture_2d<f32>;
@group(0) @binding(2) var s_diffuse : sampler;

struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) uv : vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vertex_main(model : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    out.position = uniforms.mvp * vec4<f32>(model.position, 1.0);
    out.uv = model.uv;
    return out;
}

@fragment
fn fragment_main(in : VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.uv);
}
