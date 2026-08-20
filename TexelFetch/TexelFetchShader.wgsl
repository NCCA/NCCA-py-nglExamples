struct Uniforms {
    mvp: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> y_buf: array<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@location(0) xz: vec2<f32>, @builtin(vertex_index) vidx: u32) -> VertexOutput {
    var out: VertexOutput;
    let ypos = y_buf[vidx];
    out.position = uniforms.mvp * vec4<f32>(xz.x, ypos, xz.y, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
