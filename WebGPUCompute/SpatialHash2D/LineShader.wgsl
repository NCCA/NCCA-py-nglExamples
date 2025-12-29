// LineShader.wgsl
struct Uniforms {
    projection_matrix: mat4x4<f32>,
    size: f32,
}

@binding(0) @group(0) var<uniform> uniforms: Uniforms;

@vertex
fn vertex_main(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
    return uniforms.projection_matrix * vec4<f32>(pos, 0.0, 1.0);
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0); // Grey color for grid lines
}
