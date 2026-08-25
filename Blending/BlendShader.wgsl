// Simple double-sided lambert shader used for every object in the blending
// demo. Note there is nothing "transparent" in here at all -- the fragment
// shader just writes an alpha value. Whether that alpha does anything is
// decided by the blend state baked into the render pipeline, which is the
// key difference from OpenGL where glBlendFunc is dynamic state.

struct Uniforms {
    MVP : mat4x4<f32>,
    normal_matrix : mat4x4<f32>,
    colour : vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) normal : vec3<f32>,
};

@vertex
fn vertex_main(
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>,
) -> VertexOutput {
    var output : VertexOutput;
    output.position = uniforms.MVP * vec4<f32>(position, 1.0);
    output.normal = normalize((uniforms.normal_matrix * vec4<f32>(normal, 0.0)).xyz);
    return output;
}

@fragment
fn fragment_main(input : VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.8));
    // double-sided N.L so the backs of the panels shade too
    let ndotl = abs(dot(normalize(input.normal), light_dir));
    let shaded = uniforms.colour.rgb * (0.25 + 0.75 * ndotl);
    return vec4<f32>(shaded, uniforms.colour.a);
}
