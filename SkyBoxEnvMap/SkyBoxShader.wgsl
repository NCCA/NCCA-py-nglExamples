// Skybox: unit cube sampled with a direction (its own local position),
// drawn with the *rotation-only* view matrix so it never translates with
// the camera. view_rot is built on the CPU by zeroing the translation row
// of the view matrix (row 3 -- this codebase's row-vector convention).
struct SkyUniforms {
    view_rot : mat4x4<f32>,
    proj : mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> u : SkyUniforms;
@group(0) @binding(1) var skyTexture : texture_cube<f32>;
@group(0) @binding(2) var skySampler : sampler;

struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) direction : vec3<f32>,
};

@vertex
fn vertex_main(@location(0) position : vec3<f32>) -> VertexOutput {
    var out : VertexOutput;
    out.direction = position;
    let clip = u.proj * u.view_rot * vec4<f32>(position, 1.0);
    // push the skybox to the far plane (clip.z == clip.w -> NDC z == 1)
    // so it never wins a depth test against real geometry, whatever the
    // draw order -- the WGSL equivalent of the GLSL ".xyww" trick.
    out.position = vec4<f32>(clip.x, clip.y, clip.w, clip.w);
    return out;
}

@fragment
fn fragment_main(input : VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(skyTexture, skySampler, input.direction);
}
