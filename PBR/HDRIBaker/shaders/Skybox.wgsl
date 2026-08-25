// The background sky: the baked environment cube (or, for the `E` debug
// views, the irradiance/prefilter cube at a chosen lod). Direct port of
// shaders/SkyboxVertex.glsl + SkyboxFragment.glsl.

struct SkyboxUniforms {
    mvp : mat4x4<f32>,
    lod : f32,
    _pad0 : f32,
    _pad1 : f32,
    _pad2 : f32,
};
@group(0) @binding(0) var<uniform> skyboxUniforms : SkyboxUniforms;
@group(1) @binding(0) var skybox : texture_cube<f32>;
@group(1) @binding(1) var skyboxSampler : sampler;

struct VSOut {
    @builtin(position) position : vec4<f32>,
    @location(0) direction : vec3<f32>,
};

@vertex
fn vertex_main(@location(0) inVert : vec3<f32>) -> VSOut {
    var out : VSOut;
    // the cube's local position *is* the sample direction: the skybox is a
    // unit cube, no need to normalise on the CPU side -- interpolation plus
    // normalize() in the fragment shader is enough.
    out.direction = inVert;
    let pos = skyboxUniforms.mvp * vec4<f32>(inVert, 1.0);
    // force depth to the far plane (z == w after the divide) so the skybox
    // never wins a depth test against real geometry, whatever the draw order.
    out.position = pos.xyww;
    return out;
}

@fragment
fn fragment_main(in : VSOut) -> @location(0) vec4<f32> {
    var colour = textureSampleLevel(skybox, skyboxSampler, normalize(in.direction), skyboxUniforms.lod).rgb;
    colour = colour / (colour + vec3<f32>(1.0)); // Reinhard tonemap
    colour = pow(colour, vec3<f32>(1.0 / 2.2)); // gamma
    return vec4<f32>(colour, 1.0);
}
