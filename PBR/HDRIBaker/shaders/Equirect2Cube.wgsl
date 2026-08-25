// Bakes one face of the environment cube from the source equirectangular
// HDRI. Direct port of shaders/CubeVertex.glsl + Equirect2CubeFragment.glsl.

struct CaptureUniforms {
    projection : mat4x4<f32>,
    view : mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> capture : CaptureUniforms;
@group(1) @binding(0) var equirectangularMap : texture_2d<f32>;
@group(1) @binding(1) var equirectSampler : sampler;

struct VSOut {
    @builtin(position) position : vec4<f32>,
    @location(0) localPos : vec3<f32>,
};

@vertex
fn vertex_main(@location(0) inVert : vec3<f32>) -> VSOut {
    var out : VSOut;
    out.localPos = inVert;
    out.position = capture.projection * capture.view * vec4<f32>(inVert, 1.0);
    return out;
}

// v is negated against the usual GLSL (0.1591, 0.3183): WGSL puts v = 0 at the
// *top* texel row, where GL puts it at the bottom, so the GL sign would send
// straight up (asin(y) == +1) to the bottom of the panorama -- the ground.
const invAtan : vec2<f32> = vec2<f32>(0.1591, -0.3183);

fn sampleSphericalMap(v : vec3<f32>) -> vec2<f32> {
    var uv = vec2<f32>(atan2(v.z, v.x), asin(v.y));
    uv *= invAtan;
    uv += 0.5;
    return uv;
}

@fragment
fn fragment_main(in : VSOut) -> @location(0) vec4<f32> {
    let uv = sampleSphericalMap(normalize(in.localPos));
    let colour = textureSample(equirectangularMap, equirectSampler, uv).rgb;
    return vec4<f32>(colour, 1.0);
}
