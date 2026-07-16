// Cosine-weighted hemisphere convolution of the env cube into the
// irradiance cube. Direct port of shaders/IrradianceFragment.glsl.

struct CaptureUniforms {
    projection : mat4x4<f32>,
    view : mat4x4<f32>,
    sampleDelta : f32,
};
@group(0) @binding(0) var<uniform> capture : CaptureUniforms;
@group(1) @binding(0) var environmentMap : texture_cube<f32>;
@group(1) @binding(1) var environmentSampler : sampler;

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

const PI : f32 = 3.14159265359;

@fragment
fn fragment_main(in : VSOut) -> @location(0) vec4<f32> {
    let N = normalize(in.localPos);
    var irradiance = vec3<f32>(0.0);
    var up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(up, N));
    up = normalize(cross(N, right));

    let sampleDelta = capture.sampleDelta;
    var nrSamples = 0.0;
    var phi = 0.0;
    loop {
        if phi >= 2.0 * PI {
            break;
        }
        var theta = 0.0;
        loop {
            if theta >= 0.5 * PI {
                break;
            }
            let tangentSample = vec3<f32>(
                sin(theta) * cos(phi),
                sin(theta) * sin(phi),
                cos(theta),
            );
            let sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * N;
            irradiance += textureSampleLevel(environmentMap, environmentSampler, sampleVec, 0.0).rgb
                * cos(theta) * sin(theta);
            nrSamples += 1.0;
            theta += sampleDelta;
        }
        phi += sampleDelta;
    }
    irradiance = PI * irradiance * (1.0 / nrSamples);
    return vec4<f32>(irradiance, 1.0);
}
