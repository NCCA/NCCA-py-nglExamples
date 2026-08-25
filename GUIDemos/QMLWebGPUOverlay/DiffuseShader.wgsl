// GUIDemos/QMLWebGPUOverlay/DiffuseShader.wgsl
// A minimal single-light diffuse shader, the WebGPU equivalent of PyNGL's
// DefaultShader.DIFFUSE used by the OpenGL QMLOverlayApp/QMLFloatingWidgets
// demos. Lighting is evaluated in eye space so the fixed lightPos tracks the
// camera the same way the GL demo's does.

struct Transform {
    MVP : mat4x4<f32>,
    MV : mat4x4<f32>,
    normalMatrix : mat4x4<f32>,
};

struct Lighting {
    Colour : vec4<f32>,
    lightPos : vec4<f32>,
    lightDiffuse : vec4<f32>,
};

@group(0) @binding(0) var<uniform> transform : Transform;
@group(0) @binding(1) var<uniform> lighting : Lighting;

struct VSOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) normalEye : vec3<f32>,
    @location(1) fragEye : vec3<f32>,
};

@vertex
fn vertex_main(
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>,
) -> VSOut {
    var out : VSOut;
    out.pos = transform.MVP * vec4<f32>(position, 1.0);
    out.fragEye = (transform.MV * vec4<f32>(position, 1.0)).xyz;
    out.normalEye = (transform.normalMatrix * vec4<f32>(normal, 0.0)).xyz;
    return out;
}

@fragment
fn fragment_main(in : VSOut) -> @location(0) vec4<f32> {
    let N = normalize(in.normalEye);
    let L = normalize(lighting.lightPos.xyz - in.fragEye);
    let diff = max(dot(N, L), 0.0);
    let ambient = 0.1;
    let colour = lighting.Colour.rgb * lighting.lightDiffuse.rgb * (diff + ambient);
    return vec4<f32>(colour, 1.0);
}
