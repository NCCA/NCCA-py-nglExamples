// Object shaders for the weighted blended OIT demo (WebGPU).
//
// One vertex shader and two fragment entry points:
//   fragment_colour -- plain shaded output, used for the opaque pass and
//                      for the naive / sorted alpha blend modes
//   fragment_oit    -- the OIT accumulation pass, writing to two render
//                      targets (accum RGBA16F and reveal R16F) whose blend
//                      states are set per target on the pipeline
//
// Reference: McGuire & Bavoil, "Weighted Blended Order-Independent
// Transparency", JCGT 2013. http://jcgt.org/published/0002/02/09/

struct Uniforms {
    MVP : mat4x4<f32>,
    MV : mat4x4<f32>,
    normal_matrix : mat4x4<f32>,
    colour : vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) normal : vec3<f32>,
    @location(1) view_z : f32,
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
    // view-space z, negative in front of the camera; the weight function
    // only cares about the magnitude
    output.view_z = (uniforms.MV * vec4<f32>(position, 1.0)).z;
    return output;
}

fn shade(normal : vec3<f32>) -> vec3<f32> {
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.8));
    // double-sided N.L so the backs of the panels shade too
    let ndotl = abs(dot(normalize(normal), light_dir));
    return uniforms.colour.rgb * (0.25 + 0.75 * ndotl);
}

@fragment
fn fragment_colour(input : VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(shade(input.normal), uniforms.colour.a);
}

// depth-based weight: near fragments dominate the weighted average just as
// they would dominate a correctly sorted OVER composite
fn weight(view_z : f32, alpha : f32) -> f32 {
    let z = abs(view_z);
    let w = 10.0 / (1e-5 + pow(z / 5.0, 2.0) + pow(z / 200.0, 6.0));
    return alpha * clamp(w, 1e-2, 3e3);
}

struct OITOutput {
    // blended with (one, one): a running SUM
    @location(0) accum : vec4<f32>,
    // blended with (zero, one-minus-src): a running PRODUCT of (1 - alpha)
    @location(1) reveal : f32,
};

@fragment
fn fragment_oit(input : VertexOutput) -> OITOutput {
    let a = uniforms.colour.a;
    let w = weight(input.view_z, a);
    var output : OITOutput;
    output.accum = vec4<f32>(shade(input.normal) * a, a) * w;
    output.reveal = a;
    return output;
}
