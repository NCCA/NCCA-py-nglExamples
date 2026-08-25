// Same field of cubes as the OpenGL demo (main.py), read from a per-instance
// vertex buffer with step_mode "instance". WebGPU has no equivalent of
// glVertexAttribDivisor -- the buffer layout itself declares "one record per
// instance, not per vertex" (see InstancingWebGPU.py::_create_pipeline).
// There is no "instanced" bool here: both draw modes share this exact
// pipeline, they only differ in how many draw() calls the CPU issues (see
// the README for the naive-mode first_instance discussion).

struct Uniforms {
    MVP : mat4x4<f32>,
    normal_matrix : mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) normal : vec3<f32>,
    @location(1) colour : vec4<f32>,
};

@vertex
fn vertex_main(
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(3) inst_offset_scale : vec4<f32>,
    @location(4) inst_colour : vec4<f32>,
) -> VertexOutput {
    var output : VertexOutput;
    let world_pos = position * inst_offset_scale.w + inst_offset_scale.xyz;
    output.position = uniforms.MVP * vec4<f32>(world_pos, 1.0);
    output.normal = normalize((uniforms.normal_matrix * vec4<f32>(normal, 0.0)).xyz);
    output.colour = inst_colour;
    return output;
}

@fragment
fn fragment_main(input : VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.4, 1.0, 0.6));
    let ndotl = max(dot(normalize(input.normal), light_dir), 0.0);
    let shaded = input.colour.rgb * (0.35 + 0.65 * ndotl);
    return vec4<f32>(shaded, input.colour.a);
}
