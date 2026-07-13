// Flat-coloured checker grid. Uses its own bind group / pipeline (WebGPU
// pipelines are immutable once built, unlike a GL program which can be
// re-pointed at a different binding point at runtime) but reads the exact
// same SceneBlock GPUBuffer that SceneShader.wgsl's bind group references
// -- one write_buffer() call updates the VP matrix seen by both pipelines.

struct SceneBlock {
    VP: mat4x4<f32>,
    lightPos: vec4<f32>,
    lightColour: vec4<f32>,
};

struct ObjectUniforms {
    M: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> scene: SceneBlock;
@group(0) @binding(1) var<uniform> object: ObjectUniforms;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) colour: vec3<f32>,
};

@vertex
fn vertex_main(
    @location(0) inPos: vec3<f32>,
    @location(1) inColour: vec3<f32>,
) -> VertexOut {
    var out: VertexOut;
    out.position = scene.VP * object.M * vec4<f32>(inPos, 1.0);
    out.colour = inColour;
    return out;
}

@fragment
fn fragment_main(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.colour, 1.0);
}
