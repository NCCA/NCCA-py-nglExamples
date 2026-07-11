// Final composite pass: a full-screen triangle generated from the vertex
// index (no vertex buffer). In naive / sorted modes it just copies the
// opaque buffer (the panels were blended straight into it); in OIT mode it
// resolves the two accumulation targets:
//
//     transparent = accum.rgb / accum.a           (the weighted average)
//     final = opaque * reveal + transparent * (1 - reveal)

struct Params {
    use_oit : u32,
    pad0 : u32,
    pad1 : u32,
    pad2 : u32,
};

@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var opaque_tex : texture_2d<f32>;
@group(0) @binding(2) var accum_tex : texture_2d<f32>;
@group(0) @binding(3) var reveal_tex : texture_2d<f32>;
@group(0) @binding(4) var<uniform> params : Params;

struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
    var output : VertexOutput;
    let corner = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    output.position = vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
    // flip v: clip-space y is up, texture v is down
    output.uv = vec2<f32>(corner.x, 1.0 - corner.y);
    return output;
}

@fragment
fn fragment_main(input : VertexOutput) -> @location(0) vec4<f32> {
    let opaque = textureSample(opaque_tex, samp, input.uv).rgb;
    let accum = textureSample(accum_tex, samp, input.uv);
    let reveal = textureSample(reveal_tex, samp, input.uv).r;
    if (params.use_oit == 0u) {
        return vec4<f32>(opaque, 1.0);
    }
    let transparent = accum.rgb / max(accum.a, 1e-5);
    return vec4<f32>(opaque * reveal + transparent * (1.0 - reveal), 1.0);
}
