@group(0) @binding(0) var text_tex: texture_2d<f32>;
@group(0) @binding(1) var text_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = -1.0 + f32((vertex_index & 1u) << 2u);
    let y = -1.0 + f32((vertex_index & 2u) << 1u);
    out.tex_coord = vec2<f32>((x + 1.0) * 0.5, (y + 1.0) * 0.5);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(text_tex, text_sampler, in.tex_coord);
}
