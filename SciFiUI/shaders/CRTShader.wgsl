struct Uniforms {
    i_time: f32,
    effects_on: f32,
    _pad0: vec2<f32>,
    i_resolution: vec2<f32>,
    _pad1: vec2<f32>,
    phosphor: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var screen_tex: texture_2d<f32>;
@group(0) @binding(2) var screen_sampler: sampler;

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

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var uv = in.tex_coord;

    if uniforms.effects_on > 0.5 {
        var c = uv * 2.0 - vec2<f32>(1.0, 1.0);
        let r2 = dot(c, c);
        c = c * (1.0 + 0.035 * r2 + 0.015 * r2 * r2);
        uv = c * 0.5 + vec2<f32>(0.5, 0.5);
        if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
    }

    let px = vec2<f32>(1.0, 1.0) / uniforms.i_resolution;
    var col = textureSample(screen_tex, screen_sampler, uv).rgb;
    let glow = textureSample(screen_tex, screen_sampler, uv + vec2<f32>(px.x * 2.0, 0.0)).rgb
             + textureSample(screen_tex, screen_sampler, uv - vec2<f32>(px.x * 2.0, 0.0)).rgb
             + textureSample(screen_tex, screen_sampler, uv + vec2<f32>(0.0, px.y * 2.0)).rgb
             + textureSample(screen_tex, screen_sampler, uv - vec2<f32>(0.0, px.y * 2.0)).rgb;
    col = col + glow * 0.22;

    let lum = dot(col, vec3<f32>(0.299, 0.587, 0.114));
    var tinted = uniforms.phosphor.rgb * lum;

    if uniforms.effects_on > 0.5 {
        tinted = tinted * (0.80 + 0.20 * sin(uv.y * uniforms.i_resolution.y * 3.14159));
        tinted = tinted * (0.95 + 0.05 * sin(uv.x * uniforms.i_resolution.x * 3.14159));
        let bar = exp(-45.0 * pow(fract(uv.y - uniforms.i_time * 0.04) - 0.5, 2.0));
        tinted = tinted * (1.0 + 0.06 * bar);
        tinted = tinted + uniforms.phosphor.rgb * 0.035 * hash(uv * uniforms.i_resolution + vec2<f32>(uniforms.i_time * 120.0, uniforms.i_time * 120.0));
        tinted = tinted * (0.97 + 0.03 * sin(uniforms.i_time * 47.0));
    }

    let v = uv * (vec2<f32>(1.0, 1.0) - uv);
    tinted = tinted * pow(v.x * v.y * 18.0, 0.28);

    return vec4<f32>(tinted, 1.0);
}
