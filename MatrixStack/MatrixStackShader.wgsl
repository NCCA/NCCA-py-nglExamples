struct InstanceUniforms {
    mvp: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    colour: vec4<f32>,
    light_pos: vec4<f32>,
    light_diffuse: vec4<f32>,
};
struct Uniforms {
    instances: array<InstanceUniforms, 130>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) colour: vec4<f32>,
    @location(2) light_pos: vec3<f32>,
    @location(3) light_diffuse: vec3<f32>,
};

@vertex
fn vertex_main(
    @location(0) in_vert: vec3<f32>,
    @location(1) in_normal: vec3<f32>,
    @location(2) in_uv: vec2<f32>,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let instance = u.instances[instance_index];
    out.position = instance.mvp * vec4<f32>(in_vert, 1.0);
    out.world_normal = normalize(
        (instance.normal_matrix * vec4<f32>(in_normal, 0.0)).xyz
    );
    out.colour = instance.colour;
    out.light_pos = instance.light_pos.xyz;
    out.light_diffuse = instance.light_diffuse.rgb;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let l = normalize(in.light_pos);
    let diffuse = max(dot(n, l), 0.0);
    let colour = in.colour.rgb * in.light_diffuse * diffuse;
    return vec4<f32>(colour, in.colour.a);
}
