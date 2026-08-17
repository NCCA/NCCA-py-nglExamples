struct Camera {
    view_projection: mat4x4<f32>,
    light_position: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;

struct Instance {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    colour: vec4<f32>,
}

@group(1) @binding(0) var<storage, read> instances: array<Instance>;

struct MeshInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct MeshOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) colour: vec4<f32>,
}

@vertex
fn mesh_vertex(input: MeshInput, @builtin(instance_index) index: u32) -> MeshOutput {
    let instance = instances[index];
    let world_position = instance.model * vec4<f32>(input.position, 1.0);
    var output: MeshOutput;
    output.position = camera.view_projection * world_position;
    output.world_position = world_position.xyz;
    output.normal = normalize((instance.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz);
    output.colour = instance.colour;
    return output;
}

@fragment
fn mesh_fragment(input: MeshOutput) -> @location(0) vec4<f32> {
    let light_direction = normalize(camera.light_position.xyz - input.world_position);
    let diffuse = max(dot(normalize(input.normal), light_direction), 0.0);
    let light = 0.25 + 0.75 * diffuse;
    return vec4<f32>(input.colour.rgb * light, input.colour.a);
}

struct LineInput {
    @location(0) position: vec3<f32>,
    @location(1) colour: vec4<f32>,
}

struct LineOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) colour: vec4<f32>,
}

@vertex
fn line_vertex(input: LineInput) -> LineOutput {
    var output: LineOutput;
    output.position = camera.view_projection * vec4<f32>(input.position, 1.0);
    output.colour = input.colour;
    return output;
}

@fragment
fn line_fragment(input: LineOutput) -> @location(0) vec4<f32> {
    return input.colour;
}
