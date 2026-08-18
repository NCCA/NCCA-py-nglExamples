struct Camera {
    view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
    eye_position: vec4<f32>,
    light_position: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;

@group(1) @binding(0) var<storage, read> bones: array<mat4x4<f32>>;

@group(2) @binding(0) var t_diffuse: texture_2d<f32>;
@group(2) @binding(1) var s_diffuse: sampler;

const LIGHT_AMBIENT: vec3<f32> = vec3<f32>(0.2, 0.2, 0.2);
const LIGHT_DIFFUSE: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
const LIGHT_SPECULAR: vec3<f32> = vec3<f32>(0.8, 0.8, 0.8);
const MATERIAL_AMBIENT: vec3<f32> = vec3<f32>(0.2, 0.2, 0.2);
const MATERIAL_SPECULAR: vec3<f32> = vec3<f32>(0.4, 0.4, 0.4);
const MATERIAL_SHININESS: f32 = 32.0;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) bone_ids: vec4<f32>,
    @location(4) bone_weights: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    var skin: mat4x4<f32> = bones[u32(input.bone_ids.x)] * input.bone_weights.x;
    skin += bones[u32(input.bone_ids.y)] * input.bone_weights.y;
    skin += bones[u32(input.bone_ids.z)] * input.bone_weights.z;
    skin += bones[u32(input.bone_ids.w)] * input.bone_weights.w;

    let skinned_position = skin * vec4<f32>(input.position, 1.0);
    let skinned_normal = skin * vec4<f32>(input.normal, 0.0);
    let world_position = camera.model * skinned_position;

    var output: VertexOutput;
    output.position = camera.view_projection * world_position;
    output.world_position = world_position.xyz;
    output.normal = normalize((camera.model * skinned_normal).xyz);
    output.uv = input.uv;
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // mesh.py flips V once for OpenGL's bottom-left texture origin; WebGPU's
    // origin is top-left, so this flips it back rather than giving the
    // shared loader a second UV buffer.
    let flipped_uv = vec2<f32>(input.uv.x, 1.0 - input.uv.y);
    let tex_colour = textureSample(t_diffuse, s_diffuse, flipped_uv);

    let normal = normalize(input.normal);
    let eye_direction = normalize(camera.eye_position.xyz - input.world_position);
    let light_direction = normalize(camera.light_position.xyz - input.world_position);
    let half_vector = normalize(eye_direction + light_direction);

    let lambert = max(dot(normal, light_direction), 0.0);
    var colour = MATERIAL_AMBIENT * LIGHT_AMBIENT;
    if lambert > 0.0 {
        colour += tex_colour.rgb * LIGHT_DIFFUSE * lambert;
        let specular_term = pow(max(dot(normal, half_vector), 0.0), MATERIAL_SHININESS);
        colour += MATERIAL_SPECULAR * LIGHT_SPECULAR * specular_term;
    }
    return vec4<f32>(colour, tex_colour.a);
}
