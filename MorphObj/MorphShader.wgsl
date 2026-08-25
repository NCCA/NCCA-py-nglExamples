struct Uniforms {
  mvp: mat4x4<f32>,
  model_view: mat4x4<f32>,
  normal_matrix: mat4x4<f32>,
  light_position: vec4<f32>,
  weights: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) eye_position: vec3<f32>,
  @location(1) eye_normal: vec3<f32>,
};

@vertex
fn vertex_main(
  @location(0) base_position: vec3<f32>,
  @location(1) base_normal: vec3<f32>,
  @location(2) pose_position_one: vec3<f32>,
  @location(3) pose_normal_one: vec3<f32>,
  @location(4) pose_position_two: vec3<f32>,
  @location(5) pose_normal_two: vec3<f32>,
) -> VertexOutput {
  let final_position = base_position
    + uniforms.weights.x * pose_position_one
    + uniforms.weights.y * pose_position_two;
  let final_normal = base_normal
    + uniforms.weights.x * pose_normal_one
    + uniforms.weights.y * pose_normal_two;
  var output: VertexOutput;
  output.position = uniforms.mvp * vec4<f32>(final_position, 1.0);
  output.eye_position = (uniforms.model_view * vec4<f32>(final_position, 1.0)).xyz;
  output.eye_normal = normalize((uniforms.normal_matrix * vec4<f32>(final_normal, 0.0)).xyz);
  return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let normal = normalize(input.eye_normal);
  let light_direction = normalize(uniforms.light_position.xyz - input.eye_position);
  let view_direction = normalize(-input.eye_position);
  let half_vector = normalize(light_direction + view_direction);
  let diffuse = max(dot(normal, light_direction), 0.0);
  var specular = 0.0;
  if diffuse > 0.0 {
    specular = pow(max(dot(normal, half_vector), 0.0), 64.0);
  }
  let colour = vec3<f32>(0.08) + vec3<f32>(0.72) * diffuse + vec3<f32>(0.35) * specular;
  return vec4<f32>(colour, 1.0);
}
