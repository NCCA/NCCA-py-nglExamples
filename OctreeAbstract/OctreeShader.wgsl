struct Uniforms {
  mvp: mat4x4<f32>,
  normal_matrix: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) colour: vec3<f32>,
};

@vertex
fn particle_vertex(
  @location(0) sphere_position: vec3<f32>,
  @location(1) sphere_normal: vec3<f32>,
  @location(2) particle_position_radius: vec4<f32>,
  @location(3) particle_colour: vec4<f32>,
) -> VertexOutput {
  let position = sphere_position * particle_position_radius.w + particle_position_radius.xyz;
  var output: VertexOutput;
  output.position = uniforms.mvp * vec4<f32>(position, 1.0);
  output.normal = normalize((uniforms.normal_matrix * vec4<f32>(sphere_normal, 0.0)).xyz);
  output.colour = particle_colour.rgb;
  return output;
}

@vertex
fn line_vertex(@location(0) position: vec3<f32>) -> VertexOutput {
  var output: VertexOutput;
  output.position = uniforms.mvp * vec4<f32>(position, 1.0);
  output.normal = vec3<f32>(0.0, 1.0, 0.0);
  output.colour = vec3<f32>(0.8, 0.8, 0.8);
  return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let light_direction = normalize(vec3<f32>(0.4, 1.0, 0.7));
  let diffuse = max(dot(normalize(input.normal), light_direction), 0.0);
  return vec4<f32>(input.colour * (0.2 + 0.8 * diffuse), 1.0);
}
