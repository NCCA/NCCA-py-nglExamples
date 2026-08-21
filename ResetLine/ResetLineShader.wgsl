struct Uniforms {
  mvp: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) colour: vec3<f32>,
};

@vertex
fn vertex_main(
  @location(0) position: vec3<f32>,
  @location(1) colour: vec3<f32>,
) -> VertexOutput {
  var output: VertexOutput;
  output.position = uniforms.mvp * vec4<f32>(position, 1.0);
  output.colour = colour;
  return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.colour, 1.0);
}
