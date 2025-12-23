@group(0) @binding(0) var<uniform> uniforms : Uniforms;
struct Uniforms
{
    projection_matrix : mat4x4<f32>,
    size: f32,
};

struct VertexIn {
    @location(0) position: vec2<f32>,
    @location(1) colour: vec3<f32>,
};

// We now need to pass uv to the fragment shader
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) fragColour: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vertex_main(input: VertexIn, @builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var output: VertexOut;
    let quad_offsets = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0), // bottom-left
        vec2<f32>(1.0, -1.0),  // bottom-right
        vec2<f32>(-1.0, 1.0),   // top-left
        vec2<f32>(1.0, 1.0)    // top-right
    );

    let offset = quad_offsets[vertex_index];
    let pos = vec4<f32>(input.position.xy + offset * uniforms.size, 0.0, 1.0);

    output.position = uniforms.projection_matrix * pos;
    output.fragColour = input.colour;
    // convert offset from -1 -> 1 to 0 -> 1 for uv
    output.uv = offset * 0.5 + 0.5;

    return output;
}

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4<f32>
{
    let center = vec2<f32>(0.5, 0.5); // Center of the quad in UV space
    let dist = distance(fragData.uv, center); // Distance from center
    let radius = 0.5; // Circle radius (quad is 1.0 in UV space)

    if (dist > radius)
    {
        discard; // Remove pixels outside the circle
    }

    return vec4<f32>(fragData.fragColour, 1.0); // Simple color output
}
