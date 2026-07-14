// Instanced boid rendering: the hardcoded arrow shape from
// VertexArrayObject/Boid is drawn N times with a single draw() call, and
// every instance's position/velocity is fetched here in the *vertex*
// shader via @builtin(instance_index) - there is no per-instance vertex
// buffer at all. That is the deliberate contrast with the Instancing demo,
// which supplies per-instance offset/scale/colour through a step_mode
// "instance" vertex buffer: here the same storage buffers the compute pass
// just wrote are read straight back in the next render pass, with no CPU
// round trip and no extra buffer to keep in sync.
//
// Same vec4-for-vec3 storage-buffer padding rule as BoidsCompute.wgsl
// applies to pos_buf/vel_buf - see that file for why.

struct Uniforms {
    view_proj: mat4x4<f32>,
    boid_scale: f32,
    speed_min: f32,
    speed_max: f32,
    _pad0: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> pos_buf: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vel_buf: array<vec4<f32>>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) colour: vec3<f32>,
};

@vertex
fn vertex_main(
    @location(0) local_pos: vec3<f32>,
    @builtin(instance_index) instance: u32,
) -> VertexOutput {
    let boid_pos = pos_buf[instance].xyz;
    let vel = vel_buf[instance].xyz;
    let speed = length(vel);

    // Orient the arrow's local +z (its nose, see the vertex list in main.py)
    // along the boid's velocity. Build an orthonormal basis from a world-up
    // reference, guarding the case where velocity is (near) parallel to
    // that reference - cross(world_up, forward) would otherwise collapse
    // towards zero length and normalize() would return NaN.
    var forward = vec3<f32>(0.0, 0.0, 1.0);
    if (speed > 0.00001) {
        forward = vel / speed;
    }
    var world_up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(dot(forward, world_up)) > 0.999) {
        world_up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let right = normalize(cross(world_up, forward));
    let up = cross(forward, right);

    let world_pos = boid_pos
        + right * local_pos.x * uniforms.boid_scale
        + up * local_pos.y * uniforms.boid_scale
        + forward * local_pos.z * uniforms.boid_scale;

    var output: VertexOutput;
    output.position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);

    // Colour by speed (slow = blue, fast = red) - a free readout of the
    // simulation state that costs nothing extra since vel_buf is already
    // fetched for the orientation basis above.
    let range = max(uniforms.speed_max - uniforms.speed_min, 0.0001);
    let t = clamp((speed - uniforms.speed_min) / range, 0.0, 1.0);
    output.colour = mix(vec3<f32>(0.2, 0.4, 1.0), vec3<f32>(1.0, 0.25, 0.2), t);
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.colour, 1.0);
}
