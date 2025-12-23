@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
};

struct SimParams {
    dt: f32,
    width: f32,
    height: f32,
    wind_x: f32,
    wind_y: f32,
};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&particles)) {
        return;
    }

    var p = particles[index];

    // Apply wind to velocity and update position
    let wind = vec2<f32>(params.wind_x, params.wind_y);
    let effective_velocity = p.vel + wind;
    p.pos += effective_velocity * params.dt;

    // Collision detection with boundaries
    let half_width = params.width / 2.0;
    let half_height = params.height / 2.0;

    if (p.pos.x < -half_width) {
        p.pos.x = -half_width;
        p.vel.x = -p.vel.x;
    } else if (p.pos.x > half_width) {
        p.pos.x = half_width;
        p.vel.x = -p.vel.x;
    }

    if (p.pos.y < -half_height) {
        p.pos.y = -half_height;
        p.vel.y = -p.vel.y;
    } else if (p.pos.y > half_height) {
        p.pos.y = half_height;
        p.vel.y = -p.vel.y;
    }

    particles[index] = p;
}
