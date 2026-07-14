// Reynolds flocking, one thread per boid.
//
// This is a deliberate, line-by-line transcription of step() in
// boid_maths.py onto the GPU: same three rules (separation/alignment/
// cohesion), same speed clamp, same soft wall push. Read that file's
// docstring first if the maths here looks unfamiliar - anything that
// changes on one side must change on the other.
//
// O(N^2): every boid scans every other boid, exactly like the naive pass
// this trades simplicity for. At the default N=2048 that is ~4M pairs,
// which is nothing for a GPU, but it does not scale forever - see
// WebGPUCompute/SpatialHash3D for the grid-bucketed version of the same
// idea (bin boids into cells, only scan neighbouring cells).
//
// State ping-pongs between two buffer sets (A/B): this pass reads the
// "in" buffers and writes the "out" buffers, and the Python side swaps
// which bind group is "in" vs "out" every frame rather than copying data
// back over itself (a boid must never read a value another invocation in
// the same dispatch has already overwritten).
//
// Storage-buffer pitfall: pos_in/pos_out/vel_in/vel_out are array<vec4<f32>>
// rather than array<vec3<f32>> - a vec3 in a storage buffer is padded to a
// 16-byte (vec4) stride, so a tightly-packed vec3 array on the Python side
// would desync from this shader's indexing after the first element. We
// sidestep that entirely by using vec4 on both sides and ignoring .w.

struct SimParams {
    dt: f32,
    rs: f32,           // separation radius
    ra: f32,           // alignment radius
    rc: f32,           // cohesion radius
    w_sep: f32,        // separation weight
    w_align: f32,      // alignment weight
    w_coh: f32,        // cohesion weight
    v_min: f32,        // minimum boid speed
    v_max: f32,        // maximum boid speed
    bound_x: f32,      // bounding box half-extents
    bound_y: f32,
    bound_z: f32,
    wall_margin: f32,  // distance from a wall at which avoidance starts
    w_wall: f32,       // wall avoidance weight
    n: u32,            // boid count
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> pos_in: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> vel_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pos_out: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> vel_out: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: SimParams;

// Soft push back towards the box centre once a boid is within wall_margin
// of a face, ramping linearly to full strength at the wall - the WGSL
// mirror of _wall_force() in boid_maths.py, done per-axis because WGSL
// has no clean way to loop over vec3 components by index.
fn wall_component(p: f32, half_extent: f32, margin: f32) -> f32 {
    let depth_hi = p - (half_extent - margin);
    let depth_lo = (-half_extent + margin) - p;
    var f = 0.0;
    f -= clamp(depth_hi, 0.0, margin) / margin;
    f += clamp(depth_lo, 0.0, margin) / margin;
    return f;
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n) {
        return;
    }

    let pos_i = pos_in[i].xyz;
    let vel_i = vel_in[i].xyz;

    var sep_sum = vec3<f32>(0.0, 0.0, 0.0);
    var align_sum = vec3<f32>(0.0, 0.0, 0.0);
    var coh_sum = vec3<f32>(0.0, 0.0, 0.0);
    var sep_count = 0u;
    var align_count = 0u;
    var coh_count = 0u;

    for (var j: u32 = 0u; j < params.n; j = j + 1u) {
        if (j == i) {
            continue;
        }
        let pos_j = pos_in[j].xyz;
        let diff = pos_i - pos_j;
        let dist = length(diff);

        if (dist < params.rs) {
            let safe_dist = max(dist, 0.000001);
            sep_sum += diff / safe_dist;
            sep_count += 1u;
        }
        if (dist < params.ra) {
            align_sum += vel_in[j].xyz;
            align_count += 1u;
        }
        if (dist < params.rc) {
            coh_sum += pos_j;
            coh_count += 1u;
        }
    }

    var separation = vec3<f32>(0.0, 0.0, 0.0);
    if (sep_count > 0u) {
        separation = sep_sum / f32(sep_count);
    }
    var alignment = vec3<f32>(0.0, 0.0, 0.0);
    if (align_count > 0u) {
        alignment = (align_sum / f32(align_count)) - vel_i;
    }
    var cohesion = vec3<f32>(0.0, 0.0, 0.0);
    if (coh_count > 0u) {
        cohesion = (coh_sum / f32(coh_count)) - pos_i;
    }

    let margin = max(params.wall_margin, 0.000001);
    var wall = vec3<f32>(0.0, 0.0, 0.0);
    wall.x = wall_component(pos_i.x, params.bound_x, margin);
    wall.y = wall_component(pos_i.y, params.bound_y, margin);
    wall.z = wall_component(pos_i.z, params.bound_z, margin);

    let accel = params.w_sep * separation
        + params.w_align * alignment
        + params.w_coh * cohesion
        + params.w_wall * wall;

    var new_vel = vel_i + accel * params.dt;

    // clamp speed to [v_min, v_max], preserving heading
    let speed = length(new_vel);
    let safe_speed = max(speed, 0.000001);
    let clamped_speed = clamp(speed, params.v_min, params.v_max);
    new_vel = (new_vel / safe_speed) * clamped_speed;

    let new_pos = pos_i + new_vel * params.dt;

    pos_out[i] = vec4<f32>(new_pos, 0.0);
    vel_out[i] = vec4<f32>(new_vel, 0.0);
}
