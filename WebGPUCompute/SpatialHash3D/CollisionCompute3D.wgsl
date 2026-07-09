// 3D Grid-based particle collision detection with spatial hashing

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var<storage, read_write> grid_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> grid_offsets: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cell_particle_count: array<atomic<u32>>;
// Tightly packed xyz positions for the render vertex buffer - the Particle
// struct itself cannot be used directly because vec3 fields are 16-byte
// aligned in storage buffers, giving a 32-byte stride the vertex layout
// does not expect.
@group(0) @binding(5) var<storage, read_write> render_positions: array<f32>;

// vec3 fields are 16-byte aligned in WGSL storage buffers, so this struct has
// a 32-byte stride; the Python particle dtype pads to match.
struct Particle {
    pos: vec3<f32>,
    vel: vec3<f32>,
};

struct SimParams {
    dt: f32,
    width: f32,
    height: f32,
    depth: f32,
    wind_x: f32,
    wind_y: f32,
    wind_z: f32,
    grid_width: u32,
    grid_height: u32,
    grid_depth: u32,
    cell_size: f32,
    particle_radius: f32,
};

// Hash function to convert 3D grid coordinate to 1D cell index
fn grid_hash(grid_pos: vec3<i32>) -> u32 {
    let gw = i32(params.grid_width);
    let gh = i32(params.grid_height);
    let gd = i32(params.grid_depth);

    // Clamp to grid bounds
    let x = clamp(grid_pos.x, 0, gw - 1);
    let y = clamp(grid_pos.y, 0, gh - 1);
    let z = clamp(grid_pos.z, 0, gd - 1);

    return u32((z * gh + y) * gw + x);
}

// Convert world position to grid cell
fn world_to_grid(pos: vec3<f32>) -> vec3<i32> {
    let half_width = params.width / 2.0;
    let half_height = params.height / 2.0;
    let half_depth = params.depth / 2.0;

    // Shift to [0, width], [0, height], [0, depth]
    let x = (pos.x + half_width) / params.cell_size;
    let y = (pos.y + half_height) / params.cell_size;
    let z = (pos.z + half_depth) / params.cell_size;

    return vec3<i32>(i32(floor(x)), i32(floor(y)), i32(floor(z)));
}

// Phase 1: Clear grid
@compute @workgroup_size(64)
fn clear_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let total_cells = params.grid_width * params.grid_height * params.grid_depth;

    if (index < total_cells) {
        atomicStore(&grid_offsets[index], 0u);
        atomicStore(&cell_particle_count[index], 0u);
    }
}

// Phase 2: Count particles per cell
@compute @workgroup_size(64)
fn count_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&particles)) {
        return;
    }

    let p = particles[index];
    let grid_pos = world_to_grid(p.pos);
    let cell_hash = grid_hash(grid_pos);

    atomicAdd(&cell_particle_count[cell_hash], 1u);
}

// Phase 3: Build grid offsets (prefix sum) - run with 1 workgroup
@compute @workgroup_size(1)
fn build_offsets(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_cells = params.grid_width * params.grid_height * params.grid_depth;
    var sum = 0u;

    for (var i = 0u; i < total_cells; i++) {
        let count = atomicLoad(&cell_particle_count[i]);
        atomicStore(&grid_offsets[i], sum);
        sum += count;
    }
}

// Phase 4: Fill grid with particle indices
@compute @workgroup_size(64)
fn fill_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&particles)) {
        return;
    }

    let p = particles[index];
    let grid_pos = world_to_grid(p.pos);
    let cell_hash = grid_hash(grid_pos);

    // Count how many particles with lower index are in the same cell
    var local_offset = 0u;
    for (var i = 0u; i < index; i++) {
        let other_grid = world_to_grid(particles[i].pos);
        if (grid_hash(other_grid) == cell_hash) {
            local_offset++;
        }
    }

    let base_offset = atomicLoad(&grid_offsets[cell_hash]);
    grid_indices[base_offset + local_offset] = index;
}

// Phase 5: Detect and resolve collisions
@compute @workgroup_size(64)
fn detect_collisions(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&particles)) {
        return;
    }

    var p = particles[index];
    let grid_pos = world_to_grid(p.pos);

    // Check 27 neighboring cells (3x3x3 cube including self)
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighbor_pos = grid_pos + vec3<i32>(dx, dy, dz);
                let neighbor_x = neighbor_pos.x;
                let neighbor_y = neighbor_pos.y;
                let neighbor_z = neighbor_pos.z;

                // Check bounds
                if (neighbor_x < 0 || neighbor_x >= i32(params.grid_width) ||
                    neighbor_y < 0 || neighbor_y >= i32(params.grid_height) ||
                    neighbor_z < 0 || neighbor_z >= i32(params.grid_depth)) {
                    continue;
                }

                let cell_hash = grid_hash(neighbor_pos);

                // Get the range of particles in this cell
                let cell_start = atomicLoad(&grid_offsets[cell_hash]);
                let cell_count = atomicLoad(&cell_particle_count[cell_hash]);
                let cell_end = cell_start + cell_count;

                // Check collisions with particles in this cell
                for (var i = cell_start; i < cell_end; i++) {
                    let other_index = grid_indices[i];

                    if (other_index == index) {
                        continue;
                    }

                    let other = particles[other_index];
                    let diff = p.pos - other.pos;
                    let dist = length(diff);
                    let collision_dist = params.particle_radius * 2.0;

                    if (dist < collision_dist && dist > 0.001) {
                        // Collision detected - elastic collision response
                        let normal = diff / dist;

                        // Separate particles
                        let overlap = collision_dist - dist;
                        p.pos += normal * overlap * 0.5;

                        // Velocity response (elastic collision)
                        let rel_vel = p.vel - other.vel;
                        let vel_along_normal = dot(rel_vel, normal);

                        if (vel_along_normal < 0.0) {
                            // Damping factor for energy loss
                            let restitution = 0.8;
                            p.vel -= (1.0 + restitution) * vel_along_normal * normal * 0.5;
                        }
                    }
                }
            }
        }
    }

    particles[index] = p;
}

fn wrap_boundaries(p_pos: vec3<f32>, half_width: f32, half_height: f32, half_depth: f32) -> vec3<f32> {
    var pos = p_pos;

    // Wrap X
    if (pos.x < -half_width) {
        pos.x += params.width;
    } else if (pos.x > half_width) {
        pos.x -= params.width;
    }

    // Wrap Y
    if (pos.y < -half_height) {
        pos.y += params.height;
    } else if (pos.y > half_height) {
        pos.y -= params.height;
    }

    // Wrap Z
    if (pos.z < -half_depth) {
        pos.z += params.depth;
    } else if (pos.z > half_depth) {
        pos.z -= params.depth;
    }

    return pos;
}

// Phase 6: Update physics (movement and boundary collision)
@compute @workgroup_size(64)
fn update_physics(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&particles)) {
        return;
    }

    let half_width = params.width / 2.0;
    let half_height = params.height / 2.0;
    let half_depth = params.depth / 2.0;

    var p = particles[index];

    // Apply wind and update position
    let wind = vec3<f32>(params.wind_x, params.wind_y, params.wind_z);
    let effective_velocity = p.vel + wind;
    p.pos += effective_velocity * params.dt;

    p.pos = wrap_boundaries(p.pos, half_width, half_height, half_depth);

    particles[index] = p;

    // Write the final position into the packed render buffer
    render_positions[index * 3u + 0u] = p.pos.x;
    render_positions[index * 3u + 1u] = p.pos.y;
    render_positions[index * 3u + 2u] = p.pos.z;
}