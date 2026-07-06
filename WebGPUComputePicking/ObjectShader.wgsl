// Diffuse shading + single-pass barycentric wireframe for the selectable
// scene objects, plus an integer object-ID entry point for picking.
//
// Every object is one instance in a storage buffer (model / normal matrix /
// colour / flags). Two fragment entry points share the vertex stage:
//
//   fragment_main -> shaded : world-space diffuse, plus a white wireframe
//                    over selected objects derived from per-triangle
//                    barycentric coordinates (rgba8unorm target)
//   fragment_pick -> picking: writes the object's integer ID straight to an
//                    r32uint target. Unlike colour-ID picking there is no
//                    float -> byte encoding and no 24-bit limit: the ID is a
//                    real u32 (0 is reserved for "background"). The result
//                    is consumed on the GPU by PickCompute.wgsl, never read
//                    back as an image.

struct Instance {
    model : mat4x4<f32>,
    normal_matrix : mat4x4<f32>,
    colour : vec4<f32>,
    flags : vec4<f32>,          // x = selected (0/1), y = pick id (as float)
};

struct Globals {
    view : mat4x4<f32>,
    projection : mat4x4<f32>,
    light_pos : vec4<f32>,
    light_diffuse : vec4<f32>,
    params : vec4<f32>,
};

@group(0) @binding(0) var<storage, read> instances : array<Instance>;
@group(0) @binding(1) var<uniform> globals : Globals;

struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
    @location(0) normal : vec3<f32>,
    @location(1) world_pos : vec3<f32>,
    @location(2) bary : vec3<f32>,
    @location(3) colour : vec4<f32>,
    @location(4) @interpolate(flat) pick_id : u32,
    @location(5) selected : f32,
};

fn extract_mat3x3(m : mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
}

@vertex
fn vertex_main(
    input : VertexInput,
    @builtin(vertex_index) vid : u32,
    @builtin(instance_index) iid : u32,
) -> VertexOutput {
    var out : VertexOutput;
    let inst = instances[iid];

    let world = inst.model * vec4<f32>(input.position, 1.0);
    out.clip_position = globals.projection * globals.view * world;
    out.world_pos = world.xyz;
    out.normal = extract_mat3x3(inst.normal_matrix) * input.normal;

    let corner = vid % 3u;
    out.bary = vec3<f32>(
        f32(corner == 0u),
        f32(corner == 1u),
        f32(corner == 2u),
    );

    out.colour = inst.colour;
    out.pick_id = u32(inst.flags.y + 0.5);
    out.selected = inst.flags.x;
    return out;
}

@fragment
fn fragment_main(input : VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(input.normal);
    let L = normalize(globals.light_pos.xyz - input.world_pos);
    let diffuse = max(dot(N, L), 0.0);
    var colour = input.colour.rgb * (0.15 + globals.light_diffuse.rgb * diffuse);

    // Wireframe overdraw for selected objects: edge / fwidth(edge) is the
    // pixel distance to the nearest triangle edge, giving a constant-width
    // line; the smoothstep on extent_px fades the wire out on triangles too
    // small to show a readable line (see SelectionManipulatorWebGPU README).
    if (input.selected > 0.5) {
        let edge = min(min(input.bary.x, input.bary.y), input.bary.z);
        let deriv = max(fwidth(edge), 1e-5);

        let thickness = 0.6;  // line half-width in pixels
        let dist_px = edge / deriv;
        var wire = 1.0 - smoothstep(thickness, thickness + 1.0, dist_px);

        let extent_px = 1.0 / deriv;
        wire = wire * smoothstep(4.0, 14.0, extent_px);

        colour = mix(colour, vec3<f32>(1.0, 1.0, 1.0), wire);
    }

    return vec4<f32>(colour, 1.0);
}

// Picking pass: the flat-interpolated integer object ID goes straight into
// the r32uint attachment. Only the .r component lands in the texture.
@fragment
fn fragment_pick(input : VertexOutput) -> @location(0) vec4<u32> {
    return vec4<u32>(input.pick_id, 0u, 0u, 1u);
}
