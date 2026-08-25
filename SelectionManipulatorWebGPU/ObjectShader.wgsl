// Diffuse shading + single-pass barycentric wireframe overdraw for the
// selectable scene objects.
//
// Every object is one instance in a storage buffer (model / normal matrix /
// colour / pick colour / flags). The same pipeline renders two ways depending
// on globals.params.x:
//   0 -> shaded  : world-space diffuse, plus a white wireframe over selected
//                  objects derived from per-triangle barycentric coordinates
//   1 -> picking : flat pick_colour (unique colour-ID per object) with no
//                  lighting, read back on the CPU to resolve what was clicked
//
// The barycentric coordinate needs no extra vertex buffer: the geometry is a
// non-indexed triangle soup, so vertex_index % 3 identifies the triangle
// corner and we emit (1,0,0)/(0,1,0)/(0,0,1) which interpolate to ~0 on edges.

struct Instance {
    model : mat4x4<f32>,
    normal_matrix : mat4x4<f32>,
    colour : vec4<f32>,
    pick_colour : vec4<f32>,
    flags : vec4<f32>,          // x = selected (0/1)
};

struct Globals {
    view : mat4x4<f32>,
    projection : mat4x4<f32>,
    light_pos : vec4<f32>,
    light_diffuse : vec4<f32>,
    params : vec4<f32>,         // x = render_mode (0 shade, 1 pick)
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
    @location(4) pick_colour : vec4<f32>,
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
    out.pick_colour = inst.pick_colour;
    out.selected = inst.flags.x;
    return out;
}

@fragment
fn fragment_main(input : VertexOutput) -> @location(0) vec4<f32> {
    // Picking pass: flat unique colour, no lighting, no wireframe.
    if (globals.params.x > 0.5) {
        return input.pick_colour;
    }

    let N = normalize(input.normal);
    let L = normalize(globals.light_pos.xyz - input.world_pos);
    let diffuse = max(dot(N, L), 0.0);
    var colour = input.colour.rgb * (0.15 + globals.light_diffuse.rgb * diffuse);

    // Wireframe overdraw for selected objects. See the README ("Adaptive
    // wireframe") for the derivation; briefly:
    //   * edge  = smallest barycentric coord = barycentric distance to the
    //             nearest triangle edge.
    //   * deriv = its screen-space rate of change (fwidth). edge / deriv is
    //             therefore the distance to the edge measured in *pixels*, so
    //             the line keeps a constant thickness however far away or
    //             foreshortened the triangle is.
    //   * 1 / deriv approximates the triangle's on-screen extent in pixels; we
    //             fade the wire out for triangles only a few pixels across so a
    //             dense mesh keeps its shaded surface instead of washing out to
    //             solid white.
    if (input.selected > 0.5) {
        let edge = min(min(input.bary.x, input.bary.y), input.bary.z);
        let deriv = max(fwidth(edge), 1e-5);

        let thickness = 0.6;  // line half-width in pixels
        let dist_px = edge / deriv;
        var wire = 1.0 - smoothstep(thickness, thickness + 1.0, dist_px);

        // dense-mesh thinning: extent_px is roughly how many pixels the
        // triangle spans; below ~4px it cannot show a readable line, so fade.
        let extent_px = 1.0 / deriv;
        wire = wire * smoothstep(4.0, 14.0, extent_px);

        colour = mix(colour, vec3<f32>(1.0, 1.0, 1.0), wire);
    }

    return vec4<f32>(colour, 1.0);
}
