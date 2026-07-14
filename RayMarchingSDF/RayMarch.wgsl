// Sphere-traced signed distance fields (WebGPU).
//
// Same technique, same scene, same maths as the OpenGL sibling
// (shaders/RayMarchFragment.glsl) -- every function below has an
// identical GLSL twin there and a tested numpy twin in sdf_maths.py.
// WGSL has no #define/#version, so constants use `const` instead.

struct Params {
    cam_pos : vec3<f32>,
    fov_scale : f32,
    cam_forward : vec3<f32>,
    aspect : f32,
    cam_right : vec3<f32>,
    time : f32,
    cam_up : vec3<f32>,
    smooth_k : f32,
    shadows_on : u32,
    ao_on : u32,
    show_normals : u32,
    show_iterations : u32,
};

@group(0) @binding(0) var<uniform> params : Params;

const MAX_STEPS : i32 = 100;
const EPSILON : f32 = 1e-3;
const FAR : f32 = 40.0;

const PLANE_HEIGHT : f32 = 0.0;
const SPHERE_CENTRE = vec3<f32>(-0.9, 0.9, 0.0);
const SPHERE_RADIUS : f32 = 0.9;
const BOX_CENTRE = vec3<f32>(0.9, 0.6, 0.0);
const BOX_HALF_EXTENTS = vec3<f32>(0.6, 0.6, 0.6);
const TORUS_CENTRE = vec3<f32>(0.0, 0.4, -1.4);
const TORUS_MAJOR : f32 = 0.7;
const TORUS_MINOR : f32 = 0.22;
const MOVING_SPHERE_RADIUS : f32 = 0.5;
const MOVING_SPHERE_ORBIT_RADIUS : f32 = 1.6;
const MOVING_SPHERE_HEIGHT : f32 = 1.1;

struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) ndc : vec2<f32>,
};

// fullscreen triangle from the vertex index -- no vertex buffer, mirrors
// the GL side's gl_VertexID trick (see ScreenTri/ and RayMarchVertex.glsl)
@vertex
fn vertex_main(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
    var output : VertexOutput;
    let corner = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    let ndc = corner * 2.0 - 1.0;
    output.position = vec4<f32>(ndc, 0.0, 1.0);
    output.ndc = ndc;
    return output;
}

// ---- SDF primitives (mirrors sdf_maths.py / RayMarchFragment.glsl) -------

fn sd_sphere(p : vec3<f32>, centre : vec3<f32>, radius : f32) -> f32 {
    return length(p - centre) - radius;
}

fn sd_box(p : vec3<f32>, half_extents : vec3<f32>) -> f32 {
    let q = abs(p) - half_extents;
    let outside = length(max(q, vec3<f32>(0.0)));
    let inside = min(max(q.x, max(q.y, q.z)), 0.0);
    return outside + inside;
}

fn sd_torus(p : vec3<f32>, major_radius : f32, minor_radius : f32) -> f32 {
    let q = vec2<f32>(length(p.xz) - major_radius, p.y);
    return length(q) - minor_radius;
}

fn sd_plane(p : vec3<f32>, normal : vec3<f32>, height : f32) -> f32 {
    return dot(p, normal) - height;
}

fn smooth_min(a : f32, b : f32, k : f32) -> f32 {
    if (k <= 0.0) {
        return min(a, b);
    }
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

fn moving_sphere_centre(t : f32) -> vec3<f32> {
    let x = MOVING_SPHERE_ORBIT_RADIUS * cos(t);
    let z = MOVING_SPHERE_ORBIT_RADIUS * sin(t);
    return vec3<f32>(x, MOVING_SPHERE_HEIGHT, z);
}

// scene(): plane + (sphere, box, torus, moving sphere) smooth-blended
// together. Matches scene() in sdf_maths.py term for term.
fn scene(p : vec3<f32>, t : f32, k : f32) -> f32 {
    let d_plane = sd_plane(p, vec3<f32>(0.0, 1.0, 0.0), PLANE_HEIGHT);
    let d_sphere = sd_sphere(p, SPHERE_CENTRE, SPHERE_RADIUS);
    let d_box = sd_box(p - BOX_CENTRE, BOX_HALF_EXTENTS);
    let d_torus = sd_torus(p - TORUS_CENTRE, TORUS_MAJOR, TORUS_MINOR);
    let d_moving = sd_sphere(p, moving_sphere_centre(t), MOVING_SPHERE_RADIUS);

    var d = smooth_min(d_sphere, d_box, k);
    d = smooth_min(d, d_torus, k);
    d = smooth_min(d, d_moving, k);
    return min(d_plane, d);
}

// central-difference gradient -- mirrors estimate_normal() in sdf_maths.py
fn estimate_normal(p : vec3<f32>, t : f32, k : f32) -> vec3<f32> {
    let eps = 1e-3;
    let dx = vec3<f32>(eps, 0.0, 0.0);
    let dy = vec3<f32>(0.0, eps, 0.0);
    let dz = vec3<f32>(0.0, 0.0, eps);
    return normalize(vec3<f32>(
        scene(p + dx, t, k) - scene(p - dx, t, k),
        scene(p + dy, t, k) - scene(p - dy, t, k),
        scene(p + dz, t, k) - scene(p - dz, t, k)
    ));
}

fn soft_shadow(p : vec3<f32>, light_dir : vec3<f32>, t : f32, k : f32) -> f32 {
    var res = 1.0;
    var dist = 0.02;
    for (var i = 0; i < 32; i = i + 1) {
        let d = scene(p + light_dir * dist, t, k);
        if (d < EPSILON) {
            return 0.0;
        }
        res = min(res, 8.0 * d / dist);
        dist = dist + d;
        if (dist > 20.0) {
            break;
        }
    }
    return clamp(res, 0.0, 1.0);
}

fn ambient_occlusion(p : vec3<f32>, n : vec3<f32>, t : f32, k : f32) -> f32 {
    var occlusion = 0.0;
    var scale = 1.0;
    for (var i = 1; i <= 5; i = i + 1) {
        let step_dist = 0.03 * f32(i);
        let d = scene(p + n * step_dist, t, k);
        occlusion = occlusion + (step_dist - d) * scale;
        scale = scale * 0.6;
    }
    return clamp(1.0 - occlusion, 0.0, 1.0);
}

@fragment
fn fragment_main(input : VertexOutput) -> @location(0) vec4<f32> {
    let t = params.time;
    let k = params.smooth_k;

    let ray_dir = normalize(
        params.cam_forward
        + input.ndc.x * params.aspect * params.fov_scale * params.cam_right
        + input.ndc.y * params.fov_scale * params.cam_up
    );

    var travelled = 0.0;
    var steps = 0;
    var hit = false;
    var p = params.cam_pos;

    for (steps = 0; steps < MAX_STEPS; steps = steps + 1) {
        p = params.cam_pos + ray_dir * travelled;
        let d = scene(p, t, k);
        if (d < EPSILON) {
            hit = true;
            break;
        }
        travelled = travelled + d;
        if (travelled > FAR) {
            break;
        }
    }

    if (params.show_iterations == 1u) {
        let heat = f32(steps) / f32(MAX_STEPS);
        return vec4<f32>(heat, 0.3 * (1.0 - heat), 1.0 - heat, 1.0);
    }

    if (!hit) {
        let sky = mix(vec3<f32>(0.55, 0.65, 0.8), vec3<f32>(0.15, 0.18, 0.25), 0.5 + 0.5 * ray_dir.y);
        return vec4<f32>(sky, 1.0);
    }

    let n = estimate_normal(p, t, k);
    if (params.show_normals == 1u) {
        return vec4<f32>(n * 0.5 + 0.5, 1.0);
    }

    let light_dir = normalize(vec3<f32>(0.6, 0.8, 0.4));
    let diffuse = max(dot(n, light_dir), 0.0);

    var shadow = 1.0;
    if (params.shadows_on == 1u) {
        shadow = soft_shadow(p + n * EPSILON * 2.0, light_dir, t, k);
    }

    var ao = 1.0;
    if (params.ao_on == 1u) {
        ao = ambient_occlusion(p, n, t, k);
    }

    let base_colour = vec3<f32>(0.75, 0.72, 0.68);
    let ambient = 0.15 * base_colour * ao;
    let lit = base_colour * diffuse * shadow * ao;
    var colour = ambient + lit;

    let fog_amount = 1.0 - exp(-travelled * 0.035);
    let fog_colour = vec3<f32>(0.55, 0.65, 0.8);
    colour = mix(colour, fog_colour, fog_amount);

    return vec4<f32>(colour, 1.0);
}
