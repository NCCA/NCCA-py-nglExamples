// WGSL port of shaders/PhongVertex.glsl + shaders/PhongFragment.glsl -- three
// point lights (ambient/diffuse/specular each), Blinn-Phong specular term.
// Lighting is done in world space (worldPosition = M * vertex, not the
// view-space M*V), so normal_matrix here is the inverse-transpose of M
// alone, matching the OpenGL sibling.

struct Transform {
    M : mat4x4<f32>,
    MVP : mat4x4<f32>,
    normal_matrix : mat4x4<f32>,
    viewerPos : vec3<f32>,
};

struct Material {
    ambient : vec4<f32>,
    diffuse : vec4<f32>,
    specular : vec4<f32>,
    shininess : f32,
};

struct Light {
    position : vec3<f32>,
    ambient : vec4<f32>,
    diffuse : vec4<f32>,
    specular : vec4<f32>,
};

struct Lighting {
    material : Material,
    lights : array<Light, 3>,
};

@group(0) @binding(0) var<uniform> transform : Transform;
@group(0) @binding(1) var<uniform> lighting : Lighting;

struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) worldPosition : vec3<f32>,
    @location(1) normal : vec3<f32>,
};

@vertex
fn vertex_main(
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>,
) -> VertexOutput {
    var output : VertexOutput;
    output.worldPosition = (transform.M * vec4<f32>(position, 1.0)).xyz;
    output.normal = normalize((transform.normal_matrix * vec4<f32>(normal, 0.0)).xyz);
    output.position = transform.MVP * vec4<f32>(position, 1.0);
    return output;
}

fn point_light(i : i32, n : vec3<f32>, eye_dir : vec3<f32>, world_position : vec3<f32>) -> vec4<f32> {
    let light = lighting.lights[i];
    let l = normalize(light.position - world_position);
    let lambert = max(dot(n, l), 0.0);
    let diffuse = lighting.material.diffuse * light.diffuse * lambert;
    let ambient = lighting.material.ambient * light.ambient;
    var specular = vec4<f32>(0.0);
    if (lambert > 0.0) {
        let half_v = normalize(eye_dir + l);
        let n_dot_hv = max(dot(n, half_v), 0.0);
        specular = lighting.material.specular * light.specular * pow(n_dot_hv, lighting.material.shininess);
    }
    return ambient + diffuse + specular;
}

@fragment
fn fragment_main(input : VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(input.normal);
    let eye_dir = normalize(transform.viewerPos - input.worldPosition);
    var colour = vec4<f32>(0.0);
    for (var i = 0; i < 3; i = i + 1) {
        colour = colour + point_light(i, n, eye_dir, input.worldPosition);
    }
    colour.a = 1.0;
    return colour;
}
