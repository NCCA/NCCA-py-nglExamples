// Ported from shaders/common.glsl -- the Materials/Lights structs and the
// light/material/time/repeat uniforms shared between the vertex and
// fragment stages there become one shared set of WGSL uniform structs here
// (std140-style layout, explicit padding fields where WGSL's own alignment
// rules would otherwise leave a gap numpy doesn't know to insert).

struct Material {
    ambient: vec4<f32>,
    diffuse: vec4<f32>,
    specular: vec4<f32>,
    shininess: f32,
    _pad0: vec3<f32>,
};

struct Light {
    position: vec3<f32>,
    _pad0: f32,
    ambient: vec4<f32>,
    diffuse: vec4<f32>,
    specular: vec4<f32>,
};

struct Transform {
    m: mat4x4<f32>,
    mvp: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    viewer_pos: vec3<f32>,
    _pad0: f32,
};

struct Params {
    time: f32,
    repeat: f32,
    _pad0: vec2<f32>,
};

@group(0) @binding(0) var<uniform> transform: Transform;
@group(0) @binding(1) var<uniform> material: Material;
@group(0) @binding(2) var<uniform> light: Light;
@group(0) @binding(3) var<uniform> params: Params;
