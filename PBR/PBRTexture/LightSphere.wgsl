// Minimal unlit solid-colour shader used to draw the light-indicator spheres.
// One dynamic-offset uniform slot per light supplies its MVP and colour.

struct LightSphere {
    MVP : mat4x4<f32>,
    colour : vec4<f32>,
};
@group(0) @binding(0) var<uniform> data : LightSphere;

struct VSOut {
    @builtin(position) position : vec4<f32>,
    @location(0) colour : vec4<f32>,
};

@vertex
fn vertex_main(@location(0) inVert : vec3<f32>) -> VSOut {
    var out : VSOut;
    out.position = data.MVP * vec4<f32>(inVert, 1.0);
    out.colour = data.colour;
    return out;
}

@fragment
fn fragment_main(in : VSOut) -> @location(0) vec4<f32> {
    return in.colour;
}
