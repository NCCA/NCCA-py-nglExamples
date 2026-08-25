// Diffuse teapot: a uniform SceneBlock (shared with GridShader.wgsl via the
// SAME bind group layout / buffer) + a uniform MaterialBlock that carries
// the same std140-style vec3 padding trap as the OpenGL half of this demo
// -- WGSL's default "uniform address space" layout applies the identical
// 16-byte vec3 alignment rule: `specularColour` sits at byte offset 16
// (not 12, albedo's 12 bytes are followed by 4 bytes of padding) while
// `shininess`, a lone f32 with 4-byte alignment, packs straight after it
// at offset 28.
//
// The point-light accumulation loop reads a var<storage, read> array whose
// LENGTH is not fixed at shader-compile time (arrayLength(&lights)) -- a
// GL 4.1 UBO cannot do this at all (its arrays must be a fixed max size
// declared in the shader); resizing the light count means recreating the
// storage buffer, never touching this shader.

struct SceneBlock {
    VP: mat4x4<f32>,
    lightPos: vec4<f32>,
    lightColour: vec4<f32>,
};

struct MaterialBlock {
    albedo: vec3<f32>,
    specularColour: vec3<f32>,
    shininess: f32,
};

struct PointLight {
    position: vec3<f32>,
    intensity: f32,
    colour: vec3<f32>,
    _pad: f32,
};

struct ObjectUniforms {
    M: mat4x4<f32>,
    normalMatrix: mat4x4<f32>,
    viewPos: vec4<f32>,
};

@group(0) @binding(0) var<uniform> scene: SceneBlock;
@group(0) @binding(1) var<uniform> material: MaterialBlock;
@group(0) @binding(2) var<storage, read> lights: array<PointLight>;
@group(0) @binding(3) var<uniform> object: ObjectUniforms;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vertex_main(
    @location(0) inPos: vec3<f32>,
    @location(1) inNormal: vec3<f32>,
    @location(2) inUV: vec2<f32>,
) -> VertexOut {
    var out: VertexOut;
    let worldPos = object.M * vec4<f32>(inPos, 1.0);
    out.worldPos = worldPos.xyz;
    out.normal = normalize((object.normalMatrix * vec4<f32>(inNormal, 0.0)).xyz);
    out.position = scene.VP * worldPos;
    return out;
}

@fragment
fn fragment_main(in: VertexOut) -> @location(0) vec4<f32> {
    let N = normalize(in.normal);
    let V = normalize(object.viewPos.xyz - in.worldPos);

    // key light from SceneBlock, always present
    var result = 0.15 * material.albedo;
    let L0 = normalize(scene.lightPos.xyz - in.worldPos);
    let diff0 = max(dot(N, L0), 0.0);
    result += diff0 * material.albedo * scene.lightColour.rgb;

    // In the corrupted/naive case shininess reads 0 (max() clamps the
    // exponent to 1, smearing the highlight across the lit side) and
    // specularColour reads (specular.g, specular.b, shininess) -- the
    // scrambled blue channel holds the CPU-side shininess value, blowing
    // the glare out blue-white. Same visible cue as the GL demo.
    let H0 = normalize(L0 + V);
    let spec0 = pow(max(dot(N, H0), 0.0), max(material.shininess, 1.0));
    result += spec0 * material.specularColour * scene.lightColour.rgb;

    // runtime-sized storage-buffer point lights -- the thing a UBO cannot
    // express: this loop bound comes from the buffer's actual size, not a
    // compile-time constant.
    let count = arrayLength(&lights);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let light = lights[i];
        let toLight = light.position - in.worldPos;
        let dist2 = max(dot(toLight, toLight), 0.05);
        let L = toLight / sqrt(dist2);
        let diff = max(dot(N, L), 0.0);
        result += diff * material.albedo * light.colour * (light.intensity / dist2);
    }

    return vec4<f32>(result, 1.0);
}
