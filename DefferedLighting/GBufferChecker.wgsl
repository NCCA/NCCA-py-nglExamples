// simple checker shader
struct TransformUniforms {
    MVP : mat4x4<f32>,
    normalMatrix : mat4x4<f32>,
    M : mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> transforms : TransformUniforms;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vertex_main(input: VertexIn) -> VertexOut {
    var output: VertexOut;
    // The floor carries the mouse rotation and a -0.45 y offset, so the G-buffer
    // needs the transformed position or the lighting pass shades the wrong point.
    output.worldPos = (transforms.M * vec4<f32>(input.position, 1.0)).xyz;
    output.normal = normalize((transforms.normalMatrix * vec4<f32>(input.normal, 0.0)).xyz);
    output.position = transforms.MVP * vec4<f32>(input.position, 1.0);
    output.uv = input.uv;

    return output;
}

// Fragment Shader

struct MaterialParams {
    colour1: vec4<f32>,
    colour2: vec4<f32>,
    checkSize: f32,
    checkOn: u32,
};
@group(1) @binding(0) var<uniform> material: MaterialParams;


struct GBufferOutput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) albedo: vec4<f32>,
};


fn checker(uv: vec2<f32>) -> vec4<f32>
{
    if (material.checkOn == 0u)
    {
        return material.colour1;
    }
    else
    {
        let v_int = i32(floor(material.checkSize * uv.x)) + i32(floor(material.checkSize * uv.y));
        if (v_int % 2 == 0)
        {
            return material.colour2;
        }
        else
        {
            return material.colour1;
        }
    }
}

@fragment
fn fragment_main(in: VertexOut) -> GBufferOutput
{
    var output: GBufferOutput;
    output.position = vec4<f32>(in.worldPos, 1.0);
    output.normal = vec4<f32>(normalize(in.normal), 0.8);
    output.albedo = checker(in.uv);
    return output;
}
