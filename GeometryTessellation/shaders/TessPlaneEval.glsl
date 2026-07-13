#version 410 core

// quads-domain, fractional_even spacing: integer tessellation levels would
// make new triangles pop into existence abruptly as the LOD crosses each
// integer boundary; fractional_even spacing grows/shrinks the outermost
// triangle ring's edges continuously instead, trading a (harmless) uneven
// last ring for the absence of visible popping. equal_spacing is the
// "wrong" one to reach for here -- it *always* pops.
layout(quads, fractional_even_spacing, ccw) in;

in vec3 tcWorldPos[];

uniform mat4 VP;
uniform float heightScale;
uniform float noiseScale;

out vec3 teWorldPos;
out vec3 teNormalWorld;
out float teHeight;

// ---- fbm: 4-octave value noise, no texture lookups ----------------------
float hash(vec2 p)
{
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float valueNoise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm(vec2 p)
{
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    for (int i = 0; i < 4; ++i)
    {
        value += amplitude * valueNoise(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

float heightAt(vec2 xz)
{
    return fbm(xz * noiseScale) * heightScale;
}

void main()
{
    // Bilinear interpolation of the 4 patch corners using gl_TessCoord.xy
    // -- this is the entire "tessellation" step for a quad patch: every
    // generated vertex is just a weighted blend of the same 4 inputs.
    vec3 bottom = mix(tcWorldPos[0], tcWorldPos[1], gl_TessCoord.x);
    vec3 top = mix(tcWorldPos[3], tcWorldPos[2], gl_TessCoord.x);
    vec3 worldPos = mix(bottom, top, gl_TessCoord.y);

    float height = heightAt(worldPos.xz);
    worldPos.y += height;

    // Normal by finite differences of the same noise field: sample the
    // height a small step away in x and z, the surface tangents are then
    // (eps, dHeight, 0) and (0, dHeight, eps), and their cross product is
    // the surface normal. No analytic derivative of fbm() needed.
    //
    // worldPos already carries the patch's control points through M (see
    // TessPlaneVertex.glsl), so these tangents -- and therefore the
    // normal below -- are already expressed in that same "world" frame;
    // no further model/normal-matrix transform is applied or needed.
    const float eps = 0.05;
    float hx = heightAt(worldPos.xz + vec2(eps, 0.0));
    float hz = heightAt(worldPos.xz + vec2(0.0, eps));
    vec3 tangentX = vec3(eps, hx - height, 0.0);
    vec3 tangentZ = vec3(0.0, hz - height, eps);

    teWorldPos = worldPos;
    teNormalWorld = normalize(cross(tangentZ, tangentX));
    teHeight = height;
    gl_Position = VP * vec4(worldPos, 1.0);
}
