#version 410 core

// Sphere-traced signed distance fields, entirely in one fragment shader.
// Every SDF function here (sd_sphere, sd_box, sd_torus, sd_plane,
// smooth_min, scene) is a line-for-line transcription of the numpy
// versions in sdf_maths.py -- that file is what the tests/ folder checks,
// this is the same maths running per-pixel on the GPU. The WebGPU sibling
// (RayMarch.wgsl) is the same transcription again, just in WGSL syntax.
//
// The technique: instead of rasterising triangles, march a ray outward
// from the camera in steps sized by the scene's distance field (the field
// tells you the *safe* step size -- nothing is closer than that, so you
// never overshoot a surface). Stop when you're within epsilon of a
// surface, or give up after MAX_STEPS, or after travelling FAR units with
// nothing hit.

in vec2 ndc;
out vec4 fragColour;

uniform vec3 camPos;
uniform vec3 camForward;
uniform vec3 camRight;
uniform vec3 camUp;
uniform float fovScale;   // tan(fov * 0.5)
uniform float aspect;
uniform float time;
uniform float smoothK;
uniform int shadowsOn;
uniform int aoOn;
uniform int showNormals;
uniform int showIterations;

const int MAX_STEPS = 100;
const float EPSILON = 1e-3;
const float FAR = 40.0;

const float PLANE_HEIGHT = 0.0;
const vec3 SPHERE_CENTRE = vec3(-0.9, 0.9, 0.0);
const float SPHERE_RADIUS = 0.9;
const vec3 BOX_CENTRE = vec3(0.9, 0.6, 0.0);
const vec3 BOX_HALF_EXTENTS = vec3(0.6, 0.6, 0.6);
const vec3 TORUS_CENTRE = vec3(0.0, 0.4, -1.4);
const float TORUS_MAJOR = 0.7;
const float TORUS_MINOR = 0.22;
const float MOVING_SPHERE_RADIUS = 0.5;
const float MOVING_SPHERE_ORBIT_RADIUS = 1.6;
const float MOVING_SPHERE_HEIGHT = 1.1;

// ---- SDF primitives (mirrors sdf_maths.py) --------------------------------

float sd_sphere(vec3 p, vec3 centre, float radius)
{
    return length(p - centre) - radius;
}

float sd_box(vec3 p, vec3 halfExtents)
{
    vec3 q = abs(p) - halfExtents;
    float outside = length(max(q, 0.0));
    float inside = min(max(q.x, max(q.y, q.z)), 0.0);
    return outside + inside;
}

float sd_torus(vec3 p, float majorRadius, float minorRadius)
{
    vec2 q = vec2(length(p.xz) - majorRadius, p.y);
    return length(q) - minorRadius;
}

float sd_plane(vec3 p, vec3 normal, float height)
{
    return dot(p, normal) - height;
}

float smooth_min(float a, float b, float k)
{
    if (k <= 0.0) {
        return min(a, b);
    }
    float h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

vec3 moving_sphere_centre(float t)
{
    float x = MOVING_SPHERE_ORBIT_RADIUS * cos(t);
    float z = MOVING_SPHERE_ORBIT_RADIUS * sin(t);
    return vec3(x, MOVING_SPHERE_HEIGHT, z);
}

// scene(): plane + (sphere, box, torus, moving sphere) smooth-blended
// together. Matches scene() in sdf_maths.py term for term.
float scene(vec3 p, float t, float k)
{
    float dPlane = sd_plane(p, vec3(0.0, 1.0, 0.0), PLANE_HEIGHT);
    float dSphere = sd_sphere(p, SPHERE_CENTRE, SPHERE_RADIUS);
    float dBox = sd_box(p - BOX_CENTRE, BOX_HALF_EXTENTS);
    float dTorus = sd_torus(p - TORUS_CENTRE, TORUS_MAJOR, TORUS_MINOR);
    float dMoving = sd_sphere(p, moving_sphere_centre(t), MOVING_SPHERE_RADIUS);

    float d = smooth_min(dSphere, dBox, k);
    d = smooth_min(d, dTorus, k);
    d = smooth_min(d, dMoving, k);
    return min(dPlane, d);
}

// central-difference gradient -- mirrors estimate_normal() in sdf_maths.py
vec3 estimate_normal(vec3 p, float t, float k)
{
    float eps = 1e-3;
    vec2 e = vec2(eps, 0.0);
    return normalize(vec3(
        scene(p + e.xyy, t, k) - scene(p - e.xyy, t, k),
        scene(p + e.yxy, t, k) - scene(p - e.yxy, t, k),
        scene(p + e.yyx, t, k) - scene(p - e.yyx, t, k)
    ));
}

// March from `p` toward the light, accumulating a penumbra factor as the
// ray grazes past other geometry -- the closer the closest approach gets
// to zero without actually hitting anything, the darker the shadow.
float soft_shadow(vec3 p, vec3 lightDir, float t, float k)
{
    float res = 1.0;
    float dist = 0.02;
    for (int i = 0; i < 32; ++i) {
        float d = scene(p + lightDir * dist, t, k);
        if (d < EPSILON) {
            return 0.0;
        }
        res = min(res, 8.0 * d / dist);
        dist += d;
        if (dist > 20.0) {
            break;
        }
    }
    return clamp(res, 0.0, 1.0);
}

// 5-tap ambient occlusion: sample the field a few small steps along the
// normal and see how much closer it is to a surface than "empty space"
// would predict.
float ambient_occlusion(vec3 p, vec3 n, float t, float k)
{
    float occlusion = 0.0;
    float scale = 1.0;
    for (int i = 1; i <= 5; ++i) {
        float stepDist = 0.03 * float(i);
        float d = scene(p + n * stepDist, t, k);
        occlusion += (stepDist - d) * scale;
        scale *= 0.6;
    }
    return clamp(1.0 - occlusion, 0.0, 1.0);
}

void main()
{
    float t = time;
    float k = smoothK;

    // ray basis is built entirely on the CPU (camForward/Right/Up) so the
    // shader only has to combine it with the pixel's NDC offset and fov.
    vec3 rayDir = normalize(camForward + ndc.x * aspect * fovScale * camRight
                                        + ndc.y * fovScale * camUp);

    float travelled = 0.0;
    int steps = 0;
    bool hit = false;
    vec3 p = camPos;

    for (steps = 0; steps < MAX_STEPS; ++steps) {
        p = camPos + rayDir * travelled;
        float d = scene(p, t, k);
        if (d < EPSILON) {
            hit = true;
            break;
        }
        travelled += d;
        if (travelled > FAR) {
            break;
        }
    }

    if (showIterations == 1) {
        // heat map: blue (cheap, few steps) through red (expensive, near
        // MAX_STEPS) -- the single most useful picture for explaining why
        // ray marching cost is scene- and view-dependent.
        float heat = float(steps) / float(MAX_STEPS);
        fragColour = vec4(heat, 0.3 * (1.0 - heat), 1.0 - heat, 1.0);
        return;
    }

    if (!hit) {
        // sky: a simple vertical gradient, darkened slightly by distance fog
        vec3 sky = mix(vec3(0.55, 0.65, 0.8), vec3(0.15, 0.18, 0.25), 0.5 + 0.5 * rayDir.y);
        fragColour = vec4(sky, 1.0);
        return;
    }

    vec3 n = estimate_normal(p, t, k);
    if (showNormals == 1) {
        fragColour = vec4(n * 0.5 + 0.5, 1.0);
        return;
    }

    vec3 lightDir = normalize(vec3(0.6, 0.8, 0.4));
    float diffuse = max(dot(n, lightDir), 0.0);

    float shadow = 1.0;
    if (shadowsOn == 1) {
        shadow = soft_shadow(p + n * EPSILON * 2.0, lightDir, t, k);
    }

    float ao = 1.0;
    if (aoOn == 1) {
        ao = ambient_occlusion(p, n, t, k);
    }

    vec3 baseColour = vec3(0.75, 0.72, 0.68);
    vec3 ambient = 0.15 * baseColour * ao;
    vec3 lit = baseColour * diffuse * shadow * ao;
    vec3 colour = ambient + lit;

    // simple exponential distance fog fades geometry into the sky colour
    float fogAmount = 1.0 - exp(-travelled * 0.035);
    vec3 fogColour = vec3(0.55, 0.65, 0.8);
    colour = mix(colour, fogColour, fogAmount);

    fragColour = vec4(colour, 1.0);
}
