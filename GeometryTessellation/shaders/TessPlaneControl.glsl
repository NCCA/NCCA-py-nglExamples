#version 410 core

// 4 control points per patch (a quad), quads-domain evaluation.
layout(vertices = 4) out;

in vec3 vWorldPos[];
out vec3 tcWorldPos[];

uniform vec3 cameraPosWorld;
uniform float nearDistance;
uniform float farDistance;
uniform bool fixedLevel;
uniform float fixedLevelValue;

// Distance -> tessellation level: close geometry gets 64, far geometry
// decays linearly (in distance) down to 1. tess_grid.py's
// tess_level_from_distance() implements the identical policy in numpy so
// the LOD curve itself is pytest-covered without a GL context.
float levelFromDistance(vec3 worldPos)
{
    if (fixedLevel)
    {
        return fixedLevelValue;
    }
    float d = distance(worldPos, cameraPosWorld);
    float t = clamp((d - nearDistance) / (farDistance - nearDistance), 0.0, 1.0);
    return clamp(mix(64.0, 1.0, t), 1.0, 64.0);
}

void main()
{
    tcWorldPos[gl_InvocationID] = vWorldPos[gl_InvocationID];

    // The tessellation levels are per-patch state, not per-invocation --
    // writing them from every invocation is a race, so only invocation 0
    // may touch gl_TessLevelOuter/Inner.
    if (gl_InvocationID == 0)
    {
        // Level per edge, driven by the distance to the edge's midpoint
        // (approximated with the average of its two corners) so a patch
        // straddling the near/far LOD boundary tessellates unevenly
        // rather than popping as a whole.
        float l0 = levelFromDistance(0.5 * (vWorldPos[0] + vWorldPos[3]));
        float l1 = levelFromDistance(0.5 * (vWorldPos[0] + vWorldPos[1]));
        float l2 = levelFromDistance(0.5 * (vWorldPos[1] + vWorldPos[2]));
        float l3 = levelFromDistance(0.5 * (vWorldPos[2] + vWorldPos[3]));

        gl_TessLevelOuter[0] = l0;
        gl_TessLevelOuter[1] = l1;
        gl_TessLevelOuter[2] = l2;
        gl_TessLevelOuter[3] = l3;
        gl_TessLevelInner[0] = max(l1, l3);
        gl_TessLevelInner[1] = max(l0, l2);
    }
}
