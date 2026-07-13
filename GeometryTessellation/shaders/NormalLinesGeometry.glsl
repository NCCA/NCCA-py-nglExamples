#version 410 core

// One triangle in, at most one line per vertex (3) or one line for the
// whole face out -- either way that is <= 6 vertices (3 line segments).
layout(triangles) in;
layout(line_strip, max_vertices = 6) out;

in vec3 vPosView[];
in vec3 vNormalView[];

uniform mat4 project;
uniform float normalLength;
uniform bool faceMode;

void emitLine(vec3 start, vec3 dir)
{
    gl_Position = project * vec4(start, 1.0);
    EmitVertex();
    gl_Position = project * vec4(start + dir * normalLength, 1.0);
    EmitVertex();
    EndPrimitive();
}

void main()
{
    if (faceMode)
    {
        // One line per *face*: average position and average (unnormalised
        // sum, then normalised) normal of the triangle's three corners --
        // this is what makes faceted/flat shading normals visible, as
        // opposed to the smooth per-vertex normals below.
        vec3 centre = (vPosView[0] + vPosView[1] + vPosView[2]) / 3.0;
        vec3 faceNormal = normalize(vNormalView[0] + vNormalView[1] + vNormalView[2]);
        emitLine(centre, faceNormal);
    }
    else
    {
        // One line per vertex, along its own interpolated normal.
        for (int i = 0; i < 3; ++i)
        {
            emitLine(vPosView[i], vNormalView[i]);
        }
    }
}
