#version 330 core

/// Second pass of the two-pass stencil outline: every vertex is pushed
/// outward along its own normal in object space before MVP is applied, so
/// the silhouette pass renders a slightly fattened copy of the mesh. Where
/// that fattened copy overlaps the object's own footprint the stencil test
/// (GL_NOTEQUAL against the value written in pass 1) rejects it -- only the
/// fringe beyond the original silhouette survives.
layout (location = 0) in vec3 inVert;
layout (location = 1) in vec3 inNormal;

uniform mat4 MVP;
/// object-space fattening distance, ~0.03 looks right for these primitives
uniform float outlineScale;

void main()
{
    vec3 fattened = inVert + inNormal * outlineScale;
    gl_Position = MVP * vec4(fattened, 1.0);
}
