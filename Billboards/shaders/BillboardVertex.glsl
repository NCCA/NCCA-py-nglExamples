#version 330 core

// Non-standard attribute layout for this demo: the CPU rebuilds this VBO
// from scratch every frame (see billboard_maths.py / main.py), so there is
// no shared vertex format with ncca.ngl's Primitives (0=inVert, 1=inNormal,
// 2=inUV). Just position and UV -- the quad's facing is already baked into
// the positions on the CPU, so the shader has nothing clever to do.
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUV;

uniform mat4 MVP;

out vec2 vertUV;

void main()
{
    vertUV = inUV;
    gl_Position = MVP * vec4(inPosition, 1.0);
}
