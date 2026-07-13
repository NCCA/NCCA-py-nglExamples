#version 410 core
// Minimal position-only vertex shader shared by the terrain pass and the
// 2D UI pass. The MVP uniform is either perspective*view (terrain) or an
// orthographic pixel-space matrix (UI panels / buttons).
layout(location = 0) in vec3 inPos;
uniform mat4 MVP;

void main()
{
    gl_Position = MVP * vec4(inPos, 1.0);
}
