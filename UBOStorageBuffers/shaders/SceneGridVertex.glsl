#version 330 core

// Custom VAO for the colour grid: location 0 = position, location 1 = flat
// per-vertex colour (a checkerboard baked in on the CPU).
layout(location = 0) in vec3 inVert;
layout(location = 1) in vec3 inColour;

// The SAME SceneBlock declaration as SceneDiffuseVertex.glsl, in a
// completely different program. glUniformBlockBinding points this
// program's "SceneBlock" at the same GL_UNIFORM_BUFFER binding point (0)
// as the teapot program, so one glBufferSubData updates both every frame.
layout(std140) uniform SceneBlock
{
    mat4 VP;
    vec4 lightPos;
    vec4 lightColour;
};

uniform mat4 M;

out vec3 fragColour;

void main()
{
    fragColour = inColour;
    gl_Position = VP * M * vec4(inVert, 1.0);
}
