#version 410 core

/// @brief the vertex passed in
layout(location =0)in vec3 inVert;
/// @brief the normal passed in
layout(location =1)in vec3 inColour;

uniform mat4 MVP;
out vec3 vertColour;

void main()
{
  // calculate the vertex position
  gl_Position = MVP*vec4(inVert,1.0);
  vertColour = inColour;
}
