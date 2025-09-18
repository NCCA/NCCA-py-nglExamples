#version 410 core

layout(location=0) in vec3 inPos;
layout(location=1) in vec4 inColour;

uniform mat4 MVP;
out vec4 colour;
void main()
{
  gl_Position=MVP*vec4(inPos,1);
  colour=inColour;
}
