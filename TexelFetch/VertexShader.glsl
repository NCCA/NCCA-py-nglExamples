#version 410 core
layout(location=0) in vec2 xz;
uniform mat4 MVP;
uniform samplerBuffer yPosSampler;
void main()
{
  float ypos = texelFetch(yPosSampler, gl_VertexID).r;
  gl_Position = MVP * vec4(xz.x, ypos, xz.y, 1.0);
}
