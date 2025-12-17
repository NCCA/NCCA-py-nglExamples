#version 410 core

// Shader to generate a screen triangle with texture
// co-ordinates User just needs to render 3 verts
// With no inputs (see source)
// https://rauwendaal.net/2014/06/14/rendering-a-screen-covering-triangle-in-opengl/

out vec2 texCoord;
 
void main()
{
    float x = -1.0 + float((gl_VertexID & 1) << 2);
    float y = -1.0 + float((gl_VertexID & 2) << 1);
    texCoord.x = (x+1.0)*0.5;
    texCoord.y = (y+1.0)*0.5;
    gl_Position = vec4(x, y, 0, 1);
}