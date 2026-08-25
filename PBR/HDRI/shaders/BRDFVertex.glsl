#version 410 core
out vec2 TexCoords;
void main() {
    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    TexCoords = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
