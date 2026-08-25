#version 330 core
// Debug-view quad for the BRDF LUT: four NDC corners generated purely from
// gl_VertexID (drawn as a GL_TRIANGLE_STRIP against an empty VAO) so the
// demo doesn't need a dedicated screen-space quad primitive just for this.
out vec2 uv;

void main()
{
    vec2 corners[4] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0,  1.0)
    );
    vec2 p = corners[gl_VertexID];
    uv = p * 0.5 + 0.5;
    // shrink to a small quad, parked in the bottom-right corner
    vec2 screen_pos = p * 0.22 + vec2(0.72, -0.72);
    gl_Position = vec4(screen_pos, 0.0, 1.0);
}
