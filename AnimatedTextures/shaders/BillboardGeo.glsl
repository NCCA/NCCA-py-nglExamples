#version 330 core
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

flat in float whichTexture[];
flat in float frameOffset[];

flat out float texID;
out vec2 texCoord;

uniform mat4 MV;
uniform mat4 projection;
uniform float time;

const float bbWidth = 0.5;
const float bbHeight = 1.0;
const float spriteOffset = 0.1;

void main()
{
    // Billboard in view space, where "facing the camera" is just the XY
    // plane -- this stays correct under any scene rotation/zoom (MV), unlike
    // computing a facing vector from a world-space camera position and then
    // letting MV rotate the already-built quad out of alignment.
    vec4 viewPos = MV * vec4(gl_in[0].gl_Position.xyz, 1.0);

    float ctime = floor(time + frameOffset[0]);
    float u0 = ctime * spriteOffset;
    float u1 = (ctime + 1.0) * spriteOffset;

    texID = whichTexture[0];

    vec4 p0 = viewPos + vec4(-bbWidth, 0.0, 0.0, 0.0);
    gl_Position = projection * p0;
    texCoord = vec2(u0, 0.0);
    EmitVertex();

    vec4 p1 = viewPos + vec4(-bbWidth, bbHeight, 0.0, 0.0);
    gl_Position = projection * p1;
    texCoord = vec2(u0, 1.0);
    EmitVertex();

    vec4 p2 = viewPos + vec4(bbWidth, 0.0, 0.0, 0.0);
    gl_Position = projection * p2;
    texCoord = vec2(u1, 0.0);
    EmitVertex();

    vec4 p3 = viewPos + vec4(bbWidth, bbHeight, 0.0, 0.0);
    gl_Position = projection * p3;
    texCoord = vec2(u1, 1.0);
    EmitVertex();

    EndPrimitive();
}
