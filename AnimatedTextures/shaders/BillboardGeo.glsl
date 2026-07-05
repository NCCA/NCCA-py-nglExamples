#version 330 core
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

flat in float whichTexture[];
flat in float frameOffset[];

flat out float texID;
out vec2 texCoord;

uniform mat4 MVP;
uniform vec3 cameraPos;
uniform float time;

const float bbWidth = 0.5;
const float bbHeight = 1.0;
const float spriteOffset = 0.1;

void main()
{
    vec3 pos = gl_in[0].gl_Position.xyz;
    vec3 toCamera = normalize(cameraPos - pos);
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 right = cross(toCamera, up);

    float ctime = floor(time + frameOffset[0]);
    float u0 = ctime * spriteOffset;
    float u1 = (ctime + 1.0) * spriteOffset;

    texID = whichTexture[0];

    vec3 p0 = pos - right * bbWidth;
    gl_Position = MVP * vec4(p0, 1.0);
    texCoord = vec2(u0, 0.0);
    EmitVertex();

    vec3 p1 = pos - right * bbWidth + up * bbHeight;
    gl_Position = MVP * vec4(p1, 1.0);
    texCoord = vec2(u0, 1.0);
    EmitVertex();

    vec3 p2 = pos + right * bbWidth;
    gl_Position = MVP * vec4(p2, 1.0);
    texCoord = vec2(u1, 0.0);
    EmitVertex();

    vec3 p3 = pos + right * bbWidth + up * bbHeight;
    gl_Position = MVP * vec4(p3, 1.0);
    texCoord = vec2(u1, 1.0);
    EmitVertex();

    EndPrimitive();
}
