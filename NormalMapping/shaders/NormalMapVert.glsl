#version 410 core
layout (location = 0) in vec3 inVert;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inTangent;
layout (location = 4) in vec3 inBiTangent;

out VS_OUT {
    vec3 fragPos;
    vec2 uv;
    vec3 tangentLightPos[3];
    vec3 tangentViewPos[3];
    vec3 tangentFragPos[3];
} vs_out;


uniform mat4 M;
uniform mat4 MVP;
uniform mat3 normalMatrix;
uniform vec3 viewPos;


struct Light
{
  vec3 position;
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
};
uniform Light light[3];


void main()
{
    // world-space fragment position
    vs_out.fragPos = vec3(M * vec4(inVert, 1.0));
    vs_out.uv = inUV;

    // transform and orthonormalize tangent/normal
    vec3 T = normalize(normalMatrix * inTangent);
    vec3 N = normalize(normalMatrix * inNormal);
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);

    // build TBN with T, B, N as columns (maps tangent-space -> world-space)
    mat3 TBN = mat3(T, B, N);

    // transform world-space light/view positions into tangent space by subtracting fragPos
    for (int i = 0; i < 3; ++i)
    {
        vs_out.tangentLightPos[i] = TBN * (light[i].position - vs_out.fragPos);
        vs_out.tangentViewPos[i]  = TBN * (viewPos - vs_out.fragPos);
        // after subtracting fragPos, the fragment is at the origin in tangent space
        vs_out.tangentFragPos[i]  = vec3(0.0);
    }

    gl_Position = MVP * vec4(inVert, 1.0);
}
