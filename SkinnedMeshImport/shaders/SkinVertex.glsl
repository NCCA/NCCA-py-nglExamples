#version 410 core

layout(location = 0) in vec3 inVert;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
// Bone IDs arrive as floats (PyNGL vertex buffers are float32) and are
// cast to int below, rather than adding an integer-attribute path to the
// VAO for this one demo.
layout(location = 3) in vec4 inBoneIDs;
layout(location = 4) in vec4 inBoneWeights;

const int MAX_BONES = 64;
uniform mat4 MVP;
uniform mat4 M;
uniform mat4 MV;
uniform mat4 gBones[MAX_BONES];
uniform vec3 viewerPos;

out vec2 texCoord;
out vec3 fragmentNormal;
out vec3 eyeDirection;
out vec3 lightDir;
out vec3 halfVector;
out vec3 vPosition;

struct Light
{
    vec4 position;
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};
uniform Light light;

void main()
{
    mat4 boneTransform = gBones[int(inBoneIDs[0])] * inBoneWeights[0];
    boneTransform += gBones[int(inBoneIDs[1])] * inBoneWeights[1];
    boneTransform += gBones[int(inBoneIDs[2])] * inBoneWeights[2];
    boneTransform += gBones[int(inBoneIDs[3])] * inBoneWeights[3];

    vec4 skinnedPos = boneTransform * vec4(inVert, 1.0);
    gl_Position = MVP * skinnedPos;

    texCoord = inUV;
    vec4 skinnedNormal = boneTransform * vec4(inNormal, 0.0);
    fragmentNormal = normalize((M * skinnedNormal).xyz);

    vec4 worldPosition = M * vec4(inVert, 1.0);
    eyeDirection = normalize(viewerPos - worldPosition.xyz);

    vec4 eyeCoord = MV * skinnedPos;
    vPosition = eyeCoord.xyz / eyeCoord.w;

    lightDir = light.position.xyz - eyeCoord.xyz;
    float dist = length(lightDir);
    lightDir /= dist;
    halfVector = normalize(eyeDirection + lightDir);
}
