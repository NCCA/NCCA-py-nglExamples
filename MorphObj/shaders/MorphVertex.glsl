#version 410 core

layout(location = 0) in vec3 baseVert;
layout(location = 1) in vec3 baseNormal;
layout(location = 2) in vec3 poseVert1;
layout(location = 3) in vec3 poseNormal1;
layout(location = 4) in vec3 poseVert2;
layout(location = 5) in vec3 poseNormal2;

uniform mat4 MVP;
uniform mat4 MV;
uniform mat3 normalMatrix;
uniform float weight1;
uniform float weight2;

out vec3 eyePosition;
out vec3 eyeNormal;

void main()
{
  vec3 finalPosition = baseVert + weight1 * poseVert1 + weight2 * poseVert2;
  vec3 finalNormal = baseNormal + weight1 * poseNormal1 + weight2 * poseNormal2;
  eyePosition = vec3(MV * vec4(finalPosition, 1.0));
  eyeNormal = normalize(normalMatrix * finalNormal);
  gl_Position = MVP * vec4(finalPosition, 1.0);
}
