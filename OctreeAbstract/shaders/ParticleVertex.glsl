#version 410 core

layout(location = 0) in vec3 spherePosition;
layout(location = 1) in vec3 sphereNormal;
layout(location = 2) in vec4 particlePositionRadius;
layout(location = 3) in vec4 particleColour;

uniform mat4 MVP;
uniform mat3 normalMatrix;

out vec3 vertNormal;
out vec3 vertColour;

void main()
{
  vec3 position = spherePosition * particlePositionRadius.w + particlePositionRadius.xyz;
  gl_Position = MVP * vec4(position, 1.0);
  vertNormal = normalize(normalMatrix * sphereNormal);
  vertColour = particleColour.rgb;
}
