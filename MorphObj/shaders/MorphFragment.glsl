#version 410 core

in vec3 eyePosition;
in vec3 eyeNormal;
layout(location = 0) out vec4 fragColour;

uniform vec3 lightPosition;

void main()
{
  vec3 normal = normalize(eyeNormal);
  vec3 lightDirection = normalize(lightPosition - eyePosition);
  vec3 viewDirection = normalize(-eyePosition);
  vec3 halfVector = normalize(lightDirection + viewDirection);
  float diffuse = max(dot(normal, lightDirection), 0.0);
  float specular = diffuse > 0.0 ? pow(max(dot(normal, halfVector), 0.0), 64.0) : 0.0;
  vec3 colour = vec3(0.08) + vec3(0.72) * diffuse + vec3(0.35) * specular;
  fragColour = vec4(colour, 1.0);
}
