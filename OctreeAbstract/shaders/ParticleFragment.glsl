#version 410 core

in vec3 vertNormal;
in vec3 vertColour;
layout(location = 0) out vec4 fragColour;

void main()
{
  vec3 lightDirection = normalize(vec3(0.4, 1.0, 0.7));
  float diffuse = max(dot(normalize(vertNormal), lightDirection), 0.0);
  fragColour = vec4(vertColour * (0.2 + 0.8 * diffuse), 1.0);
}
