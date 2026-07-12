#version 330 core
in vec3 fragPos;
in vec3 fragNormal;
out vec4 fragColour;

uniform vec3 lightPos;
uniform vec3 viewerPos;
uniform vec3 ambient;
uniform vec3 diffuseColour;
uniform vec3 specularColour;
uniform float shininess;

void main()
{
    vec3 n = normalize(fragNormal);
    vec3 l = normalize(lightPos - fragPos);
    vec3 v = normalize(viewerPos - fragPos);
    vec3 r = reflect(-l, n);

    vec3 diffuse = diffuseColour * max(dot(n, l), 0.0);
    vec3 specular = specularColour * pow(max(dot(r, v), 0.0), shininess);

    fragColour = vec4(ambient + diffuse + specular, 1.0);
}
