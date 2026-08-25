#version 330 core
in vec3 fragPos;
in vec3 fragNormal;
out vec4 fragColour;

uniform vec3 lightPos;
uniform vec3 viewerPos;

const vec3 ambient = vec3(0.274725, 0.1995, 0.0745);
const vec3 diffuseColour = vec3(0.75164, 0.60648, 0.22648);
const vec3 specularColour = vec3(0.628281, 0.555802, 0.3666065);
const float shininess = 51.2;

void main()
{
    vec3 n = gl_FrontFacing ? normalize(fragNormal) : normalize(-fragNormal);
    vec3 l = normalize(lightPos - fragPos);
    vec3 v = normalize(viewerPos - fragPos);
    vec3 r = reflect(-l, n);

    vec3 diffuse = diffuseColour * max(dot(n, l), 0.0);
    vec3 specular = specularColour * pow(max(dot(r, v), 0.0), shininess);

    fragColour = vec4(ambient + diffuse + specular, 1.0);
}
