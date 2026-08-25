#version 330 core

in vec3 worldPos;
in vec3 worldNormal;

// mode: 0 = reflect, 1 = refract, 2 = Schlick-Fresnel mix, 3 = plain diffuse
uniform int mode;
uniform float ior;
uniform vec3 camPos;
uniform vec3 lightDir;
uniform samplerCube skybox;

out vec4 fragColour;

void main()
{
    vec3 N = normalize(worldNormal);
    vec3 I = normalize(worldPos - camPos);
    vec3 colour;

    if (mode == 0)
    {
        vec3 R = reflect(I, N);
        colour = texture(skybox, R).rgb;
    }
    else if (mode == 1)
    {
        vec3 R = refract(I, N, 1.0 / ior);
        colour = texture(skybox, R).rgb;
    }
    else if (mode == 2)
    {
        vec3 reflectDir = reflect(I, N);
        vec3 refractDir = refract(I, N, 1.0 / ior);
        // Schlick's approximation: F0 from the two IORs (air = 1.0), then
        // grazing angles reflect more (that's why puddles turn to mirrors
        // near the horizon).
        float f0 = pow((1.0 - ior) / (1.0 + ior), 2.0);
        float cosTheta = max(dot(-I, N), 0.0);
        float fresnel = f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
        colour = mix(texture(skybox, refractDir).rgb, texture(skybox, reflectDir).rgb, fresnel);
    }
    else
    {
        float diffuse = max(dot(N, normalize(lightDir)), 0.0);
        colour = vec3(0.8, 0.75, 0.7) * (0.2 + 0.8 * diffuse);
    }

    fragColour = vec4(colour, 1.0);
}
