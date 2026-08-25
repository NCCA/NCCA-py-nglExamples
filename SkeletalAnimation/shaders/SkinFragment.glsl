#version 330 core

in vec3 fragNormalView;

uniform vec3 lightDirView;
uniform vec4 colour;

out vec4 fragColour;

void main()
{
    // A flat ambient floor plus one directional term is all a teaching
    // demo about skinning needs -- the point is the silhouette, not the
    // shading model.
    float nDotL = max(dot(normalize(fragNormalView), normalize(lightDirView)), 0.15);
    fragColour = vec4(colour.rgb * nDotL, colour.a);
}
