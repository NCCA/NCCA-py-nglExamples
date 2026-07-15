#version 410 core
out vec4 FragColour;
in vec3 localPos;
uniform sampler2D equirectangularMap;
const vec2 invAtan = vec2(0.1591, 0.3183);
vec2 sampleSphericalMap(vec3 v) {
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= invAtan;
    uv += 0.5;
    return uv;
}
void main() {
    vec2 uv = sampleSphericalMap(normalize(localPos));
    FragColour = vec4(texture(equirectangularMap, uv).rgb, 1.0);
}
