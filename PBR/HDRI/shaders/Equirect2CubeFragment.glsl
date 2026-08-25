#version 410 core
out vec4 FragColour;
in vec3 localPos;
uniform sampler2D equirectangularMap;
// v is negated against the usual GLSL constant: we upload the panorama
// unflipped, so its first row (straight up) lands at v == 0. The stock sign
// assumes a bottom-up upload and would fetch the ground when looking up.
const vec2 invAtan = vec2(0.1591, -0.3183);
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
