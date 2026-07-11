#version 330 core

/// Weighted blended OIT accumulation pass (McGuire & Bavoil, JCGT 2013).
///
/// Writes to two render targets at once (MRT):
///   location 0 "accum"  (RGBA16F) blended with  ONE, ONE
///       accumulates weight * (premultiplied colour, alpha) -- a SUM
///   location 1 "reveal" (R16F)    blended with  ZERO, ONE_MINUS_SRC_COLOR
///       accumulates (1 - alpha) products -- how much of the background
///       "shows through" all transparent layers -- a PRODUCT
///
/// Sums and products are commutative, so unlike the OVER operator the
/// result does not depend on the order fragments arrive in. The depth
/// weight makes near fragments dominate the average, approximating what a
/// correct sort would have produced.

in vec3 fragNormal;
in float viewZ;

uniform vec4 Colour;
uniform vec3 lightDir;

layout (location = 0) out vec4 accum;
layout (location = 1) out float reveal;

float weight(float z, float alpha)
{
    z = abs(z);
    return alpha * clamp(10.0 / (1e-5 + pow(z / 5.0, 2.0) + pow(z / 200.0, 6.0)),
                         1e-2, 3e3);
}

void main()
{
    float ndotl = abs(dot(normalize(fragNormal), normalize(lightDir)));
    vec3 shaded = Colour.rgb * (0.25 + 0.75 * ndotl);
    float a = Colour.a;

    float w = weight(viewZ, a);
    // premultiplied colour and alpha, both scaled by the depth weight
    accum = vec4(shaded * a, a) * w;
    // with blend (ZERO, ONE_MINUS_SRC_COLOR) the target becomes
    // dst *= (1 - a): the running transmittance of all layers
    reveal = a;
}
