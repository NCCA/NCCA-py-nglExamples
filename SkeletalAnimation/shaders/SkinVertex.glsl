#version 330 core

// Linear blend skinning (LBS) vs dual-quaternion skinning (DQS), selected
// per-draw by the useDQS uniform. This is a line-for-line transcription of
// skinning_maths.py's skin_lbs/skin_dqs (see that file's docstring) -- when
// one changes, the other must too, and tests/test_skinning_maths.py is the
// cross-check that they still agree.

layout(location = 0) in vec3 inVert;
layout(location = 1) in vec3 inNormal;
// Two nearest bone indices/weights per vertex (skinning_maths.compute_weights).
// Plain floats at locations 3/4 -- fine at this vertex count, no need for
// an integer attribute.
layout(location = 3) in vec2 inBoneIndices;
layout(location = 4) in vec2 inBoneWeights;

uniform mat4 MVP;
uniform mat3 normalMatrix;

// LBS palette: skinMatrix[i] = inverse_bind[i] @ pose[i], computed on the
// CPU by skinning_maths.compute_palette. Uploaded with glUniformMatrix4fv's
// transpose flag set to GL_TRUE (our numpy matrices are row-vector /
// row-major, translation in row 3; GL_TRUE tells GL to transpose them into
// its own column-major storage), so this shader can use plain `M * v`.
uniform mat4 bones[8];

// DQS palette: two vec4 per bone, (qr, qd), each stored **w-last** (x,y,z,w)
// -- reordered on upload from skinning_maths.py's (s,x,y,z) numpy layout,
// purely so this shader can use ordinary .xyz/.w swizzles instead of index
// juggling.
//   boneDQs[2*i]     = qr for bone i
//   boneDQs[2*i + 1] = qd for bone i
uniform vec4 boneDQs[16];

uniform bool useDQS;

out vec3 fragNormalView;

// Hamilton product, quaternions stored w-last (x, y, z, w).
vec4 quatMult(vec4 a, vec4 b)
{
    return vec4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

vec4 quatConj(vec4 q)
{
    return vec4(-q.xyz, q.w);
}

// Rotate v by the unit quaternion q via the sandwich product q v q*,
// expanded into its cross-product form -- avoids building a rotation
// matrix (and any row/column-major mismatch) altogether. Mirrors what
// skinning_maths.quat_to_mat3 computes, just applied directly to a vector.
vec3 quatRotate(vec4 q, vec3 v)
{
    vec3 u = q.xyz;
    return v + 2.0 * cross(u, cross(u, v) + q.w * v);
}

// Dual-quaternion linear blending (DLB, Kavan et al. 2007): hemisphere-align
// each influence against influence 0 before summing -- q and -q encode the
// same rotation but *cancel* if summed with opposite sign -- then
// renormalise both halves by the blended rotation's own length. Mirrors
// skinning_maths.dlb_blend exactly.
void blendDualQuats(vec2 idx, vec2 weights, out vec4 qrOut, out vec4 qdOut)
{
    vec4 qr0 = boneDQs[2 * int(idx.x)];
    vec4 qd0 = boneDQs[2 * int(idx.x) + 1];
    vec4 qr1 = boneDQs[2 * int(idx.y)];
    vec4 qd1 = boneDQs[2 * int(idx.y) + 1];

    if (dot(qr1, qr0) < 0.0) {
        qr1 = -qr1;
        qd1 = -qd1;
    }

    vec4 qrSum = weights.x * qr0 + weights.y * qr1;
    vec4 qdSum = weights.x * qd0 + weights.y * qd1;
    float n = length(qrSum);
    qrOut = qrSum / n;
    qdOut = qdSum / n;
}

void main()
{
    vec3 skinnedPos;
    vec3 skinnedNormal;

    if (useDQS) {
        vec4 qr;
        vec4 qd;
        blendDualQuats(inBoneIndices, inBoneWeights, qr, qd);
        // translation = 2 * (qd * qr_conjugate), vector part -- mirrors
        // skinning_maths.dq_to_mat.
        vec3 t = 2.0 * quatMult(qd, quatConj(qr)).xyz;
        skinnedPos = quatRotate(qr, inVert) + t;
        skinnedNormal = quatRotate(qr, inNormal);
    } else {
        // Linear blend of the two influences' skin matrices -- valid
        // because matrix application is linear in the matrix and the
        // weights sum to 1 (skinning_maths.skin_lbs relies on the same
        // identity to blend transformed points instead of matrices).
        mat4 skinMatrix =
            inBoneWeights.x * bones[int(inBoneIndices.x)] +
            inBoneWeights.y * bones[int(inBoneIndices.y)];
        skinnedPos = (skinMatrix * vec4(inVert, 1.0)).xyz;
        // Same palette for normals, no inverse-transpose -- fine for the
        // near-rigid, non-scaling bones this demo uses (see README).
        skinnedNormal = mat3(skinMatrix) * inNormal;
    }

    gl_Position = MVP * vec4(skinnedPos, 1.0);
    fragNormalView = normalize(normalMatrix * skinnedNormal);
}
