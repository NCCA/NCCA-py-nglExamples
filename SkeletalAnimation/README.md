# Skeletal Animation: LBS vs DQS

![](SkeletalAnimation.png)

A 4-bone chain along +y wrapped in a procedurally-built tube (no mesh or rig assets — everything, skeleton, weights and geometry, is generated in numpy). Every bone twists about its own long axis, and `D` swaps the skinning method between linear blend skinning (LBS) and dual-quaternion skinning (DQS) so you can watch them disagree.

- `main.py` — OpenGL, `PySide6.QtOpenGL.QOpenGLWindow`
- `skinning_maths.py` — the numpy-only CPU reference (skeleton FK, mesh, weights, both skinning methods), unit tested headless in `tests/`
- `shaders/SkinVertex.glsl` — a line-for-line transcription of the same maths, run on the GPU

## Controls

| Key               | Action                                                              |
| :---------------- | :------------------------------------------------------------------ |
| `D`               | toggle LBS / DQS                                                    |
| `T`               | snap to the candy-wrapper pose — a fixed 180° twist of the end bone |
| `S`               | toggle the skeleton overlay                                         |
| LMB / RMB / wheel | rotate / pan / zoom, `Space` resets, `Esc` quits                    |

## The candy-wrapper artefact

Both skinning methods start from the same bind pose, the same per-vertex weights (linear falloff between the two nearest bones), and the same posed bone matrices — the only thing that differs is _how the two influences per vertex get blended_.

LBS blends each influence's whole 4×4 matrix linearly: `v' = w0 * (v @ M0) + w1 * (v @ M1)`. That's fine for a bend, but twist a joint far enough and the two matrices start rotating a vertex towards opposite sides of the bone axis. Averaging them pulls the vertex _towards the axis itself_ rather than around it, so the cross-section pinches in — like paper twisted at both ends, hence "candy wrapper". Push the twist to 180° and a ring weighted 50/50 between the two bones collapses to (near) zero radius: `tests/test_skinning_maths.py::TestCandyWrapper` pins this down to the meter, plus a check that DQS gets the same case wrong (right, rather) by keeping the ring's radius.

Dual-quaternion skinning blends the _rotations_ instead of the matrices, via dual quaternion linear blending (DLB): each bone's rigid transform becomes a unit dual quaternion, the two influences are hemisphere-aligned (`q` and `-q` are the same rotation, but summing them unaligned cancels instead of blending) and blended, then renormalised. That blend takes the shortest path between the two orientations rather than the two positions, so the cross-section stays rigid all the way through the twist.

This is the artefact skeletal animation textbooks always reach for to justify DQS over LBS — see Kavan, Collins, Žára & O'Sullivan, ["Skinning with Dual Quaternions"](https://users.cs.utah.edu/~ladislav/kavan08geometric/kavan08geometric.pdf), I3D 2007, which coined the "candy wrapper" name and the DLB blending rule used here.

## Implementation notes

- The skeleton is a straight chain: `pose_matrices` twists each bone about its own +y axis (the axis the chain runs along), so twisting never moves a bone's origin, only its orientation — this isolates the twist deformation from any bending.
- Vertex weights come from the two nearest bone origins with linear (inverse-distance) falloff — no painting, no asset.
- Normals are skinned with the same palette as positions (no inverse-transpose). That's a simplification that only holds up because these bones never scale non-uniformly; a rig with scaling joints would need the proper inverse-transpose normal matrix per bone.
- The GLSL DQS path never builds a matrix — it rotates the vertex directly via the quaternion sandwich product, which sidesteps any row-major/column-major mismatch between the numpy reference and GLSL's usual column-vector convention. The LBS path does use `uniform mat4 bones[8]`, uploaded with `glUniformMatrix4fv(..., transpose=GL_TRUE)` since the CPU matrices are row-vector (translation in row 3).

## Tests

```bash
uv run pytest SkeletalAnimation/tests
```

## References

- L. Kavan, S. Collins, J. Žára, C. O'Sullivan, ["Skinning with Dual Quaternions"](https://users.cs.utah.edu/~ladislav/kavan08geometric/kavan08geometric.pdf), I3D 2007 — the DLB blending rule and the candy-wrapper name.
- [LearnOpenGL — Skeletal Animation](https://learnopengl.com/Guest-Articles/2020/Skeletal-Animation) — LBS from a loaded rig, for comparison against the procedural chain here.
- [Ben Kenwright, "A Beginners Guide to Dual-Quaternions"](https://cs.gmu.edu/~jmlien/teaching/cs451/uploads/Main/dual-quaternion.pdf) — a gentler derivation of the dual-quaternion maths than Kavan's original paper.
