# QuatSlerp

Slerps between two quaternion orientations (built from Euler-angle rotation
matrices) and shows the interpolated teapot (centre) alongside the start
(left) and end (right) orientations. A side panel (matching the original
NGL9Demos Qt Designer UI) lets you edit the start/end Euler rotations, drag
the interpolation slider, and read out the start/end/interpolated
quaternions and the resulting rotation matrix.

## Controls
Side panel : edit start/end rotation, drag the interpolate slider
`w`/`s` : wireframe / solid fill (in the 3D viewport)
Left-drag : orbit, Right-drag : pan, Wheel : zoom, `space` : reset (in the 3D viewport)

## References

- K. Shoemake, "Animating Rotation with Quaternion Curves", SIGGRAPH 1985 — [ACM](https://dl.acm.org/doi/10.1145/325165.325242) — the paper that introduced slerp to computer graphics.
- [Quaternion (songho.ca)](https://www.songho.ca/math/quaternion/quaternion.html) — quaternion algebra and rotation-matrix conversion.
- [Interpolating rotations with SLERP (John D. Cook)](https://www.johndcook.com/blog/2023/03/15/slerp/) — a short, clear derivation of the slerp formula.
