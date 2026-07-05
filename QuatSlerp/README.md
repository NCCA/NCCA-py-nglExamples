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
