# BvhViewer (PyNGL)

A PyNGL / PySide6 port of the C++ NGL [BvhViewer](../) in this repo, which plays
back `.bvh` motion-capture files as a skeleton of spheres and cylinders. Same
sample files, same camera, same controls.

To run it:

```bash
cd pyngl
uv run main.py
```

By default it loads `../bvh/Male1_B10_WalkTurnLeft45.bvh`; pass `--bvh path/to/file.bvh`
for any of the others in [`../bvh`](../bvh).

Left mouse rotates, right mouse pans, wheel zooms.

| Key | Does |
| --- | --- |
| R | replay from frame 0 |
| P | pause / continue |
| ← / → | step one frame back / forward (works while paused) |
| Space | clear the character from the scene |
| T | toggle trace mode (stop clearing the framebuffer, for a motion-trail look) |
| W / S | wireframe / filled |
| F | fullscreen |
| Esc | quit |

The window title doubles as the HUD (filename, frame number, play state), so
it's readable even on a machine with none of the handful of system fonts this
tries for the optional on-screen overlay text.

## How it works

- `bvh.py` -- parses the file and holds playback state (`current_frame`,
  `step_forward`/`step_backward`/`replay`/`advance`). No Qt, no OpenGL, so it's
  headlessly testable.
- `bvh_scene.py` -- the drawing: walks the joint tree each frame, building a
  sphere at every joint and a cylinder for every bone.
- `main.py` -- the window, camera and playback timer.
- `tests/test_bvh.py` -- pytest, run with `uv run pytest tests/`.

## Differences from the C++

The C++ `Bvh::load()` hand-rolls a `std::stack<Joint*>` to track hierarchy
depth while reading the file line by line. The `.bvh` hierarchy block is
really just an S-expression with `{`/`}` as the parens, so here it's a
recursive-descent parser over a flat token stream instead, and Python's own
call stack does the depth tracking the C++ had to manage by hand.

PyNGL's stock cylinder runs along **+Y**; the old NGL's ran along **+Z**, which
is what the C++ `getRotationFromZ` was aligning bones to. The equivalent here
is `rotation_from_y`, and it's built from `Quaternion.from_axis_angle` rather
than the C++'s own hand-written Rodrigues-formula matrix, since PyNGL already
has that constructor.

A `.bvh` joint can declare its rotation channels in any of the six axis
orders, and that order changes the resulting pose. The C++ tracks this with a
per-joint `m_rotate_order` index array. PyNGL's `Transform` already has this
as `rot_order`, covering all six permutations, so `Bvh.local_matrix` reads the
declared channel order straight off the joint -- but has to **reverse** it
before handing it to `Transform.set_order`. A BVH file's declared order is a
physical application order (first-declared channel rotates about the
original axis first, the next about the now-rotated axis, and so on --
intrinsic composition); PyNGL's `rot_order` table composes the other way,
with the *last* letter of its order string applied first. Declared
`Zrotation Xrotation Yrotation` therefore needs `Transform.set_order("yxz")`,
not `"zxy"`. Getting this backwards doesn't error -- it still produces a
rigid, animated skeleton, just the wrong pose (worked out the hard way:
compared this parser's output against the independent
[`bvh`](https://pypi.org/project/bvh/) PyPI package on one of the sample
files, joint by joint, until they matched).

The C++'s `m_heightOfRoot` didn't carry over as-is. It walks the rest pose
looking for the lowest cumulative Y-offset under the root and subtracts it
from the root's height every frame -- clearly meant to keep the character's
feet near the ground, but the sign is backwards (it pushes the root *up* by
that amount, not down), and the original commit that introduced the drawing
code called it out as broken. Skipping it entirely isn't right either,
though: this file's root translation channels put the hips around Y=20 with
the feet never coming within 10 units of the ground, and start the walk
33 units off to the side of the camera's look-at point, in a completely
different corner of world space to the ground grid. `Bvh._compute_root_offset`
fixes both, computed once at load time: X/Z is just frame 0's root
translation negated, so the walk starts centred on the origin; Y walks the
rest pose (offsets only, no rotation) to find the lowest point -- typically a
foot or toe -- and grounds *that* to Y=0, the same idea as the C++ but with
the correct sign, and folded in with the frame-0 cancellation rather than
applied as a separate step. It's a constant per-file shift, so the
animation's own motion (walking, turning, any vertical bob) is untouched;
only the starting point moves.

Multiple named walls with arbitrary normals didn't carry over either. The
C++ `Scene` supports a list of them, but the app only ever adds one flat
ground plane at start-up. That generality was never exercised, so `BvhScene`
just draws a single hardcoded ground grid.

## References

The original demo, which this is a port of:

- [`../`](../) -- the C++ NGL version in this repo.

BVH and the maths this build on:

- [BVH file format (Motion Capture Society, via CMU)](https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/BVH.html)
  -- the HIERARCHY/MOTION grammar this parser reads.
- [Rotation matrix -- Wikipedia](https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle)
  -- the axis-angle construction behind `rotation_from_y`, same reference the
  C++ `getRotationFromZ` cited.
- [Euler angles -- Wikipedia](https://en.wikipedia.org/wiki/Euler_angles) --
  why a joint's declared rotation channel order changes its pose.
