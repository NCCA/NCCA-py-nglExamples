# BVHViewer

This is a PyNGL / PySide6 port of the C++ NGL BVH viewer. It plays `.bvh`
motion-capture files as a skeleton of spheres and cylinders, with a GUI much
closer to the animation tools I normally use in Maya or Houdini.

Run it from the root of the demos repository:

```bash
uv run BVHViewer/main.py
```

There is also a WebGPU version. It has the same timeline, menus, camera controls
and Four Views layout, but renders the ground, traces and skeleton with WebGPU:

```bash
uv run BVHViewer/main_webgpu.py
```

The File menu loads another BVH clip. You can also select the initial file from
the command line:

```bash
uv run BVHViewer/main.py --bvh BVHViewer/bvh/test.bvh
```

The bottom panel shows the current frame, clip range and playback rate. Drag the
orange timeline handle to scrub, type a frame into the frame field, or use the
transport buttons to jump, step and play. Manual frame changes pause playback.

The second orange bar sets the playback range. Drag either end of the bar or
type the start and end frames into the fields. Playback and the transport
controls then stay inside that range. The FPS field changes the playback speed
without changing the motion data.

| Key | Does |
| --- | --- |
| Space | play / pause, or maximize the pane under the mouse in Four Views |
| Home / End | first / last frame |
| ← / → | previous / next frame |
| T | toggle trace mode (draw a different coloured motion path for each joint) |
| 4 | toggle the Top, Perspective, Front and Side layout |
| W / A / S / D | move the camera |
| F | fullscreen |

Drag with the left mouse button to look around and use the wheel to change the
field of view. Four Views in the View menu splits the viewport into the usual
Maya layout: the left mouse still only rotates the Perspective pane, but each
of Top, Front and Side now has its own zoom and pan -- wheel over a pane to
zoom just that one, middle-drag to pan it, or right-drag with a mouse. A
two-finger click and drag does the same thing on a Mac trackpad. Space maximizes
whichever pane the mouse is over to fill the window, and space again puts the
four-way layout back; move the mouse over the timeline first if you actually
want play/pause.
The status bar shows the loaded filename, current frame and play state. Trace
mode keeps the floor and animated skeleton visible under the joint paths.

## How it works

- `bvh.py` -- parses the file and holds playback state (`current_frame`,
  `step_forward`/`step_backward`/`replay`/`advance`). No Qt, no OpenGL, so it's
  headlessly testable.
- `bvh_scene.py` -- the drawing: walks the joint tree each frame, building a
  sphere at every joint and a cylinder for every bone. Trace mode samples the
  joint positions into NumPy arrays and draws them as coloured VAO line strips.
- `timeline.py` -- the scrubber, draggable playback range, numeric frame/FPS
  fields and transport controls.
- `main.py` -- the application window, first-person and orthographic cameras,
  file menu, playback timer and OpenGL viewport.
- `main_webgpu.py` -- the WebGPU viewport and application entry point.
- `bvh_scene_webgpu.py` and `webgpu_renderer.py` -- build the WebGPU line and
  instance data, then draw the skeleton with the shader in `bvh_webgpu.wgsl`.
- `tests/` -- pytest, run with `uv run pytest BVHViewer/tests` from the root of
  the repository.

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
