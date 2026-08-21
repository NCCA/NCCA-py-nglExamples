# Core Demos Phase 7 Implementation Plan

Phase 7 is the final group in the core demo roadmap: `ResetLine`, `MorphObj`
and `OctreeAbstract`. Each demo gets an OpenGL and WebGPU version. The shared
Python modules hold the geometry or simulation rules so the two renderers use
the same data.

## Task 1: ResetLine

- Add tests for the seeded blade field, primitive-restart indices, WebGPU line
  expansion and animation.
- Port the OpenGL primitive-restart draw to `ResetLine/main.py`.
- Add `ResetLine/main_webgpu.py`; WebGPU has no primitive restart, so expand
  each blade into independent line-list segments before drawing.
- Keep `A` to pause the wind and Space to reset the view.

## Task 2: MorphObj

- Add tests for OBJ topology checks, base-plus-delta packing, weight clamping
  and the short punch animation.
- Copy the three Bruce pose OBJ files from the C++ demo.
- Add matching OpenGL and WebGPU vertex shaders which blend the same packed
  pose data on the GPU.
- Keep `Q`/`W` and `A`/`S` for manual weights, with `Z` and `X` for the two
  punch animations.

## Task 3: OctreeAbstract

- Add tests for perfect-tree subdivision, overlapping leaf membership,
  duplicate-free collision pairs, wall reflection and particle collisions.
- Replace the C++ template with a normal Python `Octree` class. The particle
  simulation stays on the CPU, which is the point of this example.
- Render the particles as instanced spheres in both backends and draw the
  simulation bounds as a line box.
- Use a 10 by 10 by 10 particle grid by default. `--grid 20` reproduces the
  original 8000-particle setup when required.

## Task 4: Documentation and checks

- Add one README and preview per demo, plus rows in the root catalogue.
- Run the focused and full pytest suites, ruff check and format checks, all six
  smoketests and `uv build`.
- Export the Codex session, add the session summary and commit with conventional
  commit messages on `agent/core-demos-phase7`.
