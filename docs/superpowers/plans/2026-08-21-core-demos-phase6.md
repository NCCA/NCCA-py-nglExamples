# Core Demos Phase 6 Implementation Plan

Phase 6 is `ImageMaze`, roadmap row 12. I am porting the original C++ demo to
OpenGL and WebGPU whilst keeping the same small teaching example: load an image,
draw every non-white pixel as a coloured cube, then move an actor through the
white pixels with the arrow keys.

The image and actor rules live in `ImageMaze/maze_scene.py`. This keeps image
coordinates, world positions and collision tests in one place so both renderers
show the same maze. The small WebGPU vertex conversion lives in
`ImageMaze/mesh_data.py`; it turns the existing PyNGL cube and troll data into
position-and-colour buffers without putting GPU code into the shared scene.

## Task 1: Shared maze rules and tests

- Add tests for wall extraction, colour conversion, actor movement, blocked
  moves, world positions and vertex conversion.
- Run the tests first and confirm they fail because the modules do not exist.
- Implement the smallest shared modules needed to make the tests pass.

## Task 2: OpenGL demo

- Add `ImageMaze/main.py` with the overhead and actor cameras from the source.
- Draw coloured wall cubes, the red troll actor and the grey ground plane.
- Keep the original controls: arrows move, `1`/`2` select a camera, `W` toggles
  wireframe, Space resets the mouse transform and Escape quits.
- Add `--map`, `--smoketest` and `--debug` command-line options.

## Task 3: WebGPU demo

- Add `ImageMaze/main_webgpu.py` and `ImageMaze/ImageMazeShader.wgsl`.
- Reuse the same maze and actor state, with one static wall draw plus separate
  actor and ground draws.
- Match the OpenGL controls and command-line options.

## Task 4: Assets, documentation and checks

- Copy `small.png`, `colour.png` and `test.png` from the C++ demo.
- Add the demo README, root catalogue row, preview and session summary.
- Run pytest, ruff, both smoketests and the repository build.
- Commit the finished demo on `agent/core-demos-phase6` using conventional
  commit messages. This work remains local as requested.
