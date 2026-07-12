# ParticleQuads

A WebGPU particle system where each particle is rendered as a camera-facing
billboard quad (rather than a point). The `Emitter` class updates 50,000
particles on the CPU with vectorized NumPy (position, velocity, colour, life
and alive/dead state), and the quads are expanded and orientated towards the
camera in the vertex shader. Particles can be drawn as soft circles or
squares.

## Files

- `main.py` - scene, camera and render pipeline
- `Emitter.py` - NumPy particle emitter (birth, update, recycle)
- `particle_shader.wgsl` - billboard quad vertex / fragment shader
- `WebGPUWidget.py` - Qt widget hosting the WebGPU surface

## Controls

- Left-drag : look around (first person camera)
- Arrow keys : move the camera
- `space` : toggle animation
- `c` : toggle circle / square particles
- `u` : single-step the emitter and print debug info
- `Esc` : quit

## References

- W. T. Reeves, "Particle Systems — A Technique for Modeling a Class of Fuzzy Objects", ACM TOG 1983 — [ACM](https://dl.acm.org/doi/10.1145/357318.357320) — the original particle-system paper (emission, lifetime, stochastic variation).
- [opengl-tutorial — Billboards](http://www.opengl-tutorial.org/intermediate-tutorials/billboards-particles/billboards/) — expanding a point into a camera-facing quad, done here in the vertex shader.
- [WebGPU Fundamentals — Storage Buffers](https://webgpufundamentals.org/webgpu/lessons/webgpu-storage-buffers.html) — feeding the per-particle data the quads are pulled from.
