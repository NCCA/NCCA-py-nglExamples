# PyNGL Project Briefs

Project ideas for undergraduate and postgraduate animation and graphics students,
built around the demos in this repository. Every brief names one or more existing
demos as the mandated starting point, so students begin from working code rather
than a blank window.

## Brief index

| Level | API | File | Projects |
|-------|-----|------|----------|
| Undergraduate | OpenGL | [undergraduate-opengl.md](undergraduate-opengl.md) | 8 |
| Undergraduate | WebGPU | [undergraduate-webgpu.md](undergraduate-webgpu.md) | 5 |
| Postgraduate | OpenGL | [postgraduate-opengl.md](postgraduate-opengl.md) | 7 |
| Postgraduate | WebGPU | [postgraduate-webgpu.md](postgraduate-webgpu.md) | 7 |

## Common requirements (all projects)

- **Starting templates**: use `BlankPySide6NGL` (OpenGL), `BlankWebGPU` (WebGPU),
  or `BlankPySDL3` as the application skeleton, plus the demos named in the brief.
- **Version control**: work in a git repository with regular, meaningful commits.
- **Evaluation**: every submission must include a short evaluation section —
  frame-time measurements, parameter studies, or comparison of alternative
  techniques. The Qt UI in the templates makes live instrumentation easy;
  a simple on-screen frame-time readout is the minimum.
- **Report**: a concise write-up covering design decisions, the mathematics or
  algorithms used, known limitations, and references.
- **Attribution**: any third-party assets (models, textures, papers implemented)
  must be credited.

## Suggested marking weighting

| Component | UG | PG |
|-----------|----|----|
| Core technical implementation | 45% | 35% |
| Software design and code quality | 20% | 15% |
| Evaluation and analysis | 15% | 30% |
| Report and presentation | 10% | 10% |
| Stretch goals / novelty | 10% | 10% |

## Choosing a project

- Students stronger on **animation** should look at the curve, keyframe, IK,
  skeletal, and crowd briefs.
- Students stronger on **rendering** should look at the deferred/PBR,
  path-tracing, shadow, and post-processing briefs.
- Students stronger on **simulation** should look at the particle, boids,
  cloth, and fluid briefs.
- A low-risk but legitimate PG option is **port and compare**: several demos
  exist in both APIs (particles, shadows, render-to-texture), and the
  cross-API renderer brief formalises this.
