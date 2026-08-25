# Blending and Transparency

![](Blending.png)

Five transparent panels straddle an opaque teapot and the different modes can be toggled from the keyboard.

- `main.py` — OpenGL, where blending is _dynamic_ state (`glEnable(GL_BLEND)`, `glBlendFunc`, `glDepthMask`)
- `BlendingWebGPU.py` — WebGPU, where the blend state is _baked into the render pipeline_, so each toggle switches between pre-built pipeline variants

Both render the same scene, defined once in `blend_scene.py` along with the depth-sorting maths (numpy only, unit tested in `tests/`).

## Controls

| Key               | Action                                                                            |
| :---------------- | :-------------------------------------------------------------------------------- |
| `B`               | toggle blending — without it the panels are simply opaque                         |
| `D`               | toggle depth _write_ for the panels — writing depth punches holes in later panels |
| `O`               | toggle back-to-front sorting — the OVER operator is order dependent               |
| `F`               | cycle the blend function: over / additive / premultiplied / multiply              |
| `A` / `Z`         | increase / decrease panel alpha                                                   |
| LMB / RMB / wheel | rotate / pan / zoom, `Space` resets, `Esc` quits                                  |

## Things to know

1. **Transparency is a framebuffer operation, not a material property.** The fragment shader just writes an alpha value; the blend equation `result = src * srcFactor + dst * dstFactor` decides what it means.
2. **The OVER operator is order dependent**, so transparent geometry must be drawn after all opaque geometry, sorted back to front. Rotate the camera 180° with sorting off (`O`) and watch the composite break.
3. **Transparent geometry is depth _tested_ but not depth _written_.** Turn depth write on (`D`) with sorting off and near panels will occlude far ones entirely instead of blending.
4. **Additive blending is order independent** (a sum commutes) — cycle to it with `F` and note that sorting stops mattering. This observation is the seed of the [OITransparency](../OITransparency) demo, which extends it into a general order-independent technique.

Sorting _per object_ also breaks down entirely once transparent objects intersect — see [OITransparency](../OITransparency) for that case and its fix.

## Tests

```bash
uv run pytest Blending/tests
```

## References

- T. Porter & T. Duff, "Compositing Digital Images", SIGGRAPH 1984 — [ACM](https://dl.acm.org/doi/10.1145/800031.808606) — the OVER operator and the alpha channel.
- [LearnOpenGL — Blending](https://learnopengl.com/Advanced-OpenGL/Blending) — blend functions, sorting and the depth-write rule demonstrated by the toggles.
- [Alpha Compositing (Bartosz Ciechanowski)](https://ciechanow.ski/alpha-compositing/) — an interactive walkthrough of OVER, premultiplied alpha and compositing maths.
- [OpenGL Wiki — Blending](https://www.khronos.org/opengl/wiki/Blending) — the full blend equation/factor reference behind `glBlendFunc`.
