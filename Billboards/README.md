# Billboards

![](Billboards.png)

_(screenshot still to add — TODO for Jon)_

Camera-facing quads: the trick behind particles, sprites, impostors and HUD markers. Thirty textured sprites are scattered around a teapot (the depth reference — it never billboards, so you've always got something solid to judge distance against). Press `M` to cycle three ways of orienting the sprites and watch the same 30 quads go from obviously wrong to always correct.

## Controls

| Key               | Action                                                       |
| :----------------- | :------------------------------------------------------------ |
| `M`               | cycle billboard mode — fixed world-space / cylindrical / spherical |
| `B`               | toggle alpha blending (sorted back-to-front) vs. alpha-tested cutout |
| LMB / RMB / wheel | rotate / pan / zoom, `Space` resets, `Esc` quits             |

## The three modes

1. **Fixed world-space** — the quad's right/up vectors never change. Orbit the scene and at some angles you're looking at the sprite edge-on: it thins to a sliver and vanishes. This is what "just draw a textured quad" gets you without billboarding at all — the whole point of the demo is to show why the other two modes exist.
2. **Cylindrical** — `up` is locked to world +y (trees, lampposts, anything that should stay upright), `right` is rebuilt from the camera direction every frame. Robust to orbiting sideways; drag vertically to pitch the view and it visibly tips, because a cylindrical billboard has no sensible answer for "which way is sideways" when you're looking straight down its locked axis.
3. **Spherical** — both `right` and `up` are rebuilt from the camera every frame. Always face-on, from any angle. This is the one particle systems and impostors actually use.

## The maths (`billboard_maths.py`)

Numpy only, no GL/Qt — see `Billboards/tests/test_billboard_maths.py` for the tests this was built against test-first.

The repo's row-vector convention (`RayPickingSelection/picking_maths.py`, the design spec) means a point transforms as `row_vector @ matrix`. For a view matrix built the way `ncca.ngl.look_at` builds one, that convention puts the camera's world-space **right** axis in column 0 of the view matrix, its world-space **up** axis in column 1, and its world-space **backward** axis (target back towards the eye) in column 2 — equivalently, rows 0..2 of `view.T`. `spherical_basis(view)` is just reading columns 0 and 1 straight back out:

```python
right = view[:3, 0] / norm
up    = view[:3, 1] / norm
```

`cylindrical_basis(view)` keeps `up` pinned to world +y and rebuilds `right` as `cross(world_up, backward)`, where `backward` is column 2 of the same matrix. When the camera looks straight up or down, `backward` is parallel to world +y and the cross product collapses to zero — there's no well-defined "sideways" left. Rather than divide by zero and hand back NaN, `cylindrical_basis` falls back to a fixed world +x for `right` in that case (documented in the function's docstring, pinned by `test_degenerate_straight_down_view_has_no_nan`); real engines hit the same wall, a tree billboard genuinely has no good answer viewed from directly above.

`main.py` doesn't feed the raw camera view into these functions, though — it passes the *combined* `view @ global_tx` (the same matrix that also carries the teapot and the scatter positions through the mouse-drag rotation), and builds the quad's vertices in local, pre-`global_tx` space. Because `global_tx` is a pure rotation (orthonormal), reapplying it on the GPU exactly cancels the correction baked into `right`/`up`, so the spherical mode faces the camera correctly no matter how far you've orbited — this is what makes mode 3 provably always-correct rather than "correct until you rotate too far". Fixed world-space mode skips the correction entirely (`right = (1,0,0)`, `up = (0,1,0)` verbatim), which is exactly why it breaks.

`back_to_front` mirrors `Blending/blend_scene.py`'s helper — sort billboard centres by view-space depth (ascending = furthest first) so `B` composites transparent sprites correctly with the OVER operator.

## Design choices

- **CPU rebuilds the quad VBO every frame.** With only ~30 billboards there's no reason to push per-draw `centre`/`size`/`right`/`up` uniforms and expand a unit quad in the vertex shader — the CPU just writes six vertices (two triangles) per sprite straight into a dynamic `VertexData` buffer each frame, and the shader (`shaders/BillboardVertex.glsl`) does nothing but `MVP * position`. Fine for a teaching demo at this scale; wouldn't scale to a real particle system, which is exactly the point of the following [BoidsCompute](../BoidsCompute) demo.
- **Non-standard vertex layout.** The dynamic VBO only carries position (location 0, vec3) and UV (location 1, vec2) — there's no normal, so it doesn't match `ncca.ngl` Primitives' fixed 0/1/2 (vert/normal/uv) layout. That's fine because this VAO is never shared with a `Primitives`-created one.
- **Procedural sprite texture.** A 64×64 soft radial-gradient RGBA glow, generated in numpy at startup and uploaded with `glTexImage2D` — no binary asset in the repo.

## Tests

```bash
uv run pytest Billboards/tests
```

## References

- [LearnOpenGL — Billboarding](https://learnopengl.com/) forum/wiki discussions of spherical vs. cylindrical billboards are the standard reference for this technique; see also any particle-system chapter in *Real-Time Rendering*.
- [OpenGL Wiki — Billboards, Cheating](https://www.khronos.org/opengl/wiki/Billboards_Cheating) — the classic overview of the technique this demo walks through.
