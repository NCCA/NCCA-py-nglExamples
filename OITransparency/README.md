# Weighted Blended Order-Independent Transparency

![](OIT.png)

Two transparent panels intersect in an X through an opaque teapot — a case no per-object sort can ever draw correctly — alongside ordinary front/back panels. Keys `1`/`2`/`3` switch between naive alpha blending, per-object sorting, and weighted blended OIT (McGuire & Bavoil, [JCGT 2013](http://jcgt.org/published/0002/02/09/)) to compare all three on the same scene. This is the MSc follow-on to the [Blending](../Blending) demo.

There are two versions:

- `main.py` — OpenGL, using an MRT FBO and `glBlendFunci` (a different blend function per attachment)
- `OITWebGPU.py` — WebGPU, where the per-attachment blends become per-target `blend` states on the pipeline

The scene and the OIT maths are defined once in `oit_common.py` — a numpy-only reference implementation of exactly what the shaders compute, unit tested in `tests/` (including the key property: the composite is invariant under every permutation of fragment order).

## Controls

| Key | Action |
| :-- | :-- |
| `1` | naive alpha blend in scene order (wrong wherever order is wrong) |
| `2` | per-object back-to-front sort (fixes the parallel panels; still wrong along the intersection of the X pair) |
| `3` | weighted blended OIT — order independent, no sorting at all |
| `A` / `Z` | increase / decrease panel alpha |
| LMB / RMB / wheel | rotate / pan / zoom, `Space` resets, `Esc` quits |

## How it works

The OVER operator is order dependent, but a *sum* and a *product* are not. Weighted blended OIT replaces the sorted composite with two commutative accumulators, rendered in three passes:

1. **Opaque pass** → opaque colour texture + depth texture.
2. **Accumulation pass** — all transparent geometry in *any* order, depth tested against (but never writing) the opaque depth, into two render targets with different blend functions:
   - `accum` (RGBA16F, blend `ONE, ONE`): accumulates `weight(z, a) * vec4(rgb * a, a)` — a weighted sum of premultiplied colour
   - `reveal` (R16F, blend `ZERO, ONE_MINUS_SRC_COLOR`): accumulates `Π(1 - aᵢ)` — the total transmittance
3. **Composite pass** — a full-screen triangle resolves the weighted average and lerps it over the opaque colour:

   ```glsl
   vec3 transparent = accum.rgb / max(accum.a, 1e-5);
   colour = opaque * reveal + transparent * (1.0 - reveal);
   ```

The depth-dependent weight makes near fragments dominate the average, *approximating* what a correct sort would have produced. It is an approximation — with very high alpha or extreme depth ranges it drifts from ground truth (try `A` to push alpha up) — but it is a single geometry pass, needs no sorting, and handles intersecting geometry that defeats sorting entirely.

## Tests

```bash
uv run pytest OITransparency/tests
```
