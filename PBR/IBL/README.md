# IBL — Image Based Lighting

![](IBL.png)
*(screenshot TODO — cannot be captured headlessly, see below)*

Completes the PBR story started in [`PBR/SimplePBR`](../SimplePBR): that demo's
direct-only Cook-Torrance lighting used a flat `vec3(0.03) * albedo * ao` for
ambient, with a comment saying "the next IBL tutorial will replace this". This
demo is that replacement — the sphere grid's ambient term comes from the
procedural sky in [`SkyBoxEnvMap`](../../SkyBoxEnvMap) (its `cubemap_gen.py`,
copied here since demo folders stay standalone), precomputed into a diffuse
irradiance map and a BRDF split-sum LUT.

## Why precompute on the CPU

The textbook approach renders the scene six times into an FBO per cubemap
face to build the irradiance/prefilter chain on the GPU every launch. That's
a lot of render-to-cubemap plumbing for a teaching demo. Since the source
environment here is a small procedural sky (not a huge HDRI), the whole
precompute — both stages below — runs in well under a second in numpy and is
cached to `.npy` files next to the script, so it only actually computes once.
This also means the maths is plain, testable Python: see
`tests/test_ibl_precompute.py`.

## What's implemented (and what's cut)

The brief lists four precompute stages. Two shipped, one was cut for time —
this is the largest of the six new demos in the batch, and the brief
explicitly allows shipping a smaller-but-working subset over a half-finished
larger one:

| Stage | Status | Notes |
| --- | --- | --- |
| 1. Irradiance map | **shipped** | 16²-per-face cubemap, cosine-weighted hemisphere convolution of the source sky |
| 2. Prefiltered specular chain | **cut** | roughness-mipped GGX importance-sampled cubemap chain — see below |
| 3. BRDF LUT | **shipped** | 256² RG16F split-sum `A`/`B` table, pure numpy GGX integral |
| 4. Shader ambient term | **shipped (approximated)** | see below |

**Stage 2 cut**: the prefiltered specular mip chain (source blurred
per-roughness across 5 mip levels) was not built. `shaders/IBLFragment.glsl`
samples the *unfiltered* base cubemap directly along the reflection vector
instead, so specular reflections stay sharp at every roughness value rather
than blurring out the way a real prefiltered environment would. Everything
else in the split-sum term — the Fresnel-weighted `A`/`B` scale/bias from the
real LUT, the real irradiance map for diffuse — is unapproximated. There is
no test for the prefilter chain, because there is no code for it.

## The split-sum approximation

Karis's split-sum trick (see references) factors the environment specular
integral into two independently-precomputable pieces so it can be evaluated
per-pixel at runtime with two texture fetches instead of an integral:

```
ambient = kD * irradiance(N) * albedo         // diffuse term (real)
        + prefiltered(R, roughness) * (F * A + B)   // specular term (A/B real, prefiltered() approximated -- see above)
```

`irradiance(N)` and the `(A, B)` LUT are precomputed in `ibl_precompute.py`;
`prefiltered(R, roughness)` is approximated by the base environment cubemap
(no mip chain, see the cut above).

## Controls

| Key | Effect |
| --- | --- |
| `I` | toggle IBL ambient vs. SimplePBR's flat direct-only ambient (the money shot) |
| `E` | cycle debug view: off / irradiance-as-skybox / BRDF LUT (corner overlay) |
| LMB | rotate |
| RMB | pan |
| Wheel | zoom |
| Space | reset camera |
| Esc | quit |

The 7×7 sphere grid sweeps metallic (rows) against roughness (columns), the
same layout `PBR/SimplePBR` uses.

## Sandbox note

`--smoketest` is implemented (argparse flag, `QTimer.singleShot` → print
`SMOKETEST OK` → quit) but this sandbox's offscreen Qt platform plugin cannot
create a real OpenGL context — `QT_QPA_PLATFORM=offscreen` segfaults here on
first paint, the same pre-existing limitation confirmed on the `Blending`
reference demo. Run it normally (`uv run PBR/IBL/main.py --smoketest`) on a
machine with a real GL context to verify.

## References

- LearnOpenGL — [Diffuse Irradiance](https://learnopengl.com/PBR/IBL/Diffuse-irradiance)
  and [Specular IBL](https://learnopengl.com/PBR/IBL/Specular-IBL) — the
  derivations this demo's precompute and shader ambient term are ported from.
- B. Karis, "Real Shading in Unreal Engine 4", SIGGRAPH 2013 course notes —
  [course page](https://blog.selfshadow.com/publications/s2013-shading-course/)
  — the split-sum approximation and the Hammersley/GGX importance-sampling
  BRDF LUT derivation.
- [LearnOpenGL — PBR Theory](https://learnopengl.com/PBR/Theory) and
  [PBR Lighting](https://learnopengl.com/PBR/Lighting) — the direct-lighting
  Cook-Torrance term this demo extends, shared with `PBR/SimplePBR`.
