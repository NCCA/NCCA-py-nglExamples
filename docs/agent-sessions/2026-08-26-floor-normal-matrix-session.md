# 2026-08-26 -- SimplePyNGL floor darkness

## Goal

Work out why the checkered floor in the SimplePyNGL demos starts off almost
black, and fix it.

## What was wrong

All three demos built the floor's normal matrix from the full MVP:

```python
mvp = self.project @ self.view @ self.mouse_global_tx @ tx
normal_matrix = Mat3.from_mat4(mvp)
normal_matrix = normal_matrix.inverse().transposed()
```

`Mat3.from_mat4` just lifts the upper-left 3x3, so the perspective matrix's
non-uniform scale comes along for the ride and skews the normal. The
inverse-transpose itself is fine -- with PyNGL's row-major upload it lands the
right way round in GLSL -- it is the matrix going in that is wrong.

The checker fragment shader in PyNGL is a bare Lambert term with no ambient and
no clamp, so brightness *is* the cosine factor:

```glsl
fragColour += checker(uv) * lightDiffuse * dot(L, N);
```

At the startup camera that came out at -0.09, which clamps to zero on write.
Hence the black floor. Using the model-view instead gives 0.95.

I checked the term over a range of orbit angles: the old code was negative at
nearly every orientation, the new code stays between 0.83 and 1.0.

One thing I did not change: `L = normalize(lightPos)` treats the light position
as a direction and compares a world-space vector against an eye-space normal.
It happens to look right here because the light and camera sit in roughly the
same direction, but it is a coincidence rather than correct lighting.

## Files changed

- `SimplePyNGL/PySideSimpleNGL.py`
- `SimplePyNGL/SDL3NGL.py`
- `SimplePyNGL/ArcBallRotation.py`
- `SimplePyNGL/PySDL3NGLDemo.png` (re-shot, the old one showed the dark floor)

## Commands run

```console
uv run PySideSimpleNGL.py --smoketest 800
uv run ArcBallRotation.py --smoketest 800
uv run SDL3NGL.py --smoketest 800
uv run ruff check SimplePyNGL/
uv run ruff format --check SimplePyNGL/
uv run pytest -q
```

All three demos smoketested, Ruff passed and 834 tests passed.
