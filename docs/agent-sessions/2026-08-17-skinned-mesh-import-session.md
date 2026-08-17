# 2026-08-17 session: SkinnedMeshImport demo (impasse skeletal animation)

## Goal

Port `AssetImportDemos/SkeletalAnimation` from NGL9Demos to PyNGL: import
the rigged "boblampclean" guard model with `impasse` (Python assimp
bindings) and skin it on the GPU with linear blend skinning, with a
desktop animation transport for playback. `SkeletalAnimation` was already
taken in this repo (a procedural LBS/DQS comparison with no mesh assets),
so the new demo is `SkinnedMeshImport`.

## Files changed

New demo folder `SkinnedMeshImport/`:

- `mesh.py` — loads the scene via `impasse.load()`, merges all sub-meshes
  into one indexed vertex/index buffer (baking each sub-mesh's vertex
  offset into its indices), and walks the node/animation hierarchy per
  frame to build per-bone skinning matrices. Works around two impasse
  library bugs (see below).
- `main.py` — `QMainWindow` hosting a `QOpenGLWindow` viewport
  (`createWindowContainer`) plus a copy of `BVHViewer/timeline.py`'s
  `TimelineWidget` for play/pause/scrub/fps/range transport. Standard
  arcball mouse controls via `PySideEventHandlingMixin`.
- `MultiBufferIndexVAO.py` — copied from
  `VertexArrayObject/ExtendedVAOFactory/`; bone IDs go in as a `vec4`
  (float, cast to `int()` in GLSL) rather than adding an integer vertex
  attribute path for one demo.
- `shaders/SkinVertex.glsl`, `shaders/SkinFragment.glsl` — linear blend
  skinning + Blinn-Phong, adapted from the C++ original's shaders.
- `models/guard/` — mesh, animation and textures copied from
  `NGL9Demos/AssetImportDemos/Models/guard`.
- `tests/test_mesh_maths.py` — headless, loads the real model through
  impasse; checks weight sums, rotation-key monotonicity, the
  interpolation helpers, and a bind-pose round-trip identity check.
- `README.md`, `SkinnedMeshImport.png`.

Root `README.md` — added `SkinnedMeshImport` to the Animation table.
`pyproject.toml` / `uv.lock` — added the `impasse` dependency (committed
to `Version1.0` separately, before the worktree was created, per the
"branch must be clean before editing" rule).

## Two impasse bugs found and worked around

impasse 5.4.2 (the latest release on PyPI) has two struct-layout bugs
relative to the installed `libassimp` 6.0.5 (confirmed against
`/opt/homebrew/include/assimp/mesh.h` / `anim.h`). Neither is fixable by
using the library differently — both are worked around locally by
re-casting the same underlying struct pointer through a private `cffi`
definition with the corrected layout, rather than patching the installed
package (which `uv sync` would just overwrite).

1. **`aiBone`**: impasse orders the fields `mName, mArmature, mNode,
   mNumWeights, mWeights, mOffsetMatrix`; real assimp has `mNumWeights`
   immediately after `mName`, before the armature/node pointers. Reading
   `Bone.weights` or `Bone.offset_matrix` through impasse's own accessors
   reads across that offset and segfaults.
2. **`aiQuatKey`**: impasse's cdef is missing the trailing
   `aiAnimInterpolation mInterpolation` enum that real assimp has on both
   `aiVectorKey` and `aiQuatKey`. It doesn't bite `aiVectorKey` (its 20
   real bytes round up to a 24-byte stride for alignment either way, so
   the missing field is absorbed by padding — position/scaling keys read
   fine). It does bite `aiQuatKey`: its 24 real bytes are already
   aligned, so impasse computes a 24-byte stride where the real array
   uses 32. `rotation_keys[0]` reads correctly regardless (offset zero),
   which is why a quick single-key smoke check looked fine; every key
   after that read garbage, non-monotonic times, which only showed up as
   the animation visibly falling apart partway through playback.

## Other bugs hit during development (all fixed, all covered by the tests
or caught visually)

- Matrix-composition order: PyNGL's `Mat4` is row-vector (`A @ B` applies
  A then B, per `mat_base.py`'s own docstring), and assimp matrices are
  column-vector with translation in the last column, not the last row.
  Getting the conversion direction and the hierarchy composition order
  wrong the first time produced a fully shattered mesh, not a subtly
  wrong pose — a brute-force search over the 4 plausible
  order/inverse-placement combinations against the "offset, bind-pose
  global, global-inverse should cancel to the identity" invariant found
  the right one immediately (now `test_bind_pose_round_trip_is_identity`).
- impasse's key sequences don't support negative indexing (`keys[-1]`
  raises `IndexError`) — the end-of-animation fallback branch needed
  `keys[len(keys) - 1]` instead.
- `Mat4 @ Vec3` isn't valid (only `Mat4 @ Mat4` or `Mat4 @ Vec4`) — the
  light-position-to-eye-space transform needed a `Vec4`.
- The mesh's animation/bounding-box data is Z-up (MD5/idTech convention);
  the camera needed `up = (0, 0, 1)` and a front-on eye position, not
  NGL's usual Y-up setup.
- Loading the mesh lazily in `initializeGL` and trying to signal
  `MainWindow` via `QOpenGLWindow.frameSwapped` once it was ready didn't
  fire reliably through `createWindowContainer` — the timeline was built
  but the playback timer never started. Fixed by loading the mesh (pure
  Python/impasse, no GL context needed) eagerly in `SkinViewport.__init__`
  instead, so `MainWindow.__init__` can wire up the timeline directly.

## Commands run

```bash
uv run pytest SkinnedMeshImport/tests -v   # 8 passed
uv run ruff check SkinnedMeshImport/       # All checks passed!
uv run ruff format --check SkinnedMeshImport/  # 5 files already formatted
uv run SkinnedMeshImport/main.py --smoketest 1500 --debug   # SMOKETEST OK
```

Visual verification: ran the demo live and screenshotted it at multiple
points across the full 140-frame animation (0, 45, 56, 84, 106, 139) to
confirm the pose stays coherent throughout, not just at the frame or two
a quick smoke test would happen to catch.

## Follow-up: File > Open, tested against other meshes

Jon asked for a File > Open dialog so other rigged meshes could be tried,
not just the bundled guard model. Added `SkinViewport.load_model()` (tears
down the old VAO/textures, loads the new `SkinnedMesh`, rebuilds) and a
File menu (`Cmd/Ctrl+O`) in `main.py`. Testing it against the other
rigged assets sitting in `NGL9Demos/AssetImportDemos/Models/` (bobWalk.dae,
horse-3ds.3ds, JaiquaFromXSI.dae, TurbochiFromXSI.dae) surfaced four more
real bugs, all fixed:

- **`impasse.errors.AssimpError` subclasses `BaseException`, not
  `Exception`.** The dialog's `except Exception` silently failed to catch
  a genuinely-broken file (`bobWalk.dae` — assimp itself can't import it,
  unrelated to this demo) and crashed instead of showing the intended
  error dialog. Needed `except (Exception, AssimpError)` explicitly.
- **A mesh with a missing texture file crashed instead of loading.**
  `JaiquaFromXSI.dae` references `jaiqua_red21a_fix.jpg`, which was never
  shipped alongside it in NGL9Demos. `ncca.ngl.opengl.Texture`/`Image`
  raises `AttributeError: 'Image' object has no attribute '_width'` for a
  missing file rather than something catchable/clean. `_load_textures`
  now catches any texture-load failure per-submesh and substitutes a 1x1
  white fallback texture, logging a warning, rather than losing the whole
  mesh load.
- **Camera framing was hardcoded to MD5's Z-up convention.** Loading
  `JaiquaFromXSI.dae` (Y-up, like most non-MD5 formats) rendered sideways.
  Tried a "tallest bbox axis = up" heuristic first — it's wrong: the
  guard model's own arm-out lamp pose makes X (85 units, arm span) taller
  than Z (67, actual height), so bbox shape isn't a reliable signal.
  Switched to keying off the file extension instead: Z-up only for
  `.md5mesh` specifically (a fixed, known property of that format),
  Y-up for everything else.
- **`impasse.helper.get_bounding_box` computes a *static* scene-graph
  bbox, not the actually-rendered (bone-skinned) one.** For MD5 the mesh
  node and skeleton root happen to be coincident so this went unnoticed;
  for the XSI-exported DAE they're not, and the static bbox put the
  camera looking at empty space (right up-axis, wrong position/distance
  entirely — only legs on screen, badly cropped). Replaced with
  `SkinnedMesh.skinned_positions()`/`bounding_box()`, which runs the same
  LBS blend as the vertex shader in numpy and takes min/max over the
  actual result.

Also, while re-screenshotting for the README with a working "other mesh"
to compare against: the guard model's own textures turned out to be
V-flipped the whole time (muddy/wrong-looking but not obviously upside
down, since the mesh shape and skinning were already correct — easy to
miss). `ncca.ngl.opengl.Texture` uploads pixels top-row-first without
flipping for OpenGL's bottom-left origin; `mesh.py` now flips V itself
(`1.0 - v`) when building the UV buffer.

Commands re-run after all of the above: `pytest` (8 passed), `ruff check`
and `ruff format --check` (clean), plus live visual checks of both the
guard model and `JaiquaFromXSI.dae` loaded via the new Open dialog.
