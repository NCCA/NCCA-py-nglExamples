# SkinnedMeshImport

![](SkinnedMeshImport.png)

A PyNGL port of the `AssetImportDemos/SkeletalAnimation` demo from [NGL9Demos](https://github.com/NCCA/NGL9Demos), which itself follows [ogldev's assimp skinning tutorial](http://ogldev.atspace.co.uk/www/tutorial38/tutorial38.html). Where the original uses NGL's C++ assimp bindings, this loads the same rigged "boblampclean" guard model with [impasse](https://pypi.org/project/impasse/), the Python assimp wrapper, and does the bone-matrix skinning on the GPU. The other `SkeletalAnimation` demo in this repo is unrelated — that one is a procedural LBS-vs-DQS comparison with no mesh assets; this one is about importing a real rig.

- `mesh.py` — loads the scene via impasse, merges all sub-meshes into one indexed vertex buffer, and walks the node/animation hierarchy each frame to build the per-bone skinning matrices
- `main.py` — OpenGL, `PySide6.QtOpenGL.QOpenGLWindow` inside a `QMainWindow`, with an animation transport underneath (borrowed from `BVHViewer/timeline.py`)
- `shaders/SkinVertex.glsl` — linear blend skinning in the vertex shader, four bones per vertex
- `models/guard/` — the boblampclean mesh, animation and textures, copied from NGL9Demos

## Controls

| Key / control            | Action                                        |
| :------------------------ | :---------------------------------------------- |
| LMB / RMB / wheel        | rotate / pan / zoom                           |
| `W` / `S`                  | wireframe / solid fill                        |
| `Space`                     | reset camera                                  |
| `Esc`                       | quit                                          |
| `Cmd`/`Ctrl` + `O`, or File > Open | load a different rigged mesh                |
| Timeline transport       | play/pause, step, scrub, playback rate/range  |

File > Open isn't limited to MD5 — it'll hand anything you pick to impasse, so any rigged format assimp supports (COLLADA, FBX, glTF, ...) is worth trying. Two things it copes with rather than crashing on: a file assimp can't import at all (`impasse.errors.AssimpError`, which — unusually — subclasses `BaseException` rather than `Exception`, so it needs its own `except` clause), and a mesh whose material references a texture file that was never shipped alongside it (falls back to flat white per missing texture, logged as a warning, rather than losing the whole mesh — PyNGL's `Texture`/`Image` raises a confusing `AttributeError` for a missing file rather than a clean one). The camera also picks Z-up framing only for `.md5mesh` specifically (a fixed property of that format) and Y-up for everything else, rather than guessing from the mesh's bounding-box shape — a character posed with a limb held out (like this demo's own guard, holding its lamp arm out) can be wider than it is tall, so shape isn't a reliable signal.

## impasse has two struct bugs

Getting this working meant finding two genuine bugs in impasse 5.4.2 (the latest release on PyPI), both a mismatch between the struct layouts impasse's `cffi` bindings declare and what the installed `libassimp` (6.0.5 here) actually compiled. Both are worked around locally in `mesh.py` rather than patched in the installed package, since `uv sync` would just overwrite that.

1. **`aiBone`**: impasse declares the field order `mName, mArmature, mNode, mNumWeights, mWeights, mOffsetMatrix`. Real assimp (see `mesh.h`) has `mNumWeights` straight after `mName`, before the armature/node pointers. Reading `Bone.weights` or `Bone.offset_matrix` through impasse's own accessors walks off that offset and segfaults.
2. **`aiQuatKey`**: impasse's cdef is missing the trailing `aiAnimInterpolation mInterpolation` enum that real assimp has on both `aiVectorKey` and `aiQuatKey`. It doesn't matter for `aiVectorKey` — the struct's 20 real bytes round up to 24 for 8-byte alignment regardless, so the missing field is absorbed by padding. It does matter for `aiQuatKey`, whose 24 real bytes are already aligned: impasse computes a 24-byte element stride where the real array uses 32, so `rotation_keys[i]` for `i > 0` silently reads from the wrong offset — the first key looks fine, everything after doesn't.

Both fields are read instead by re-casting the same underlying struct pointer through a small private `cffi` definition with the corrected layout — see `_read_bone_weights_and_offset` and `_read_rotation_keys` in `mesh.py`.

## One PyNGL texture-convention gotcha, not an impasse bug

`ncca.ngl.opengl.Texture` uploads pixels in the order PIL decodes them (row 0 = the top of the image) without flipping them for OpenGL's bottom-left texture origin. Everything worked *except* the textures until this was accounted for — the model was clearly the right shape, correctly posed, just muddy and wrong-looking, which took a moment to place as a V-flip rather than something wrong with the skinning. `mesh.py` flips V itself (`1.0 - v`) when building the UV buffer, rather than in the library.

## Tests

```bash
uv run pytest SkinnedMeshImport/tests
```

Headless, no GL/Qt: they load the real model through impasse and check the maths — weights summing to one per vertex, rotation keys monotonic, the bracketing/interpolation helpers, and (the one that would have caught the composition-order bug I hit while writing this) that a bone's offset, current-pose and the root's global inverse cancel back out to the identity at the bind pose.

## References

- [ogldev — Skeletal Animation With Assimp](http://ogldev.atspace.co.uk/www/tutorial38/tutorial38.html) — the tutorial this and the original NGL demo are based on.
- [assimp — the Data Structures](https://assimp-docs.readthedocs.io/en/latest/usage/use_the_lib.html#data-structures) — `aiBone`, `aiNodeAnim` and how the node/bone hierarchy fits together.
- [impasse on PyPI](https://pypi.org/project/impasse/)
