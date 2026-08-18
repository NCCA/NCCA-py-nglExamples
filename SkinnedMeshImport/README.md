# SkinnedMeshImport

![](SkinnedMeshImport.png)

A PyNGL port of the `AssetImportDemos/SkeletalAnimation` demo from [NGL9Demos](https://github.com/NCCA/NGL9Demos), itself following [ogldev's assimp skinning tutorial](http://ogldev.atspace.co.uk/www/tutorial38/tutorial38.html). Where the original uses NGL's C++ assimp bindings, I load the same rigged "boblampclean" guard model with [impasse](https://pypi.org/project/impasse/), the Python assimp wrapper, and do the bone-matrix skinning on the GPU. There's another `SkeletalAnimation` demo in this repo, unrelated to this one -- that one is a procedural LBS-vs-DQS comparison with no mesh assets, whereas this is about importing a real rig.

```bash
uv run SkinnedMeshImport/main.py
```

There is also a WebGPU version. It has the same timeline, camera and
four-view controls, and File > Open, but skins and renders the mesh with
WebGPU instead:

```bash
uv run SkinnedMeshImport/main_webgpu.py
```

- `mesh.py` -- loads the scene via impasse, merges all sub-meshes into one indexed vertex buffer, and walks the node/animation hierarchy each frame to build the per-bone skinning matrices
- `main.py` -- OpenGL, `PySide6.QtOpenGL.QOpenGLWindow` inside a `QMainWindow`, with an animation transport underneath (borrowed from `BVHViewer/timeline.py`) and a `FirstPersonCamera` / four-view split borrowed from `BVHViewer/main.py`'s `BvhViewport`
- `timeline.py` -- the scrubber and transport controls, same file as BVHViewer's
- `MultiBufferIndexVAO.py` -- a small `AbstractVAO` subclass for a mesh that needs several vertex buffers plus a separate index buffer, which the stock VAO classes in `ncca.ngl` don't cover
- `shaders/SkinVertex.glsl` -- linear blend skinning in the vertex shader, four bones per vertex
- `models/guard/` -- the boblampclean mesh, animation and textures, copied from NGL9Demos
- `main_webgpu.py` -- the WebGPU viewport and application entry point, mirroring `BVHViewer/main_webgpu.py`'s split from its OpenGL `main.py`
- `webgpu_renderer.py` -- the WebGPU pipeline: five vertex buffers, an index buffer, a bone storage buffer, and one texture bind group per submesh
- `skin_webgpu.wgsl` -- the same four-bones-per-vertex linear blend skin and Blinn-Phong lighting as `shaders/SkinVertex.glsl` / `SkinFragment.glsl`, ported to WGSL

## Controls

| Key / control                      | Action                                       |
| :--------------------------------- | :------------------------------------------- |
| `W`/`A`/`S`/`D`                    | fly the camera forward/left/back/right       |
| LMB-drag                           | look around                                  |
| wheel                              | zoom (perspective pane) / scale (ortho pane) |
| MMB/RMB-drag in an ortho pane      | pan that pane                                |
| `4`, or View > Four Views          | toggle the TOP/PERSPECTIVE/FRONT/SIDE split  |
| click a pane in four-view          | maximize it; click again to restore          |
| `Cmd`/`Ctrl` + `O`, or File > Open | load a different rigged mesh                 |
| Timeline transport                 | play/pause, step, scrub, playback rate/range |

The camera setup mirrors `BVHViewer` exactly: a `FirstPersonCamera` for the perspective pane, three independent `OrthoView` panes with their own pan/zoom state for TOP/FRONT/SIDE, `glScissor`-clipped rendering into a 2x2 split, and click-to-maximize. One wrinkle specific to this demo: `FirstPersonCamera`'s yaw/pitch always treats **Y** as vertical regardless of the `up` vector passed to its constructor, but MD5 (idTech) meshes come out Z-up in post-import world space -- every other format this demo loads ends up Y-up. Rather than teach the camera two conventions, I key this off the `.md5mesh` extension itself, since that's a fixed property of the format, and give a Z-up mesh one constant `rotate_x(-90)` model matrix in `_draw_mesh` to present it to the camera (and to lighting) already in Y-up space. Guessing from the bounding box instead would have been a mistake -- a character posed with a limb held out, like this demo's own guard holding its lamp arm out, can end up wider than it is tall.

File > Open isn't limited to MD5 -- it hands whatever you pick straight to impasse, so any rigged format assimp supports (COLLADA, FBX, glTF, ...) is worth trying. Two failure modes it copes with rather than crashes on: a file assimp can't import at all (`impasse.errors.AssimpError`, which -- unusually -- subclasses `BaseException` rather than `Exception`, so it needs its own `except` clause), and a mesh whose material references a texture file that was never shipped alongside it (falls back to flat white per missing texture, logged as a warning, rather than losing the whole mesh -- PyNGL's `Texture`/`Image` otherwise raises a fairly confusing `AttributeError` for a missing file).

## impasse has two struct bugs

Getting this working meant finding two bugs in impasse 5.4.2 (the latest release on PyPI), both a mismatch between the struct layouts impasse's `cffi` bindings declare and what the installed `libassimp` (6.0.5 here) actually compiled. I've worked around both locally in `mesh.py` rather than patching the installed package, since `uv sync` would just overwrite that.

1. **`aiBone`**: impasse declares the field order `mName, mArmature, mNode, mNumWeights, mWeights, mOffsetMatrix`. Real assimp (see `mesh.h`) puts `mNumWeights` straight after `mName`, before the armature/node pointers. Reading `Bone.weights` or `Bone.offset_matrix` through impasse's own accessors walks off that offset and segfaults.
2. **`aiQuatKey`**: impasse's cdef is missing the trailing `aiAnimInterpolation mInterpolation` enum that real assimp has on both `aiVectorKey` and `aiQuatKey`. It doesn't matter for `aiVectorKey` -- the struct's 20 real bytes round up to 24 for 8-byte alignment regardless, so the missing field just gets absorbed by padding. It does matter for `aiQuatKey`, whose 24 real bytes are already aligned: impasse computes a 24-byte element stride where the real array uses 32, so `rotation_keys[i]` for `i > 0` silently reads from the wrong offset -- the first key looks fine, everything after it doesn't.

I read both fields instead by re-casting the same underlying struct pointer through a small private `cffi` definition with the corrected layout -- see `_read_bone_weights_and_offset` and `_read_rotation_keys` in `mesh.py`.

## Differences from the OpenGL version

**No bone-count ceiling.** The OpenGL shader's `gBones[128]` is a fixed-size
GLSL uniform array, so `main.py` warns and leaves extra bones unanimated
past `MAX_BONES = 128`. `skin_webgpu.wgsl` reads the bone palette from a
WGSL storage buffer instead (`var<storage, read> bones: array<mat4x4<f32>>`),
sized to the mesh's actual bone count -- there's no equivalent cap here.

**A second V-flip.** `mesh.py` flips every UV's V coordinate once
(`1.0 - v`), because `ncca.ngl.opengl.Texture` uploads pixels in PIL's
top-row-first order without flipping for OpenGL's bottom-left texture
origin. WebGPU's texture origin is top-left, so that first flip is wrong
here -- `skin_webgpu.wgsl`'s fragment shader flips V back a second time
(`vec2(uv.x, 1.0 - uv.y)`) rather than giving the shared, backend-agnostic
loader a second UV buffer.

## Tests

```bash
uv run pytest SkinnedMeshImport/tests
```

Headless, no GL/Qt: they load the real model through impasse and check the maths -- weights summing to one per vertex, rotation keys monotonic, the bracketing/interpolation helpers, and (the one that would have caught the composition-order bug I hit while writing this) that a bone's offset, current-pose and the root's global inverse cancel back out to the identity at the bind pose.

## References

- [ogldev -- Skeletal Animation With Assimp](http://ogldev.atspace.co.uk/www/tutorial38/tutorial38.html) -- the tutorial this and the original NGL demo are based on.
- [assimp -- the Data Structures](https://assimp-docs.readthedocs.io/en/latest/usage/use_the_lib.html#data-structures) -- `aiBone`, `aiNodeAnim` and how the node/bone hierarchy fits together.
- [impasse on PyPI](https://pypi.org/project/impasse/)
