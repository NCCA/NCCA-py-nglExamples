# Smoketest + argparse Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every executable demo entry script supports `--smoketest [MS]` (default 200) and, for Qt demos, `--debug`, parsed with argparse.

**Architecture:** Purely mechanical per-file edits to the `__main__` / `main()` blocks of 82 standalone demo scripts, applying one of four canonical patterns (Qt `__main__`, Qt `main()`, GUI/QML, SDL3). No shared code is introduced — each demo stays self-contained per repo convention.

**Tech Stack:** Python argparse, PySide6 (QTimer), wgpu-py demos are Qt-hosted so same pattern, PySDL3 (SDL_GetTicks), uv, ruff.

**Spec:** `docs/superpowers/specs/2026-07-13-smoketest-argparse-design.md`

## Global Constraints

- Work in worktree `.worktrees/smoketest-argparse`, branch `agent/smoketest-argparse`. Never commit to `main`/`Version1.0`.
- `--smoketest` is `nargs="?", const=200, default=None, type=int, metavar="MS"`. `--smoketest` alone = 200 ms; `--smoketest 1000` = 1000 ms. On expiry print exactly `SMOKETEST OK` and exit 0.
- `--debug` (Qt demos only, including all WebGPU demos — they are PySide6-hosted): `action="store_true"`, selects the demo's `DebugApplication`.
- Do NOT touch `RunDemos.py` or anything under `Obj2Numpy/`.
- Remove every old-style `"--smoketest" in sys.argv` / `"--debug" in sys.argv` check when converting a file — no mixed styles.
- Preserve each demo's existing behaviour and structure (QSurfaceFormat setup, window size, try/except blocks, `main()` vs `__main__` layout). Only the argument handling and smoketest wiring change.
- Verification for every touched file runs **from the file's own directory** (demos load shaders/textures by relative path) and **on-screen** — macOS's Qt `offscreen` platform cannot create OpenGL contexts, and macOS has no `timeout(1)`. Use the helper:
  `.superpowers/sdd/smoketest_all.sh <file> [<file> ...]` → prints `PASS <file>` per file (checks for `SMOKETEST OK`, 90 s perl-alarm watchdog). Record any FAIL in the final report — do not hide it.
- After each task: `uv run ruff check <files>` and `uv run ruff format --check <files>` clean (pre-commit runs these anyway).
- Conventional commit per task.

## Canonical Patterns

Every task below applies one of these patterns. Repeat: copy the code as-is, adapting only names that differ (window class, sizes, existing app subclass).

### Pattern A — Qt demo with `if __name__ == "__main__":` block

Applies whether or not the file already has a `DebugApplication`. If it does not (files are flagged per-task), first add this class above the `__main__` block (identical to `Camera/main.py`; needs `import traceback` and `logger` in the ncca.ngl import — add `logger` only if already available, otherwise drop the `logger.info` line):

```python
class DebugApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise
```

Then replace the argument/app/smoketest handling in `__main__` (keep the existing QSurfaceFormat block and window setup untouched):

```python
import argparse  # add to imports at top of file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest",
        nargs="?",
        const=200,
        default=None,
        type=int,
        metavar="MS",
        help="run for MS milliseconds (default 200), print SMOKETEST OK and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run with DebugApplication (tracebacks from Qt event handlers)",
    )
    args = parser.parse_args()

    # ... existing QSurfaceFormat setup stays exactly as it was ...

    if args.debug:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    window = MainWindow()  # keep existing constructor args/size
    window.resize(1024, 720)  # keep existing size
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
```

`QTimer` import: ensure `from PySide6.QtCore import QTimer` is present (extend the existing QtCore import line).

### Pattern B — Qt demo with `def main():` entry point

Same parser, placed at the top of `main()`. Most of these are WebGPU demos without a `DebugApplication`; add the Pattern A class above `main()`. Files with an existing `try/except` around window creation keep it.

```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest",
        nargs="?",
        const=200,
        default=None,
        type=int,
        metavar="MS",
        help="run for MS milliseconds (default 200), print SMOKETEST OK and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run with DebugApplication (tracebacks from Qt event handlers)",
    )
    args = parser.parse_args()

    if args.debug:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)
    win = WebGPUScene()  # keep existing class/size
    win.resize(1024, 720)
    win.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
```

### Pattern C — GUI/QML demos

Same as Pattern B, but these apps have extra setup (QQuickWindow graphics API, QQuickStyle, organization names) that must stay **before** `QApplication` construction, exactly where it is now. If the demo constructs a plain `QApplication`, add the standard `DebugApplication` class; if it already subclasses `QApplication` for its own reasons, add the `notify` try/except wrapping to that subclass under `--debug` is NOT needed — instead keep the existing app class for normal runs and use `DebugApplication` (standard class added to the file) when `--debug` is passed.

### Pattern D — SDL3 demos (no Qt)

Parser has `--smoketest` only (no `--debug`). Wire a tick-based timer into the existing main loop:

```python
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest",
        nargs="?",
        const=200,
        default=None,
        type=int,
        metavar="MS",
        help="run for MS milliseconds (default 200), print SMOKETEST OK and exit",
    )
    args = parser.parse_args()

    # ... existing SDL init / window / context creation unchanged ...

    running = True
    event = sdl3.SDL_Event()
    start_ticks = sdl3.SDL_GetTicks()
    while running:
        while sdl3.SDL_PollEvent(event):
            running = scene.handle_event(event)
        scene.update()
        scene.render()
        sdl3.SDL_GL_SwapWindow(window)
        if (
            args.smoketest is not None
            and sdl3.SDL_GetTicks() - start_ticks >= args.smoketest
        ):
            print("SMOKETEST OK")
            running = False

    # ... existing cleanup unchanged ...
```

### Verification loop (used by every task)

For each file in the task, from the worktree root:

```bash
d=$(dirname <file>); f=$(basename <file>)
(cd "$d" && QT_QPA_PLATFORM=offscreen timeout 60 uv run "$f" --smoketest)
```

Expected: last line `SMOKETEST OK`, exit code 0. For one file per task also verify the duration argument works: `... --smoketest 500`.

---

### Task 1: Migrate the 19 scripts that already have old-style smoketest checks

**Files (Modify):**
Pattern A (have `DebugApplication`, old `"--smoketest" in sys.argv` and `"--debug" in sys.argv` checks to remove):
- `AnimatedTextures/main.py`
- `Camera/main.py`
- `ColourObj/main.py`
- `CurveDemos/main.py`
- `EasingFunctions/main.py`
- `FrustumCull/main.py`
- `GeometryTessellation/normals_main.py`
- `GeometryTessellation/tess_main.py`
- `ImageHeightMap/main.py`
- `Interpolation/main.py`
- `KleinBottle/main.py`
- `PointCloud/main.py`
- `PostProcessChain/main.py`
- `QuatSlerp/main.py`
- `ShadowMapping/main.py`
- `SkyBoxEnvMap/main.py`
- `UBOStorageBuffers/main.py`

Pattern B (main()-style WebGPU, no `DebugApplication` — add the class):
- `SkyBoxEnvMap/SkyBoxEnvMapWebGPU.py`
- `UBOStorageBuffers/StorageWebGPU.py`

**Interfaces:** Produces the reference implementations of Pattern A and Pattern B that later tasks copy.

- [ ] **Step 1:** Apply Pattern A to the 17 `__main__`-style files: add `import argparse`, replace the `smoketest = "--smoketest" in sys.argv` / `if "--debug" in sys.argv:` logic with the parser block, keep `DebugApplication` selection via `args.debug`, replace the hard-coded `QTimer.singleShot(200, ...)` with `args.smoketest`.
- [ ] **Step 2:** Apply Pattern B to the 2 WebGPU files, adding the standard `DebugApplication` class and `import traceback`.
- [ ] **Step 3:** Run the verification loop over all 19 files; also `Camera/main.py --smoketest 500` to confirm the duration argument.
- [ ] **Step 4:** `uv run ruff check <files> && uv run ruff format <files>`.
- [ ] **Step 5:** Commit: `refactor: migrate existing smoketest demos to argparse with duration`

### Task 2: OpenGL `__main__` demos with existing DebugApplication — batch 1

**Files (Modify, all Pattern A, `DebugApplication` already present):**
- `2DDrawingOpenGL/2dDrawing.py`
- `2DDrawingOpenGL/PanZoom2D.py`
- `BlankPySide6NGL/main.py`
- `BlankPySide6NGL/using_mixin.py`
- `Blending/main.py`
- `ColourSelectionOpenGL/main.py`
- `FBODemos/Blit/main.py`
- `FBODemos/DOF/main.py`
- `FBODemos/SimpleFBO/main.py`
- `FontRendering/main.py`
- `Lights/main.py`
- `NormalMapping/NormalMapping.py`
- `OITransparency/main.py`
- `OpenGLPrimRestart/FastPyNGLVersion.py`
- `OpenGLPrimRestart/FasterVersion.py`
- `OpenGLPrimRestart/PrimRestartLine.py`

**Interfaces:** Consumes Pattern A from Task 1.

- [ ] **Step 1:** Apply Pattern A to each file (add `import argparse`, ensure `QTimer` import, parser block, `args.debug` app selection replacing any existing `"--debug" in sys.argv` check, smoketest QTimer).
- [ ] **Step 2:** Verification loop over all 16 files; one file with `--smoketest 500`.
- [ ] **Step 3:** `uv run ruff check <files> && uv run ruff format <files>`.
- [ ] **Step 4:** Commit: `feat: add argparse smoketest/debug options to OpenGL demos (batch 1)`

### Task 3: OpenGL `__main__` demos with existing DebugApplication — batch 2

**Files (Modify, all Pattern A, `DebugApplication` already present):**
- `PBR/PBRTexture/main.py`
- `PBR/SimplePBR/main.py`
- `RayPickingSelection/main.py`
- `SciFiUI/main.py`
- `ScreenTri/ScreenTri.py`
- `SelectionManipulator/main.py`
- `ShadingModels/main.py`
- `ShowMipmap/ShowMipmap.py`
- `SimplePyNGL/ArcBallRotation.py`
- `SimplePyNGL/PySideSimpleNGL.py`
- `SimpleTexture/Texture.py`
- `VAOPrimitives/main.py`
- `VertexArrayObject/ChangingVAO/main.py`
- `Voxels/main.py`

**Interfaces:** Consumes Pattern A from Task 1.

- [ ] **Step 1:** Apply Pattern A to each file.
- [ ] **Step 2:** Verification loop over all 14 files; one file with `--smoketest 500`.
- [ ] **Step 3:** `uv run ruff check <files> && uv run ruff format <files>`.
- [ ] **Step 4:** Commit: `feat: add argparse smoketest/debug options to OpenGL demos (batch 2)`

### Task 4: OpenGL demos without DebugApplication (add the class)

**Files (Modify, Pattern A + add standard `DebugApplication` class and `import traceback`):**
- `Particles/ParticleQuads/main.py`
- `VertexArrayObject/Boid/main.py`
- `VertexArrayObject/BoidShaded/main.py`
- `VertexArrayObject/ChangingVAOMultiBuffer/main.py`
- `VertexArrayObject/ExtendedVAOFactory/main.py`
- `VertexArrayObject/MultiBufferVAO/main.py`
- `VertexArrayObject/SimpleIndexVAOFactory/main.py`
- `VertexArrayObject/Sphere/main.py`

**Interfaces:** Consumes Pattern A + DebugApplication class from Task 1.

- [ ] **Step 1:** Add the standard `DebugApplication` class (with `import traceback`; include the `logger.info` line only if `logger` is already imported from `ncca.ngl`, otherwise omit it) above `__main__` in each file, then apply Pattern A.
- [ ] **Step 2:** Verification loop over all 8 files; one file with `--smoketest 500`.
- [ ] **Step 3:** `uv run ruff check <files> && uv run ruff format <files>`.
- [ ] **Step 4:** Commit: `feat: add argparse smoketest/debug options to VAO and particle demos`

### Task 5: ObjViewer — argparse with positional model/texture arguments

**Files (Modify):** `ObjViewer/ObjViewer.py`

**Interfaces:** Consumes Pattern A. Current hand-rolled logic: `--debug` removed from `sys.argv` manually; then `len(sys.argv) == 3` → obj + texture, `== 2` → obj + default texture `textures/ratGrid.png`, else defaults `models/Helix.obj` + `textures/helix_base.tif`.

- [ ] **Step 1:** Replace the hand-rolled block with the standard parser plus:

```python
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="path to an OBJ model (default: models/Helix.obj)",
    )
    parser.add_argument(
        "texture",
        nargs="?",
        default=None,
        help="path to a texture image (default: textures/ratGrid.png, or the helix texture with the default model)",
    )
    args = parser.parse_args()

    if args.model is not None:
        oname = args.model
        tname = args.texture if args.texture is not None else "textures/ratGrid.png"
    else:
        oname = "models/Helix.obj"
        tname = "textures/helix_base.tif"
```

Keep app construction (`DebugApplication` if `args.debug`) and the smoketest QTimer per Pattern A.
- [ ] **Step 2:** Verify: default run (`--smoketest`), with model arg (`(cd ObjViewer && QT_QPA_PLATFORM=offscreen uv run ObjViewer.py models/Helix.obj --smoketest)`), and `--smoketest 500`.
- [ ] **Step 3:** `uv run ruff check ObjViewer/ObjViewer.py && uv run ruff format ObjViewer/ObjViewer.py`.
- [ ] **Step 4:** Commit: `feat: use argparse for ObjViewer options and smoketest`

### Task 6: WebGPU `main()`-style demos — batch 1

**Files (Modify, Pattern B; add `DebugApplication` class unless flagged as already present):**
- `2DDrawingOpenGL/WebGPU2D.py`
- `BlankWebGPU/main.py`
- `Blending/BlendingWebGPU.py`
- `DefferedLighting/SimpleWebGPU.py`
- `FBODemos/WebGPURenderToTexture/main.py`
- `OITransparency/OITWebGPU.py`
- `SelectionManipulatorWebGPU/main.py`
- `SimpleComputeWebGPU/WebGPU2D.py`
- `SimpleTexture/TextureWebGPU.py`
- `SimpleWebGPU/SimpleWebGPU.py`

**Interfaces:** Consumes Pattern B from Task 1.

- [ ] **Step 1:** Apply Pattern B to each file (parser at top of `main()`, add `DebugApplication` class + `import traceback` + `import argparse`, ensure `QTimer` import, smoketest QTimer before `app.exec()`).
- [ ] **Step 2:** Verification loop over all 10 files (WebGPU: if offscreen fails, retry on-screen and note it); one file with `--smoketest 500`.
- [ ] **Step 3:** `uv run ruff check <files> && uv run ruff format <files>`.
- [ ] **Step 4:** Commit: `feat: add argparse smoketest/debug options to WebGPU demos (batch 1)`

### Task 7: WebGPU `main()`-style demos — batch 2

**Files (Modify, Pattern B):**
- `WebGPUCompute/SpatialHash2D/WebGPU2D.py` (has `DebugApplication`)
- `WebGPUCompute/SpatialHash2D/WebGPU2DGui.py`
- `WebGPUCompute/SpatialHash3D/WebGPU3D.py` (has `DebugApplication`)
- `WebGPUCompute/SpatialHash3D/WebGPU3DGui.py`
- `WebGPUComputePicking/main.py`
- `WebGPUMultiGeo/WebGPUMultiGeo.py`
- `WebGPUShadows/PCFShadows.py`

**Interfaces:** Consumes Pattern B from Task 1.

- [ ] **Step 1:** Apply Pattern B to each file (add `DebugApplication` where missing).
- [ ] **Step 2:** Verification loop over all 7 files; one file with `--smoketest 500`.
- [ ] **Step 3:** `uv run ruff check <files> && uv run ruff format <files>`.
- [ ] **Step 4:** Commit: `feat: add argparse smoketest/debug options to WebGPU demos (batch 2)`

### Task 8: GUI/QML demos

**Files (Modify, Pattern C):**
- `GUIDemos/NGLWidgetsOpenGL/main.py`
- `GUIDemos/PySideGUIOpenGL/main.py`
- `GUIDemos/QMLOverlayApp/main.py`
- `GUIDemos/QMLWebGPUOverlay/main.py`
- `GUIDemos/WebGPUGUI/main.py`

**Interfaces:** Consumes Pattern C. Constraint reminder: `QQuickWindow.setGraphicsApi`, `QQuickStyle.setStyle`, `setOrganizationName`/`setApplicationName`, and `QSurfaceFormat` setup keep their current order relative to app construction.

- [ ] **Step 1:** Apply Pattern C to each file: parser at top of `main()`, add standard `DebugApplication` class, select it under `--debug`, keep all existing pre-app setup in place, add smoketest QTimer before `app.exec()` (inside the existing `try` block where one exists).
- [ ] **Step 2:** Verification loop over all 5 files (QML demos may not support `offscreen` — retry on-screen and note); one file with `--smoketest 500`.
- [ ] **Step 3:** `uv run ruff check GUIDemos && uv run ruff format GUIDemos`.
- [ ] **Step 4:** Commit: `feat: add argparse smoketest/debug options to GUI/QML demos`

### Task 9: SDL3 demos

**Files (Modify, Pattern D):**
- `BlankPySDL3/main.py`
- `SimplePyNGL/SDL3NGL.py`

**Interfaces:** Consumes Pattern D. No `--debug` for these.

- [ ] **Step 1:** Apply Pattern D to both files (`import argparse`, parser with `--smoketest` only, `SDL_GetTicks` deadline check at the end of the main loop).
- [ ] **Step 2:** Verify both: `(cd BlankPySDL3 && timeout 60 uv run main.py --smoketest)` → `SMOKETEST OK` exit 0 (SDL has no offscreen platform plugin; runs on-screen briefly). Also `--smoketest 500` on one.
- [ ] **Step 3:** `uv run ruff check <files> && uv run ruff format <files>`.
- [ ] **Step 4:** Commit: `feat: add argparse smoketest option to SDL3 demos`

### Task 10: Full-repo verification sweep

**Files:** none created (report only; fix regressions in the files above if found).

- [ ] **Step 1:** Run the verification loop over **all 82 files** in one pass and collect results:

```bash
cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/smoketest-argparse
fail=0
while read f; do
  d=$(dirname "$f"); b=$(basename "$f")
  if (cd "$d" && QT_QPA_PLATFORM=offscreen timeout 60 uv run "$b" --smoketest 2>/dev/null | grep -q "SMOKETEST OK"); then
    echo "PASS $f"
  else
    echo "FAIL $f"; fail=1
  fi
done < /tmp/exec_scripts.txt
exit $fail
```

(Regenerate `/tmp/exec_scripts.txt` with: `find . -name "*.py" -perm +111 -not -path "*/.venv/*" -not -path "*/.git/*" -not -path "*/tests/*" -not -name "RunDemos.py" -not -path "*/Obj2Numpy/*" | sed 's|^\./||' | sort`.)
- [ ] **Step 2:** For each FAIL, retry on-screen (drop `QT_QPA_PLATFORM=offscreen`). Fix real regressions; record genuine platform limitations in the final report.
- [ ] **Step 3:** `uv run ruff check . && uv run ruff format --check .` clean.
- [ ] **Step 4:** `uv run pytest` — all existing tests pass (they are headless maths tests; unaffected, but confirm).
- [ ] **Step 5:** Commit any fixes: `fix: address smoketest regressions found in verification sweep`
