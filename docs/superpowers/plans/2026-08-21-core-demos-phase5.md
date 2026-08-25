# Core Demos Phase 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port NGL9Demos' "Input handling" demo — `GameKeyControl` (roadmap row 11, ported from the C++ `AdvancedGameKeyControl` source, which supersedes the plain `GameKeyControl`) — to PyNGLDemos, with an OpenGL entry point and a WebGPU entry point. A yellow spaceship is driven around a fixed 2D play area with held arrow keys combined via a bitmask, its motion looked up in a precomputed 32-entry table (one entry per possible simultaneous key combination); the session can be recorded to a `.kp` file and played back.

**Architecture:** One flat demo folder, `GameKeyControl/`, no `Parent/<SubDemo>/` nesting (matches `TexelFetch/`/`LoadShaderFromJSon/` precedent — single-topic folder). Three files: a shared pure-Python module (`game_controls.py`: the key-bitmask constants, the 32-entry motion lookup table, ship movement/transform maths, and the `KeyRecorder` record/playback class — all backend-agnostic, no GL/wgpu import), `main.py` (OpenGL), and `main_webgpu.py` (WebGPU). Unlike every demo ported so far this wave, GameKeyControl needs **no reinterpretation** for its core teaching point — bitmask-indexed lookup tables and file-based record/playback are pure Python/Qt concerns identical on both backends. The only backend-specific work is drawing the ship mesh and the HUD text, both of which this repo already has ready-made building blocks for: OpenGL's built-in flat `DefaultShader.COLOUR` shader (a byte-for-byte match for the C++'s `nglColourShader`) plus `ncca.ngl.opengl.Text`, and WebGPU's `WebGPUWidget.render_text()` HUD helper plus a small hand-rolled flat-colour WGSL shader.

- **Shared module (`game_controls.py`):** `GameControls` bitflags (`UP`/`DOWN`/`LEFT`/`RIGHT`/`ROTATE`), the `MOTION_TABLE` (32 `(offset_x, offset_y, rotation)` entries transcribed verbatim from the C++ `g_motionTable`, including its "nonsense" opposite-key-cancels-out entries), `move_ship()` (position delta + clamp-to-extents + rotation accumulation, pure), `ship_transform()` (position+Y-rotation → `Mat4`, pure, shared by both backends), and `KeyRecorder` (frame list + start position, `.kp` text-file save/load, identical file format to the C++ so a recording is interchangeable between this port and a hypothetical fresh session). Pytest-covered, same pattern as `Collisions/collision_maths.py` and `ViewToWorldTransform`'s pure-math core.
- **OpenGL (`main.py`):** `QOpenGLWindow`, **no mouse camera control** (the C++ source has none — this demo is purely keyboard-driven with a fixed camera, a deliberate, faithful omission of the usual arcball mixin). Loads `models/SpaceShip.obj` via `ncca.ngl.Obj`/`create_vao()`, draws it with the built-in `DefaultShader.COLOUR` (flat, unlit — `outColour = Colour`, no lighting math, confirmed from the library's own GLSL source) set once to yellow, exactly matching the C++. Two independent `QTimer`s (15 ms ship-update/record/playback tick, 30 ms redraw tick — the C++ genuinely decouples these; preserved, not collapsed into this repo's more common single-timer habit). HUD text via `ncca.ngl.opengl.Text`.
- **WebGPU (`main_webgpu.py`):** Same shared module, same two-timer split, same fixed camera. Ship mesh: since `ncca.ngl.Obj.create_vao()` is GL-only and there's no numpy-array-only accessor, a small local helper (`ship_mesh.py`) replicates `BaseMesh.create_vao()`'s triangulation/interleave loop (same `[x,y,z,nx,ny,nz,u,v]` layout every other WebGPU demo in this repo already uses) to build a vertex buffer directly — the same kind of demo-local workaround already established for `PrimData.sphere()` bypassing `Primitives.create()`. A small hand-rolled flat WGSL shader (`GameKeyControlShader.wgsl`, MVP transform + a flat uniform colour, no lighting — mirroring `nglColourShader`'s genuinely unlit nature, not the lit gizmo shader style seen in `AffineTransforms`) draws it. HUD text via the `WebGPUWidget.render_text()` base-class helper (already used by `Instancing/InstancingWebGPU.py`). Record/playback and the `.kp` file dialogs are **not** reinterpreted — identical Qt/file-I/O logic to the GL side, reusing the same shared module.

**Tech Stack:** Python 3.13, `ncca.ngl` (local editable package at `/Users/jmacey/teaching/Code/PyNGL`), PySide6, PyOpenGL, wgpu-py, numpy, `uv run --script`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-core-demos-roadmap-design.md` (Phase 5 = roadmap row 11: `GameKeyControl`; grouping section "5. Input handling").

## Global Constraints

- No edits to `/Users/jmacey/teaching/Code/PyNGL` — where `ncca.ngl` has no matching helper (a numpy-only accessor for `Obj`'s interleaved vertex data), a small demo-local module replicates the missing piece, matching this repo's established practice of not modifying the library for a one-demo need (the `PrimData.sphere()`-bypasses-`Primitives.create()` precedent).
- Work happens in branch `agent/core-demos-phase5`, worktree at `.worktrees/core-demos-phase5` (already created).
- Every entry script (`main.py`, `main_webgpu.py`) starts `#!/usr/bin/env -S uv run --script`, is `chmod +x`, and supports `--smoketest` (via `argparse`, `nargs="?", const=200, default=None, type=int`) using the `QTimer.singleShot(...)` pattern from `VAOPrimitives/main.py` (OpenGL) or `Blending/BlendingWebGPU.py`'s `main()` (WebGPU).
- **No mouse camera control on either backend** — this is the one demo in the whole wave without it. `main.py` is a plain `class MainWindow(QOpenGLWindow)` (NOT `PySideEventHandlingMixin`), `main_webgpu.py` is a plain `class WebGPUScene(WebGPUWidget)` with no mouse handlers. Camera is fixed: `look_at(Vec3(0, 0, 80), Vec3(0, 0, 0), Vec3(0, 1, 0))`, `perspective(45.0, w/h, 0.05, 350.0)` (GL) / `perspective(45.0, w/h, 0.05, 350.0, PerspMode.WebGPU)` (WebGPU), set once and left untouched (aside from re-deriving the aspect ratio in `resizeGL`/`resizeWebGPU`, matching every other demo's resize handling). This is a direct, verbatim match to the C++ `NGLScene::initializeGL`/`resizeGL`.
- **Two independent `QTimer`s, not one** — `ship_timer` at 15 ms (drives `move_ship()`, and while recording/playing back, advances/consumes one `KeyRecorder` frame per tick) and `redraw_timer` at 30 ms (calls `self.update()` only). This is a genuine, deliberate C++ behavior (sampling input faster than the screen redraws) — do not collapse it into a single timer even though nearly every other demo in this repo uses one; preserving it is a case of the wave's fidelity policy overriding this repo's incidental single-timer habit. `closeEvent` must stop **both** timers before `super().closeEvent(event)`.
- Background clear colour is **black** (`0, 0, 0, 1`), matching the C++ exactly — most demos in this repo use a grey `(0.4, 0.4, 0.4, 1.0)` clear colour; this one deliberately does not.
- Key handling is **held-state**, not single-press-toggle: `keyPressEvent` sets a bit in a `keys_pressed` byte, `keyReleaseEvent` clears it (`Up`/`Down`/`Left`/`Right`/`R`). This is different from the single-keypress-toggle pattern used by every other demo's `W`/`S`/`F`/`N` conventions in this repo — GameKeyControl has none of those (no wireframe toggle, no fullscreen key in the C++ source, so none are added here either). `Space` toggles recording (and captures the ship's current position as the recording's start position). `P` toggles playback. `S` opens a save-`.kp` file dialog, `L` opens a load-`.kp` file dialog (`QFileDialog`, `*.kp` filter, matching the C++ exactly). `Escape` quits.
- The `.kp` file format is preserved exactly, byte-for-byte compatible in spirit with the C++'s text format: line 1 = frame count, line 2 = `"x y z"` (space-separated start position floats), then one decimal integer per line, one per recorded frame. Implemented once in the shared module, used identically by both backends — a recording made in the GL entry point loads and plays back correctly in the WebGPU entry point and vice versa.
- **Adaptation, not a fidelity break:** the C++ `KeyRecorder::load()` calls `exit(EXIT_FAILURE)` if the file can't be opened. The Python port lets `open()`'s natural `FileNotFoundError` propagate instead of force-exiting the process — `QFileDialog.getOpenFileName` already only returns a path that exists in the normal flow, and a hard `sys.exit()` inside a library-style method would be both un-Pythonic and untestable. This is a language-idiom adaptation, not a simplification of the demo's actual behavior.
- Ship mesh asset: copy `NGL9Demos/AdvancedGameKeyControl/models/SpaceShip.obj` verbatim into `GameKeyControl/models/SpaceShip.obj` (a plain, small OBJ file, ~62 KB), matching this repo's per-demo `models/` subfolder convention (`ObjViewer/models/`, `ColourObj/models/`, `SkinnedMeshImport/models/`).
- Font asset: reuse the shared root `font/Arial.ttf` (do **not** copy a demo-local font, matching the `Instancing/main.py` precedent: `Path(__file__).parent.parent / "font" / "Arial.ttf"`).
- Screenshots: captured for real via the established `screencapture -R<bounds>` method (launch the demo, find its window bounds via `osascript`/System Events, capture, kill the process) — this wave's session has proven this reliable across every prior phase; do not fall back to leaving a screenshot as a TODO for Jon.
- `ruff check` and `ruff format --check` must pass.
- README.md per demo folder (description, controls, teaching points, `![](GameKeyControl.png)` reference).
- Root `README.md` gets one row for `GameKeyControl`, `(OpenGL + WebGPU)` suffix (matching every other dual-backend row's exact style — see `README.md`'s `MatrixStack`/`Spotlight`/etc. rows), added in Task 2 (the GL task, matching this wave's precedent of the row landing with the first backend's commit).
- One commit per task.

---

## Task 1: Shared game-controls module + tests

**Files:**
- Create: `GameKeyControl/game_controls.py`
- Create: `GameKeyControl/tests/test_game_controls.py`

**Source:** `NGL9Demos/AdvancedGameKeyControl/include/GameControls.h`, `include/KeyRecorder.h`, `src/KeyRecorder.cpp`, `src/SpaceShip.cpp` (the `move()`/`draw()` transform logic only — the mesh-loading/drawing itself is a Task 2/3 concern, not this module's).

**Design notes:**

- `GameControls` bitflags — use `enum.IntFlag` so combinations remain usable as plain integers (needed as `MOTION_TABLE` indices) while staying self-documenting:
  ```python
  import enum


  class GameControls(enum.IntFlag):
      UP = 1 << 0
      DOWN = 1 << 1
      LEFT = 1 << 2
      RIGHT = 1 << 3
      ROTATE = 1 << 4
  ```
- `MOTION_TABLE` — a 32-entry tuple of `(offset_x, offset_y, rotation)` float triples, transcribed **verbatim** from `GameControls.h`'s `g_motionTable` (index = the raw `keys_pressed` byte, 0-31). Reproduce every entry including the "nonsense" combos (opposite keys held together cancel to zero) exactly as the source has them — this bitmask-as-array-index technique, including its unglamorous edge cases, **is** the demo's teaching point, not something to clean up:
  ```python
  MOTION_TABLE: tuple[tuple[float, float, float], ...] = (
      (0.0, 0.0, 0.0),  # 0
      (0.0, 1.0, 0.0),  # UP
      (0.0, -1.0, 0.0),  # DOWN
      (0.0, 0.0, 0.0),  # UP|DOWN (nonsense)
      (-1.0, 0.0, 0.0),  # LEFT
      (-0.707, 0.707, 0.0),  # UP|LEFT
      (-0.707, -0.707, 0.0),  # DOWN|LEFT
      (-1.0, 0.0, 0.0),  # UP|DOWN (nonsense) & LEFT
      (1.0, 0.0, 0.0),  # RIGHT
      (0.707, 0.707, 0.0),  # UP|RIGHT
      (0.707, -0.707, 0.0),  # DOWN|RIGHT
      (1.0, 0.0, 0.0),  # UP|DOWN (nonsense) & RIGHT
      (0.0, 0.0, 0.0),  # LEFT|RIGHT (nonsense)
      (0.0, 1.0, 0.0),  # UP & LEFT|RIGHT (nonsense)
      (0.0, -1.0, 0.0),  # DOWN & LEFT|RIGHT (nonsense)
      (0.0, 0.0, 0.0),  # UP|DOWN (nonsense) & LEFT|RIGHT (nonsense)
      # -- ROTATE held: same 16 entries again, rotation flag set to 1 --
      (0.0, 0.0, 1.0),
      (0.0, 1.0, 1.0),
      (0.0, -1.0, 1.0),
      (0.0, 0.0, 1.0),
      (-1.0, 0.0, 1.0),
      (-0.707, 0.707, 1.0),
      (-0.707, -0.707, 1.0),
      (-1.0, 0.0, 1.0),
      (1.0, 0.0, 1.0),
      (0.707, 0.707, 1.0),
      (0.707, -0.707, 1.0),
      (1.0, 0.0, 1.0),
      (0.0, 0.0, 1.0),
      (0.0, 1.0, 1.0),
      (0.0, -1.0, 1.0),
      (0.0, 0.0, 1.0),
  )
  X_EXTENTS = 40.0
  Y_EXTENTS = 30.0
  ROTATION_UPDATE = 4.0
  ```
- `move_ship()` — pure function, mirrors `SpaceShip::move()` exactly (increment, then clamp X/Y to `±extents`; rotation accumulates unclamped, matching the C++ which never wraps or clamps it):
  ```python
  from ncca.ngl import Vec3


  def move_ship(pos: Vec3, rotation: float, keys_pressed: int) -> tuple[Vec3, float]:
      dx, dy, drot = MOTION_TABLE[keys_pressed]
      new_pos = Vec3(pos.x + dx, pos.y + dy, pos.z)
      new_pos.x = max(-X_EXTENTS, min(X_EXTENTS, new_pos.x))
      new_pos.y = max(-Y_EXTENTS, min(Y_EXTENTS, new_pos.y))
      new_rotation = rotation + ROTATION_UPDATE * drot
      return new_pos, new_rotation
  ```
- `ship_transform()` — pure function shared by both backends (row-vector convention already established throughout this repo — translation in row 3, matches `ColourObj/main.py`'s `mouse_global_tx[3, 0] = ...` pattern):
  ```python
  from ncca.ngl import Mat4


  def ship_transform(pos: Vec3, rotation: float) -> Mat4:
      tx = Mat4().rotate_y(rotation)
      tx[3, 0] = pos.x
      tx[3, 1] = pos.y
      tx[3, 2] = pos.z
      return tx
  ```
- `KeyRecorder` — mirrors `KeyRecorder.h`/`.cpp` exactly, including the text file format (`save`/`load` use the identical layout: frame count, then `"x y z"`, then one integer per line):
  ```python
  from pathlib import Path


  class KeyRecorder:
      def __init__(self) -> None:
          self._frames: list[int] = []
          self._start_position: Vec3 = Vec3(0.0, 0.0, 0.0)

      def size(self) -> int:
          return len(self._frames)

      def __getitem__(self, index: int) -> int:
          return self._frames[index]

      def add_frame(self, control_vars: int) -> None:
          self._frames.append(control_vars)

      def set_start_position(self, pos: Vec3) -> None:
          self._start_position = Vec3(pos.x, pos.y, pos.z)

      def get_start_position(self) -> Vec3:
          return Vec3(
              self._start_position.x, self._start_position.y, self._start_position.z
          )

      def save(self, path: Path) -> None:
          lines = [str(len(self._frames))]
          lines.append(
              f"{self._start_position.x} {self._start_position.y} {self._start_position.z}"
          )
          lines.extend(str(frame) for frame in self._frames)
          Path(path).write_text("\n".join(lines) + "\n")

      def load(self, path: Path) -> None:
          lines = Path(path).read_text().split()
          count = int(lines[0])
          x, y, z = float(lines[1]), float(lines[2]), float(lines[3])
          self._start_position = Vec3(x, y, z)
          self._frames = [int(v) for v in lines[4 : 4 + count]]
  ```
  (Reading with `.split()` rather than line-by-line tolerates either the writer's own newline-separated layout or any hand-edited whitespace variant — the C++ reader uses `>>` stream extraction, which is whitespace-insensitive the same way.)

- [ ] **Step 1: Write the failing tests**

```python
# GameKeyControl/tests/test_game_controls.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ncca.ngl import Vec3

from game_controls import (
    GameControls,
    KeyRecorder,
    MOTION_TABLE,
    ROTATION_UPDATE,
    X_EXTENTS,
    Y_EXTENTS,
    move_ship,
    ship_transform,
)


def test_motion_table_has_32_entries():
    assert len(MOTION_TABLE) == 32


def test_motion_table_matches_source_values():
    assert MOTION_TABLE[0] == (0.0, 0.0, 0.0)
    assert MOTION_TABLE[GameControls.UP] == (0.0, 1.0, 0.0)
    assert MOTION_TABLE[GameControls.UP | GameControls.LEFT] == (-0.707, 0.707, 0.0)
    # "nonsense" combo: opposite keys cancel
    assert MOTION_TABLE[GameControls.UP | GameControls.DOWN] == (0.0, 0.0, 0.0)
    # rotate-held half of the table mirrors the first half with rotation=1
    assert MOTION_TABLE[GameControls.ROTATE | GameControls.UP] == (0.0, 1.0, 1.0)


def test_move_ship_applies_offset():
    pos = Vec3(0.0, 0.0, 0.0)
    new_pos, new_rotation = move_ship(pos, 0.0, GameControls.UP)
    assert new_pos.x == 0.0
    assert new_pos.y == 1.0
    assert new_rotation == 0.0


def test_move_ship_clamps_to_extents():
    pos = Vec3(X_EXTENTS - 0.5, Y_EXTENTS - 0.5, 0.0)
    new_pos, _ = move_ship(pos, 0.0, GameControls.UP | GameControls.RIGHT)
    assert new_pos.x == X_EXTENTS
    assert new_pos.y == Y_EXTENTS


def test_move_ship_accumulates_rotation_while_held():
    pos = Vec3(0.0, 0.0, 0.0)
    _, rotation = move_ship(pos, 10.0, GameControls.ROTATE)
    assert rotation == 10.0 + ROTATION_UPDATE


def test_ship_transform_places_translation_in_row_3():
    tx = ship_transform(Vec3(1.0, 2.0, 3.0), 0.0)
    assert tx[3, 0] == 1.0
    assert tx[3, 1] == 2.0
    assert tx[3, 2] == 3.0


def test_key_recorder_round_trips_through_a_file(tmp_path):
    recorder = KeyRecorder()
    recorder.set_start_position(Vec3(1.5, -2.5, 0.0))
    recorder.add_frame(int(GameControls.UP))
    recorder.add_frame(int(GameControls.UP | GameControls.LEFT))
    recorder.add_frame(0)

    out_file = tmp_path / "recording.kp"
    recorder.save(out_file)

    reloaded = KeyRecorder()
    reloaded.load(out_file)
    assert reloaded.size() == 3
    assert reloaded[0] == int(GameControls.UP)
    assert reloaded[1] == int(GameControls.UP | GameControls.LEFT)
    assert reloaded[2] == 0
    start = reloaded.get_start_position()
    assert (start.x, start.y, start.z) == (1.5, -2.5, 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest GameKeyControl/tests/test_game_controls.py -v`
Expected: FAIL/ERROR — `game_controls` module not found.

- [ ] **Step 3: Write `game_controls.py`**

Implement exactly as shown in the design notes above (all of `GameControls`, `MOTION_TABLE`, `X_EXTENTS`/`Y_EXTENTS`/`ROTATION_UPDATE`, `move_ship`, `ship_transform`, `KeyRecorder`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest GameKeyControl/tests/test_game_controls.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add GameKeyControl/game_controls.py GameKeyControl/tests/test_game_controls.py
git commit -m "feat(game-key-control): add the shared key-bitmask, motion table, and recorder module"
```

---

## Task 2: GameKeyControl (OpenGL)

**Files:**
- Create: `GameKeyControl/main.py`
- Create: `GameKeyControl/models/SpaceShip.obj`
- Create: `GameKeyControl/README.md`
- Create: `GameKeyControl/GameKeyControl.png`
- Modify: `README.md` (root — add the GameKeyControl row)

**Interfaces:**
- Consumes: `game_controls.GameControls`, `game_controls.MOTION_TABLE` (indirectly, via `move_ship`), `game_controls.move_ship(pos, rotation, keys_pressed) -> (Vec3, float)`, `game_controls.ship_transform(pos, rotation) -> Mat4`, `game_controls.KeyRecorder` (from Task 1, same folder — import via `from game_controls import ...`, no `sys.path.insert` needed since both files sit in the same directory and `uv run --script` puts the script's own folder on `sys.path[0]`, same as `TextureCompressor/main.py`'s `from dxt_texture import read_cmptx`).

**Source:** `NGL9Demos/AdvancedGameKeyControl/src/NGLScene.cpp`, `src/SpaceShip.cpp`, `include/NGLScene.h`, `include/SpaceShip.h`, `src/main.cpp`.

**Design notes:**

- Copy `NGL9Demos/AdvancedGameKeyControl/models/SpaceShip.obj` verbatim into `GameKeyControl/models/SpaceShip.obj` — do not regenerate or simplify it.
- `MainWindow(QOpenGLWindow)` — **no `PySideEventHandlingMixin`**, no mouse handlers at all (matches the C++, which has none).
- `initializeGL`:
  - `gl.glClearColor(0.0, 0.0, 0.0, 1.0)`, `gl.glEnable(gl.GL_DEPTH_TEST)`, `gl.glEnable(gl.GL_MULTISAMPLE)`.
  - `self.view = look_at(Vec3(0.0, 0.0, 80.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))`; `self.project = perspective(45.0, w/h, 0.05, 350.0)`.
  - `ShaderLib.use(DefaultShader.COLOUR)`; `ShaderLib.set_uniform("Colour", 1.0, 1.0, 0.0, 1.0)` — set once, exactly as the C++ does (this shader/uniform pair is a byte-for-byte match to `ngl::nglColourShader`, confirmed against the library's own `colour_vertex.glsl`/`colour_fragment.glsl` source — a flat, unlit `outColour = Colour`, no custom GLSL files needed for this task).
  - `self.ship = Obj.from_file(str(Path(__file__).parent / "models" / "SpaceShip.obj")); self.ship.create_vao()`.
  - `self.ship_pos = Vec3(0.0, 0.0, 0.0)`; `self.ship_rotation = 0.0`.
  - `Text.add_font("Arial", str(Path(__file__).parent.parent / "font" / "Arial.ttf"), 12)`; `Text.set_screen_size(self.width(), self.height())`.
  - `self.keys_pressed = 0`; `self.recording = False`; `self.playback_active = False`; `self.current_playback_frame = 0`; `self.key_recorder = KeyRecorder()`.
  - `self.ship_timer = QTimer(self); self.ship_timer.timeout.connect(self._on_ship_tick); self.ship_timer.start(15)`.
  - `self.redraw_timer = QTimer(self); self.redraw_timer.timeout.connect(self.update); self.redraw_timer.start(30)`.
- `_on_ship_tick`:
  ```python
  def _on_ship_tick(self) -> None:
      if self.playback_active:
          if self.current_playback_frame <= 0:
              self.ship_pos = self.key_recorder.get_start_position()
          if self.current_playback_frame < self.key_recorder.size():
              self.keys_pressed = self.key_recorder[self.current_playback_frame]
              self.current_playback_frame += 1
          else:
              self.playback_active = False
              self.current_playback_frame = 0
      elif self.recording:
          self.key_recorder.add_frame(self.keys_pressed)
      self.ship_pos, self.ship_rotation = move_ship(
          self.ship_pos, self.ship_rotation, self.keys_pressed
      )
  ```
- `paintGL`: clear, `gl.glViewport(0, 0, self.window_width, self.window_height)` (device-pixel-scaled, computed in `resizeGL` the same way `ColourObj/main.py` does), `ShaderLib.use(DefaultShader.COLOUR)`, `mvp = self.project @ self.view @ ship_transform(self.ship_pos, self.ship_rotation)`, `ShaderLib.set_uniform("MVP", mvp)`, `self.ship.draw()`. Then HUD: if `self.recording`, `Text.render_text("Arial", 10, 18, "Recording", Vec3(1.0, 0.0, 0.0))`; if `self.playback_active`, `Text.render_text("Arial", 10, 18, f"Playback doing frame {self.current_playback_frame}", Vec3(1.0, 1.0, 0.0))`.
- `resizeGL`: `self.project = perspective(45.0, w / h, 0.05, 350.0)`; `self.window_width = int(w * self.devicePixelRatio())`; `self.window_height = int(h * self.devicePixelRatio())`; `Text.set_screen_size(self.window_width, self.window_height)`.
- `keyPressEvent`/`keyReleaseEvent`: held-state bitmask, `Qt.Key_Up/Down/Left/Right` → `GameControls.UP/DOWN/LEFT/RIGHT`, `Qt.Key_R` → `GameControls.ROTATE` (set on press, clear on release — both handlers needed, matching the C++'s separate `keyPressEvent`/`keyReleaseEvent`).
- `keyPressEvent` additionally handles (press-only, no release counterpart, matching the C++ `switch` which only appears in `keyPressEvent`):
  - `Qt.Key_Space`: `self.key_recorder.set_start_position(self.ship_pos); self.recording = not self.recording`.
  - `Qt.Key_P`: `self.playback_active = not self.playback_active`.
  - `Qt.Key_S`: `QFileDialog.getSaveFileName(None, "Save Keypresses", ".", "*.kp")`; if a path was chosen, `self.key_recorder.save(path)`.
  - `Qt.Key_L`: `QFileDialog.getOpenFileName(None, "Load Keypresses", ".", "*.kp")`; if a path was chosen, `self.key_recorder.load(path)`.
  - `Qt.Key_Escape`: `self.close()`.
- `closeEvent`: `self.ship_timer.stop(); self.redraw_timer.stop(); super().closeEvent(event)`.

- [ ] **Step 1: Write `main.py`**

Implement `MainWindow` per the design notes above, wired into the standard entry-point skeleton (`DebugApplication`, `argparse` with `--smoketest`/`--debug`, the `QSurfaceFormat` block copied from `VAOPrimitives/main.py`'s `__main__`).

- [ ] **Step 2: Copy the ship model**

```bash
cp /Users/jmacey/teaching/NGL9Demos/AdvancedGameKeyControl/models/SpaceShip.obj GameKeyControl/models/SpaceShip.obj
```

- [ ] **Step 3: Make executable and smoke-test**

```bash
chmod +x GameKeyControl/main.py
cd GameKeyControl && uv run --script main.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0, no traceback. Run WITHOUT `QT_QPA_PLATFORM=offscreen` (segfaults `QOpenGLWindow` on this machine).

- [ ] **Step 4: Write README.md, screenshot, and the root README row**

`GameKeyControl/README.md` — description (bitmask-indexed motion table for simultaneous held-key combos; record/playback to a `.kp` file), controls (arrow keys move, `R` rotates, `Space` toggles recording, `P` toggles playback, `S`/`L` save/load a recording, `Esc` quits), `![](GameKeyControl.png)`. Capture the screenshot via the established `screencapture -R<bounds>` method (launch, find window bounds via `osascript`, capture, kill).

Add to root `README.md`:
```markdown
| <a href="GameKeyControl"><img src="GameKeyControl/GameKeyControl.png" width="220"></a> | [GameKeyControl](GameKeyControl) | Bitmask-indexed motion table for held multi-key combos, with record/playback (OpenGL + WebGPU) |
```

- [ ] **Step 5: Commit**

```bash
git add GameKeyControl/main.py GameKeyControl/models/SpaceShip.obj GameKeyControl/README.md GameKeyControl/GameKeyControl.png README.md
git commit -m "feat(game-key-control): add the OpenGL held-key spaceship demo"
```

---

## Task 3: GameKeyControl (WebGPU)

**Files:**
- Create: `GameKeyControl/main_webgpu.py`
- Create: `GameKeyControl/ship_mesh.py`
- Create: `GameKeyControl/GameKeyControlShader.wgsl`
- Create: `GameKeyControl/tests/test_ship_mesh.py`
- Modify: `GameKeyControl/README.md` (append the WebGPU section)

**Interfaces:**
- Consumes: everything Task 2 consumes from `game_controls.py`, plus `ship_mesh.load_ship_vertex_data(path) -> tuple[np.ndarray, int]` (produced by this task, `(interleaved_float32_array, vertex_count)`).
- Does not import `main.py`; does not reinterpret `game_controls.py`'s logic (record/playback and the motion table are identical on both backends — this task's only genuinely new code is mesh loading and drawing).

**Source:** same C++ files as Task 2 (this is a faithful port of the same behavior, just a different rendering backend).

**Design notes:**

- **Why `ship_mesh.py` exists:** `ncca.ngl.Obj.create_vao()` builds the interleaved `[x, y, z, nx, ny, nz, u, v]` vertex data internally but only exposes it via a GL-specific VAO (no numpy-only accessor) — the same situation `PrimData.sphere()` solves for procedural spheres. `load_ship_vertex_data()` replicates `BaseMesh.create_vao()`'s triangulation/interleave loop (read `obj.faces`, index into `obj.vertex`/`obj.normals`/`obj.uv`, same V-flip-for-consistency as the library does) without any GL call, returning a plain numpy array a WebGPU vertex buffer can be built from directly:
  ```python
  from pathlib import Path

  import numpy as np
  from ncca.ngl import Obj, Vec3


  def load_ship_vertex_data(path: Path) -> tuple[np.ndarray, int]:
      obj = Obj()
      obj.load(str(path))
      if not obj.is_triangular():
          raise RuntimeError(f"{path} is not a triangulated mesh")

      rows = []
      for face in obj.faces:
          for i in range(3):
              v = obj.vertex[face.vertex[i]]
              if obj.normals:
                  n = obj.normals[face.normal[i]]
              else:
                  n = Vec3(0.0, 0.0, 0.0)
              if obj.uv:
                  uv = obj.uv[face.uv[i]]
                  u, vv = uv.x, 1.0 - uv.y
              else:
                  u, vv = 0.0, 0.0
              rows.append([v.x, v.y, v.z, n.x, n.y, n.z, u, vv])

      data = np.array(rows, dtype=np.float32).flatten()
      return data, len(rows)
  ```
  (Same 8-float interleave and vertex-buffer layout — `array_stride=8*4`, position/normal/uv at offsets 0/12/24 — every other WebGPU demo in this repo already uses; the fragment shader below never samples `uv`/`normal`, but keeping the layout consistent with the rest of the codebase is worth the two unused attributes.)
- `GameKeyControlShader.wgsl` — genuinely flat/unlit, mirroring `nglColourShader`'s own GLSL exactly (no lighting term at all, unlike `AffineTransforms/AffineTransformsShader.wgsl`'s lit gizmo shader):
  ```wgsl
  struct Uniforms {
      mvp: mat4x4<f32>,
      colour: vec4<f32>,
  };
  @group(0) @binding(0) var<uniform> u: Uniforms;

  struct VertexOutput {
      @builtin(position) position: vec4<f32>,
  };

  @vertex
  fn vertex_main(
      @location(0) in_vert: vec3<f32>,
      @location(1) in_normal: vec3<f32>,
      @location(2) in_uv: vec2<f32>,
  ) -> VertexOutput {
      var out: VertexOutput;
      out.position = u.mvp * vec4<f32>(in_vert, 1.0);
      return out;
  }

  @fragment
  fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
      return u.colour;
  }
  ```
- `main_webgpu.py`: `class WebGPUScene(WebGPUWidget)` — no mouse handlers (matches Task 2's fixed-camera, keyboard-only design). Single ship, single draw call per frame — **no per-draw uniform-buffer pool needed** (this wave's established queue-timeline-aliasing rule does not apply here; state it explicitly so nobody adds unneeded pool machinery).
  - `_create_scene`: `data, vertex_count = load_ship_vertex_data(Path(__file__).parent / "models" / "SpaceShip.obj")`; build the vertex buffer (`usage=wgpu.BufferUsage.VERTEX`); build one uniform buffer sized for `{mvp: mat4x4<f32>, colour: vec4<f32>}` (80 bytes: 64 + 16, both already 16-aligned, no padding gotcha here); `queue.write_buffer` the colour once at creation (`(1.0, 1.0, 0.0, 1.0)`, yellow, matching Task 2 exactly) since it never changes.
  - `paintWebGPU`: compute `mvp = self.project @ self.view @ ship_transform(self.ship_pos, self.ship_rotation)`; `queue.write_buffer` the `mvp` field of the uniform buffer; one render pass, one `draw(vertex_count)`; clear colour `(0.0, 0.0, 0.0, 1.0)` (matches Task 2's black background); after `self._update_colour_buffer()`, call `self.render_text(...)` for the same "Recording"/"Playback doing frame N" HUD Task 2 shows (via the `WebGPUWidget.render_text()` base-class helper, same signature/usage as `Instancing/InstancingWebGPU.py`'s `_draw_hud`).
  - `_on_tick` (15 ms): identical logic to Task 2's `_on_ship_tick`, calling the same `move_ship()`/`KeyRecorder` methods from `game_controls.py`.
  - A second `QTimer` (30 ms) calling `self.update()` only, matching Task 2's redraw-timer split.
  - `keyPressEvent`/`keyReleaseEvent`: identical key handling to Task 2 (same `Qt.Key_*` constants, same `GameControls` bits, same `Space`/`P`/`S`/`L`/`Escape` behavior, same `QFileDialog` calls against the same shared `KeyRecorder`).
  - `closeEvent`: stop both timers.

- [ ] **Step 1: Write the failing test for `ship_mesh.py`**

```python
# GameKeyControl/tests/test_ship_mesh.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ship_mesh import load_ship_vertex_data

_SHIP_PATH = Path(__file__).parent.parent / "models" / "SpaceShip.obj"


def test_load_ship_vertex_data_returns_interleaved_float32():
    data, vertex_count = load_ship_vertex_data(_SHIP_PATH)
    assert data.dtype.name == "float32"
    assert vertex_count > 0
    assert data.shape == (vertex_count * 8,)


def test_load_ship_vertex_data_is_a_multiple_of_a_triangle():
    _, vertex_count = load_ship_vertex_data(_SHIP_PATH)
    assert vertex_count % 3 == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest GameKeyControl/tests/test_ship_mesh.py -v`
Expected: FAIL — `ship_mesh` module not found (and the fixture path won't exist until Task 2's `models/SpaceShip.obj` lands — this task runs after Task 2, so the file is already present).

- [ ] **Step 3: Write `ship_mesh.py`**

Implement `load_ship_vertex_data` exactly as shown in the design notes above.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest GameKeyControl/tests/test_ship_mesh.py -v`
Expected: both PASS.

- [ ] **Step 5: Write `GameKeyControlShader.wgsl` and `main_webgpu.py`**

Implement per the design notes above, wired into the standard WebGPU entry-point skeleton (`--smoketest`/`--debug` argparse, `DebugApplication`, `get_default_device()`, `self.msaa_sample_count = 4`, `self._create_render_buffer()`).

- [ ] **Step 6: Make executable and smoke-test**

```bash
chmod +x GameKeyControl/main_webgpu.py
cd GameKeyControl && QT_QPA_PLATFORM=offscreen uv run --script main_webgpu.py --smoketest; cd ..
```
Expected: `SMOKETEST OK`, exit 0, no traceback, no wgpu validation warnings.

- [ ] **Step 7: Append the WebGPU section to README.md**

```markdown
## WebGPU version

`main_webgpu.py` needs no reinterpretation for this demo's actual teaching
point (the bitmask-indexed motion table and file-based record/playback are
plain Python/Qt, identical on both backends) -- the only backend-specific
work is drawing the ship and the HUD. `ncca.ngl.Obj.create_vao()` is
GL-only, so `ship_mesh.py` replicates its interleave logic to build a numpy
vertex buffer directly, and a small hand-rolled flat WGSL shader
(`GameKeyControlShader.wgsl`) stands in for the OpenGL side's built-in
`nglColourShader` equivalent. A recording saved from either backend loads
and plays back correctly in the other -- the `.kp` format and the
`KeyRecorder` class are shared, unmodified, from `game_controls.py`.
```

- [ ] **Step 8: Commit**

```bash
git add GameKeyControl/main_webgpu.py GameKeyControl/ship_mesh.py GameKeyControl/GameKeyControlShader.wgsl GameKeyControl/tests/test_ship_mesh.py GameKeyControl/README.md
git commit -m "feat(game-key-control): add the WebGPU spaceship demo"
```

---

## Final steps (after all 3 tasks)

- [ ] **Run full verification**

```bash
uv run ruff check GameKeyControl
uv run ruff format --check GameKeyControl
uv run pytest GameKeyControl/tests --ignore=MathNodeEditor/tests/test_main.py
```
Expected: ruff clean; all `GameKeyControl` tests pass. (The `--ignore` is for a pre-existing, unrelated `MathNodeEditor`/`BVHViewer` pytest basename collision that predates this phase — confirmed via `git log` in Phase 4's final review — not something to fix here.)

- [ ] **Confirm the root README row is present and correctly formatted**

One `GameKeyControl` row, `(OpenGL + WebGPU)` suffix, same style as every other dual-backend row, no duplicates.

- [ ] **Report to Jon**

Flag the known risk areas for a human/reviewer to double-check interactively, since no automated smoketest exercises them:
- Held multi-key combos actually feel responsive at the two different timer rates (15 ms sim, 30 ms redraw) on both backends — the C++'s deliberate decoupling is a genuine gameplay-feel choice, worth confirming it doesn't feel laggy in practice.
- Record (`Space`) → move around → stop recording → `P` to play back — does the ship visibly retrace the same path from the same start position, on both backends.
- Save a recording (`S`) in one backend, load it (`L`) in the other, and confirm playback matches — the cross-backend `.kp` compatibility this plan's design relies on.
- WebGPU ship mesh orientation/scale against the OpenGL version's — `ship_mesh.py` is new, untested-by-eye code; confirm the ship isn't inverted, inside-out (a back-face-culling/winding mismatch), or oddly scaled relative to the GL rendering.
