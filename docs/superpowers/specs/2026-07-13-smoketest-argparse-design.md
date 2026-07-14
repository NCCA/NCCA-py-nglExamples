# Smoketest + argparse rollout for all demos

Date: 2026-07-13
Status: approved

## Goal

Every executable demo entry script supports a uniform CLI, parsed with
`argparse`, so any demo can be verified non-interactively:

- `--smoketest [MS]` — run the demo for `MS` milliseconds (default **200**,
  matching the existing hard-coded demos), print `SMOKETEST OK`, exit 0.
- `--debug` — Qt demos only: run under a `DebugApplication` (`QApplication`
  subclass whose `notify` prints full tracebacks from Qt event handlers
  before re-raising).

This replaces the ad-hoc `"--smoketest" in sys.argv` checks already present
in ~19 scripts and adds the options to the ~55 scripts that lack them.

## Scope

All executable entry scripts (shebang `#!/usr/bin/env -S uv run --script`),
**except**:

- `RunDemos.py` — the launcher itself.
- `Obj2Numpy/` — headless CLI tool, nothing to smoketest.

That leaves ~74 files across OpenGL (PySide6), WebGPU (PySide6 +
wgpu-py), QML/GUI, and SDL3 demos.

## Standard pattern — Qt demos (OpenGL and WebGPU)

```python
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest", nargs="?", const=200, default=None, type=int, metavar="MS",
        help="run for MS milliseconds (default 200), print SMOKETEST OK and exit",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="run with DebugApplication (tracebacks from Qt event handlers)",
    )
    args = parser.parse_args()

    # ... existing QSurfaceFormat / style setup unchanged ...

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    window = MainWindow()
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))
    sys.exit(app.exec())
```

Notes:

- `--smoketest` alone → 200 ms; `--smoketest 1000` → 1 s.
- Demos that already have a `DebugApplication` class keep it; demos that
  don't get the standard ~10-line class copied in (same as
  `Camera/main.py`). Demos with their own `QApplication` subclass or special
  app setup (QML/GUI demos) keep their app class and gain the same `notify`
  wrapping behind `--debug`.
- Demos whose entry point is a `main()` function keep that structure; the
  parser lives in `main()` (or a `parse_args()` helper) rather than being
  forced into `__main__`.

## SDL3 demos (`BlankPySDL3/main.py`, `SimplePyNGL/SDL3NGL.py`)

Same `--smoketest [MS]` option, **no** `--debug` (no Qt event loop to wrap).
Implemented with an SDL tick timer in the main loop:

```python
start = sdl3.SDL_GetTicks()
while running:
    ...
    if args.smoketest is not None and sdl3.SDL_GetTicks() - start >= args.smoketest:
        print("SMOKETEST OK")
        running = False
```

## Demos with existing CLI arguments

Scripts that hand-parse `sys.argv` (e.g. `ObjViewer/ObjViewer.py`'s optional
positional obj/texture paths) fold those into the same argparse parser as
positionals/options, preserving current defaults and behaviour.

## Verification

For every touched script:

- `QT_QPA_PLATFORM=offscreen uv run <script> --smoketest` prints
  `SMOKETEST OK` and exits 0. GPU-heavy WebGPU demos that cannot create an
  offscreen surface are verified on-screen instead; any that still fail are
  reported, not hidden.
- `uv run ruff check .` and `uv run ruff format --check .` clean.
- `uv run pytest` still passes.

## Process

- Worktree `.worktrees/smoketest-argparse`, branch `agent/smoketest-argparse`
  off `Version1.0`.
- Conventional commits, batched by group (OpenGL, WebGPU, SDL, GUI/QML,
  migration of existing smoketest scripts).
