# SciFiUI WebGPU Session

Goal: add a WebGPU version of the `SciFiUI` demo in the same folder as the OpenGL version.

Files changed:

- `SciFiUI/WebGPUmain.py`
- `SciFiUI/shaders/UIShader.wgsl`
- `SciFiUI/shaders/CRTShader.wgsl`
- `SciFiUI/tests/test_webgpu_scene_data.py`
- `SciFiUI/README.md`
- `docs/agent-sessions/2026-08-09-scifiui-webgpu-session.md`

Commands run:

```bash
git status --short --branch
git worktree add .worktrees/scifiui-webgpu -b agent/scifiui-webgpu
uv run pytest SciFiUI/tests/test_webgpu_scene_data.py
uv run ruff check SciFiUI/WebGPUmain.py SciFiUI/tests/test_webgpu_scene_data.py
uv run SciFiUI/WebGPUmain.py --smoketest 300
```

Notes:

- The first pytest run failed because the WebGPU module and WGSL shaders did not exist yet.
- The WebGPU smoketest needed permission for `uv` to use its configured cache outside the sandbox, then completed with `SMOKETEST OK`.
- I did not find a separate session export command configured in the repo instructions.
