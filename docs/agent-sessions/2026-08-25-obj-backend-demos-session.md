# OBJ backend demos session

## Goal

Migrate the PyNGLDemos OBJ users to the parser-only OBJ API in PyNGL 2.0.

## Files changed

- GameKeyControl OpenGL and WebGPU entry points, tests and README
- ObjViewer, MuJoCoNGL, MathNodeEditor and Obj2Numpy
- Removed the GameKeyControl OBJ packing helper

## Commands run

- Focused GameKeyControl, MathNodeEditor and MuJoCoNGL tests
- WebGPU GameKeyControl smoke test
- Ruff checks and `git diff --check`
