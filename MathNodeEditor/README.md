# MathNodeEditor

![MathNodeEditor](MathNodeEditor.png)

This is a small PySide6 node editor for experimenting with the PyNGL maths classes. It starts with two editable `Vec3` nodes wired through a component multiply node, so changing any value updates the output straight away.

The value palette has `Vec2`, `Vec3`, `Vec4`, `Mat2`, `Mat3` and `Mat4`. The operation nodes cover add, subtract, component multiply, matrix multiply, dot, cross, normalise and transpose. `Multiply` is explicitly component-wise (PyNGL reserves `*` for scalar multiplication), whilst `Matrix Multiply` uses the normal `@` operation.

Click a palette button to add a node, then drag from an output socket to an input socket to connect it. Value nodes use zero vectors and identity matrices as their defaults. The mouse wheel zooms the canvas and nodes can be selected and dragged around.

## Running it

```bash
uv run MathNodeEditor/main.py
uv run pytest MathNodeEditor/tests
```
