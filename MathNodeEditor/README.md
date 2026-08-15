# MathNodeEditor

![MathNodeEditor](MathNodeEditor.png)

This is a small PySide6 node editor for experimenting with the PyNGL maths classes. It reopens whichever graph you had open last; the first time you run it (or if that file's gone missing) it opens `examples/vec3_multiply_demo.json` instead — two editable `Vec3` nodes wired through a component multiply node, so changing any value updates the output straight away.

The value palette has `Float`, `Vec2`, `Vec3`, `Vec4`, `Mat2`, `Mat3`, `Mat4` and `Quaternion`. Quaternions use the same `(s, x, y, z)` order as PyNGL and start as the identity quaternion.

The Mat4 nodes cover translate, scale, rotation about each axis, `look_at`, perspective, orthographic and frustum projection, plus a `Transform` node wrapping PyNGL's `Transform` class (`Position`/`Rotation`/`Scale`, rotation in degrees, xyz order). These, and the quaternion axis-angle node, take their parameters from spin boxes built into the node itself rather than wired-in Value nodes — there's nothing to connect, just numbers to type. The remaining quaternion nodes (Hamilton product, vector rotation, conversion to and from `Mat4`, slerp, conjugate, inverse) still take wired inputs, since they combine other quaternions or matrices rather than raw numbers. The original add, subtract, component multiply, matrix multiply, dot, cross, normalise and transpose nodes are unchanged.

`Multiply` is explicitly component-wise (PyNGL reserves `*` for scalar multiplication), whilst `Matrix Multiply` and `Quaternion Product` use the normal `@` operation.

The Mesh group adds an `Obj Loader` node, which loads a triangulated `.obj` file and splits it into four separate outputs: `Vertices`, `Faces`, `UVs` and `Normals`. Non-triangular meshes are rejected with a graph error rather than auto-triangulated, same as `ColourObj`/`Obj2Numpy` elsewhere in this repo. `Transform Vertices` and `Transform Normals` let you push a matrix through those arrays — vertices take a `Mat4` (translation included), normals take a `Mat3` (translation dropped) — but the normal matrix isn't built for you. Wire up `Mat4 to Mat3` → `Inverse` → `Transpose` yourself if you want it correct under non-uniform scale, which is rather the point of a teaching tool.

A `Mesh Viewer` node merges `Vertices`/`Faces`/`UVs`/`Normals` (plus an optional `Colour` `Vec4`) back into a mesh and renders it live, using PyNGL's built-in shaders — pick `Solid Colour` or `Diffuse` from the dropdown and tick `Wireframe` for a line-only view. Bad face indices are reported on the node before they reach OpenGL, and Diffuse shading needs a non-empty normal array. It renders in two places at once: a small preview embedded in the node itself, and a pop-out window with the usual arcball camera (drag to rotate, right-drag to pan, wheel to zoom) via the `Pop Out` button.

Click a palette button to add a node, or move the pointer over the canvas and press `Tab` to open the node menu at that position. Start typing to filter the menu and press `Return` to create the first matching node. You can also select an entry with the mouse.

Drag from an output socket to an input socket to connect it. Value nodes use zero vectors and identity matrices as their defaults. Generator nodes start pre-filled with sensible non-zero numbers instead, so a new Look At or Perspective node gives you something reasonable to look at straight away rather than a degenerate matrix. Numeric fields keep seven decimal places and the saved graph keeps the model's full value, so loading and saving doesn't quietly round a number to what happened to fit in the widget. The palette scrolls, the mouse wheel zooms the canvas (clamped to sensible limits) and nodes can be selected and dragged around — hover over a numeric field first and the wheel edits that value instead. Loaded graphs are framed automatically; press `H` to frame every node again after editing the layout.

Select a node or wire and press `Delete`/`Backspace` to remove it, or right-click it for the same option. The `File` menu's `New` (`Ctrl+N`), `Open...` (`Ctrl+O`), `Save` (`Ctrl+S`) and `Save As...` (`Ctrl+Shift+S`) read and write the graph as JSON — `Save` writes straight back to whichever file is open, prompting for one the first time. Version 1 files are checked completely before the current canvas is replaced, and saves use a temporary file so a failed write can't truncate the previous graph. Older files without a version number still load. `New`, `Open...` and closing the window all ask to save first if the graph has unsaved changes. The window remembers its size and the last file you had open between runs.

`examples/vec3_multiply_demo.json` is the bundled starting graph described above — two `Vec3` values multiplied component-wise into an `Output` node.

`examples/mesh_pipeline_demo.json` is a saved graph showing the mesh nodes in use: `Obj Loader` (reading `examples/cube.obj`) → `Mat4 Rotate Y` → `Transform Vertices`, plus the manual `Mat4 to Mat3` → `Inverse` → `Transpose` → `Transform Normals` pipeline, feeding a `Mesh Viewer` in `Diffuse` shading. Open it with `File` → `Open...`.

`examples/mvp_demo.json` builds a typical Model/View/Projection matrix: a `Transform` node for Model, `Look At` for View, `Perspective` for Projection — each carrying its own numbers directly, no Value nodes required — combined with two `Matrix Multiply` nodes as `MVP = Projection @ (View @ Model)` — the same order used throughout this repo's OpenGL demos (e.g. `ObjViewer.py`'s `mvp = self.project @ self.view @ self.mouse_global_tx`) — and shown on an `Output` node.

`examples/mvp_mesh_demo.json` puts the Model half of that to work on a displayed mesh instead of a plain `Output` node: `Obj Loader` (`examples/cube.obj`) feeds a `Transform` node's Position/Rotation/Scale spin boxes through `Transform Vertices` and the manual normal-matrix pipeline, landing on a `Mesh Viewer`. There's no `Look At`/`Perspective` in this one — View and Projection are the `Mesh Viewer`'s own arcball camera, which is what you're driving when you drag inside its preview or the `Pop Out` window.

## Running it

```bash
uv run MathNodeEditor/main.py
uv run pytest MathNodeEditor/tests
```
