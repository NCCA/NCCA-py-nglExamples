# MathNodeEditor

This is a small PySide6 node editor I wrote for experimenting with the
[PyNGL](https://github.com/NCCA/PyNGL) maths classes. I use it to show how
vectors, matrices, quaternions and Obj data move through graphics calculations
without hiding all the intermediate values in a script.

![](MathNodeEditor.png)

## Running it

It is expected you will use [uv](https://docs.astral.sh/uv/) to run the editor.

```bash
uv run MathNodeEditor/main.py
```

The editor reopens the last graph you used. On the first run it opens
`examples/vec3_multiply_demo.json`, so there is something to play with straight
away.

## Nodes

The value nodes cover `Float`, `Vec2`, `Vec3`, `Vec4`, `Mat2`, `Mat3`, `Mat4`
and `Quaternion`. Vectors start at zero, matrices start as identity matrices and
quaternions use PyNGL's `(s, x, y, z)` order.

There are nodes for the usual add, subtract, dot, cross, normalise, transpose
and inverse operations. `Multiply` is component wise (PyNGL uses `*` for scalar
multiplication), whilst `Matrix Multiply` and `Quaternion Product` use the
normal `@` operation.

The Mat4 nodes build translation, scale, axis rotations, camera projections and
a complete `Transform`. Generator nodes such as Look At, Perspective and
Transform keep their parameters in the node itself rather than needing a row
of extra Value nodes. Transform supports all six rotation orders and uses
degrees for rotation.

Quaternion nodes cover axis-angle construction, Hamilton products, vector
rotation, matrix conversion, slerp, conjugate and inverse. This makes it quite
easy to compare a quaternion calculation with the equivalent matrix version.

The Mesh nodes load a triangulated `.obj`, expose its vertices, faces, UVs and
normals, then put them back together in a live viewer. Vertices use a `Mat4`
transform and normals use a `Mat3`. I do not build the normal matrix
automatically: wire `Mat4 to Mat3` through `Inverse` and `Transpose` if you want
the correct result under non-uniform scale (which is rather the point of this
demo).

The Mesh Viewer has solid colour and diffuse shading, an optional wireframe and
a pop-out view with an arcball camera. The small preview and the pop-out view
update as the graph changes.

## Using the editor

Click a button in the palette to add a node. You can also move the pointer over
the canvas, press `Tab` and type part of a node name. Drag from an output socket
to an input socket to connect them; changing a value updates the downstream
nodes straight away.

Double-click a node header to rename it. Select a node or wire and press
`Delete`/`Backspace` to remove it (right-click works as well). The mouse wheel
zooms the canvas unless it is over a numeric field, and `H` frames all the
nodes.

The File menu reads and writes the graphs as JSON. New, Open and closing the
window will ask about unsaved changes, and the editor remembers its size and
the last file used.

## Examples

There are fourteen [example graphs](examples/README.md) covering vector
arithmetic, homogeneous coordinates, triangle normals, Lambert diffuse,
matrix transforms, camera projections, quaternions and mesh transforms. Each
entry suggests a value to change so you can see what it does.

Open one with `File` -> `Open...` and edit the inputs.

## Developer notes

[ARCHITECTURE.md](ARCHITECTURE.md) explains how the graph, Qt items and scene
fit together, along with the steps needed to add another operation or node
type.

Run the tests with

```bash
uv run pytest MathNodeEditor/tests
```
