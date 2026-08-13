# MathNodeEditor: Obj Loader and Mesh Viewer nodes

Date: 2026-08-13
Status: Approved, implementing on `agent/obj-mesh-nodes`

## Goal

Add two node types to `MathNodeEditor`:

1. **Obj Loader** — loads a Wavefront OBJ and exposes its Vertex, Face, UV and
   Normal data as separate graph outputs, so matrix-transform nodes can be
   wired onto the vertex/normal streams.
2. **Mesh Viewer** — merges Vertex/Face/UV/Normal (and an optional Colour)
   inputs back into a mesh and renders it live in 3D, using the built-in
   PyNGL shaders, with Solid Colour / Diffuse shading and an independent
   Wireframe toggle.

## Data model (`math_graph.py`)

Four new frozen dataclasses join the `MathValue` union, alongside the
existing `Vec2/Vec3/Vec4/Mat2/Mat3/Mat4/Quaternion/float`:

- `VertexArray(values: tuple[Vec3, ...])`
- `NormalArray(values: tuple[Vec3, ...])`
- `UVArray(values: tuple[Vec2, ...])`
- `FaceArray(triangles: tuple[Corner3, ...])` where `Corner3` is a 3-tuple of
  `(vertex_index, uv_index | None, normal_index | None)`, mirroring
  `ncca.ngl`'s own `Face` struct.

These are output-only — there is no "Values" palette entry for them, and
`MathType`/`TYPE_SHAPES`/`VALUE_CLASSES` are untouched.

New `Operation` members, all ordinary single-output operation nodes:

| Operation | Inputs | Behaviour |
|---|---|---|
| `MAT4_TO_MAT3` | Matrix | `Mat3.from_mat4(m)` |
| `INVERSE` | Matrix | `m.inverse()` |
| `TRANSFORM_VERTICES` | Matrix, Vertices | per-point `Vec4(v, 1) @ Mat4` |
| `TRANSFORM_NORMALS` | Matrix, Normals | per-vector `v @ Mat3` (linear part only) |

No automatic inverse-transpose correction is baked into `TRANSFORM_NORMALS`
— the correct normal matrix is built manually on the canvas from
`MAT4_TO_MAT3 → INVERSE → TRANSPOSE` (this is a teaching tool; the point is
to make that step explicit).

A pure conversion layer (headless-testable, no Qt/GL):

- `arrays_from_obj(obj: ncca.ngl.Obj) -> (VertexArray, FaceArray, UVArray, NormalArray)`
  — raises `GraphError` if `not obj.is_triangular()`. UV z is dropped
  (`ncca.ngl.Obj` stores UVs as `Vec3` with a placeholder z).
- `obj_from_arrays(vertices, faces, uvs, normals) -> ncca.ngl.Obj` — rebuilds
  a plain `Obj` (vertex/normals/uv/faces lists populated, no VAO) suitable
  for `obj.create_vao()`.

`MathGraph` gains:

- `LiteralNode(value: MathValue)` — a fourth node kind alongside
  `ValueNode`/`OperationNode`/`OutputNode`; evaluates to its stored value,
  no inputs, not user-editable.
- `MeshViewerNode(inputs: dict[int, str])` — a sink like `OutputNode` but
  with 5 named inputs (Vertices, Faces, UVs, Normals, Colour). Vertices and
  Faces are required (`GraphError` "needs input X" if missing); UVs/Normals/
  Colour are optional.
- `MathGraph.evaluate_mesh_viewer(node_id) -> MeshViewerInputs` — a small
  dataclass bundling the five evaluated values (`None` for unwired optional
  ones), separate from `evaluate()` since a mesh viewer has no single scalar
  result.

## Multi-output ports (`graphics_items.py`, `canvas.py`)

Every existing node has exactly one output port. The Obj Loader needs four
(different types). Rather than reworking the connection model:

- `PortItem` gains an optional `source_node_id` (defaults to
  `self.node.node_id`). Everywhere the scene currently reads
  `source.node.node_id` to resolve a connection's source (`connect_ports`,
  `to_dict`), it instead resolves through this override.
- `BaseNodeItem` gains `owned_node_ids() -> tuple[str, ...]` (default:
  `(self.node_id,)`), used by `_delete_node` to clean up every graph node a
  visual box owns, not just its primary id.

`ObjLoaderNodeItem`:

- "Load Obj..." opens a file dialog, parses with `ncca.ngl.Obj`, converts
  via `arrays_from_obj`, and creates four `LiteralNode`s in the graph. Its
  four output `PortItem`s point at those ids via `source_node_id`.
  `owned_node_ids()` returns all four.
- Shows the loaded filename and a vertex/face count readout on the node
  body (styled like `OutputNodeItem`'s result text).
- `canvas.to_dict`/`from_dict` gain an `"obj_loader"` node kind that
  serializes the raw parsed arrays (not just the file path), so a saved
  graph reloads correctly even if the source `.obj` has moved.

## Mesh Viewer node

Inputs: Vertices, Faces (required), UVs, Normals, Colour (optional —
`Vec4`, defaults to a mid-grey constant when unwired). Node-local (non-graph)
config: Shading mode (`Solid Colour` / `Diffuse`) and a Wireframe checkbox,
serialized in `to_dict` alongside the node like `ValueNodeItem.components`.

Rendering:

- On every `update_outputs()` pass, evaluate the five inputs via
  `evaluate_mesh_viewer`, rebuild an `Obj` via `obj_from_arrays`, and hand it
  to the node's view(s) to redraw. Diffuse mode without Normals wired raises
  the same `GraphError` "needs input" pattern as other nodes and is shown as
  node status text instead of rendering.
- Actual VAO construction reuses `Obj.create_vao()` (already builds the
  interleaved position/normal/uv buffer at the attribute locations the
  built-in shaders expect) rather than hand-rolling a VBO layout.
- `DefaultShader.COLOUR` for Solid, `DefaultShader.DIFFUSE` for lit;
  Wireframe independently toggles `glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)`
  and composes with either shading mode.
- **Dual view**: a small embedded `QOpenGLWidget` lives in the node body
  (interactive — `QGraphicsProxyWidget` captures its own mouse events, so it
  won't fight node dragging, the same mechanism the existing spin-box
  editors already rely on) plus a "Pop Out" button opening a standalone
  `QOpenGLWindow` using the existing `PySideEventHandlingMixin` (the
  canonical arcball camera pattern used by every other PyNGL demo). Each
  view keeps an independent camera. Shared render/merge logic lives in one
  mixin/helper used by both.

  Known risk: two live GL contexts (embedded widget + popup window) both
  driving `ShaderLib`/`DefaultShader` is untested territory for this
  codebase (every existing demo runs a single context). This will be
  verified by actually running the app; if context-sharing proves
  unreliable, fall back to one active view at a time and flag it back.

## OBJ triangulation

OBJ files with non-triangular faces are rejected with a `GraphError` (same
message style as `ColourObj`/`Obj2Numpy`'s `is_triangular()` check) rather
than auto-fan-triangulated. Matches existing repo convention.

## Testing

- `tests/test_math_graph.py`: new operation handlers (`Mat4 to Mat3`,
  `Inverse`, `Transform Vertices`, `Transform Normals`) and the
  `arrays_from_obj`/`obj_from_arrays` conversion functions, using a small
  fixture `.obj` file — all headless, no Qt/GL required.
- Manual verification: run `uv run MathNodeEditor/main.py`, load a real
  `.obj`, wire transforms, confirm both embedded and popup views render and
  live-update; capture a screenshot for the README per this repo's
  convention.

## Out of scope

- UV matrix transforms (not requested).
- A standalone "Build Mesh" operation node — the Mesh Viewer performs the
  merge internally from its four array inputs plus Colour.
- Auto fan-triangulation of n-gon faces.
- Per-vertex baked colours (`ColourObj`-style) on the Obj Loader.
