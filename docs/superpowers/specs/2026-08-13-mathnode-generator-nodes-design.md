# MathNodeEditor: parameter (generator) nodes

## Goal

Eleven of MathNodeEditor's operation nodes only ever take plain Float/Vec3
parameters (Look At, Perspective, Orthographic, Frustum, Mat4 Translate,
Mat4 Scale, Mat4 Rotate X/Y/Z, Transform, Quaternion from Axis Angle). Today
every one of those parameters needs its own separate Value node wired in,
which clutters the graph for what is really just "type in some numbers".
These eleven should instead carry their own inline spin boxes, with no input
sockets at all. Every other operation (Add, Multiply, Dot, Normalise,
Inverse, the Quaternion combinators, Transform Vertices/Normals, ...) keeps
wired sockets, since their inputs are other computed values rather than raw
numbers.

## Scope

**Becomes a spin-box-only "generator" node:**
Look At, Perspective, Orthographic, Frustum, Mat4 Translate, Mat4 Scale,
Mat4 Rotate X, Mat4 Rotate Y, Mat4 Rotate Z, Transform, Quaternion from Axis
Angle.

**Stays a wired operation node:** everything else — Add, Subtract, Multiply,
Matrix Multiply, Dot, Cross, Normalise, Transpose, Inverse, Mat4 to Mat3,
Quaternion Product, Quaternion Rotate Vector, Quaternion to Mat4, Mat4 to
Quaternion, Quaternion Slerp, Quaternion Conjugate, Quaternion Inverse,
Transform Vertices, Transform Normals.

## Data model (`math_graph.py`)

- New `GeneratorNode` dataclass: `operation: Operation`,
  `parameters: list[tuple[float, ...]]` — one components-tuple per named
  parameter, in `OPERATION_INPUT_NAMES[operation]` order (e.g. Look At
  stores three 3-tuples, for Eye/Target/Up).
- New tables:
  - `GENERATOR_OPERATIONS: frozenset[Operation]` — the eleven above.
  - `OPERATION_PARAMETER_TYPES: dict[Operation, tuple[MathType, ...]]` —
    per-operation tuple of `MathType.FLOAT`/`MathType.VEC3`, one per named
    parameter, for the eleven generator operations only.
  - `GENERATOR_OUTPUT_TYPE: dict[Operation, MathType]` — `MathType.MAT4` for
    ten of them, `MathType.QUATERNION` for Quaternion from Axis Angle.
- `MathGraph.add_generator(operation, parameters) -> str`.
- `MathGraph.set_generator_parameter(node_id, parameter_index, components) -> None`
  — validates the component count against that parameter's `MathType` (reuse
  `_validate_components`).
- `MathGraph.add_operation()` raises `ValueError` if asked to build one of
  the eleven generator operations, with a message pointing at
  `add_generator` instead — there is exactly one way to build a Look At
  node, not two parallel node representations of the same operation.
- `_evaluate` gains a `GeneratorNode` branch: build each parameter via
  `VALUE_CLASSES[parameter_type](*components)`, then
  `apply_operation(node.operation, *values)` — reuses the exact same handler
  functions as the wired path.
- `connect`, `disconnect` and the "skip, no `.inputs`" branch in
  `remove_node` extend their `isinstance(node, (ValueNode, LiteralNode))`
  checks to include `GeneratorNode`.

## Graphics (`graphics_items.py`)

New `GeneratorNodeItem(BaseNodeItem)`:

- No input ports (`input_names=()` passed to `BaseNodeItem.__init__`).
- One output port, coloured by the operation's real result type
  (`TYPE_COLOURS[GENERATOR_OUTPUT_TYPE[operation]]`) rather than the generic
  purple `OperationNodeItem` uses — consistent with how `ValueNodeItem`
  colour-codes its output today.
- One row per named parameter: a label (`Eye`, `FOV`, `Position`, ...)
  followed by 1 spin box (Float parameter) or 3 spin boxes (Vec3 parameter),
  styled the same as `ValueNodeItem`'s spin boxes.
- Exposes `.parameter_names` (parallel to `OperationNodeItem.input_names`)
  and `.spin_box_rows: list[list[QDoubleSpinBox]]` (parallel to
  `ValueNodeItem.spin_boxes`) so callers and tests can read/drive it.
- Edits call an `on_change(parameter_index, components)` callback, mirroring
  `ValueNodeItem.on_change`.
- Node width accounts for the widest label plus up to 3 spin box columns;
  height is `NODE_HEADER_HEIGHT` plus one row per parameter.

New `default_generator_parameters(operation) -> tuple[tuple[float, ...], ...]`
next to the existing `default_components`, supplying teaching-friendly
non-zero defaults so a freshly-added node doesn't immediately show a graph
error or collapse to a degenerate matrix:

| Operation | Defaults |
|---|---|
| Look At | Eye (0, 2, 8), Target (0, 0, 0), Up (0, 1, 0) |
| Perspective | FOV 45, Aspect 1.778, Near 0.1, Far 100 |
| Orthographic | Left −10, Right 10, Bottom −10, Top 10, Near 0.1, Far 100 |
| Frustum | Left −1, Right 1, Bottom −1, Top 1, Near 0.1, Far 100 |
| Mat4 Translate | X 0, Y 0, Z 0 |
| Mat4 Scale | X 1, Y 1, Z 1 |
| Mat4 Rotate X / Y / Z | Angle 0 |
| Transform | Position (0,0,0), Rotation (0,0,0), Scale (1,1,1) |
| Quaternion from Axis Angle | Axis (0, 1, 0), Angle 0 |

## Canvas / palette / window (`canvas.py`, `palette.py`, `node_editor.py`)

- `MathNodeScene.add_generator_node(operation, position=None, parameters=None)`
  — defaults `parameters` from `default_generator_parameters(operation)`,
  mirrors `add_value_node`'s shape.
- `MathNodeScene._generator_changed(node_id, parameter_index, components)`
  — mirrors `_value_changed`, calls `graph.set_generator_parameter` then
  `update_outputs()`.
- `to_dict` gains a branch for `GeneratorNodeItem`: `"kind": "generator"`,
  `"operation": node.operation.name`,
  `"parameters": [[box.value() for box in row] for row in node.spin_box_rows]`.
- `from_dict` gains a matching `"generator"` branch, building the node via
  `add_generator_node(Operation[...], position, tuple(tuple(p) for p in ...))`.
- `palette.py`'s `_operation_catalogue_entries` routes each operation to
  `canvas.add_generator_node` or `canvas.add_operation_node` based on
  `Operation in GENERATOR_OPERATIONS` — no changes to section groupings
  (palette buttons and Tab-menu entries stay exactly where they are; only
  the node type they create changes).
- `node_editor.py` imports and re-exports `GeneratorNodeItem` alongside the
  other graphics item classes.

## Examples

`examples/mvp_demo.json`, `examples/mvp_mesh_demo.json` and
`examples/mesh_pipeline_demo.json` are rewritten: the standalone Value nodes
that used to feed Look At / Perspective / Transform / Mat4 Rotate Y are
dropped, and those operation nodes become `"kind": "generator"` entries
carrying the same numbers inline. The remaining wiring (Look At → Matrix
Multiply, Transform → Transform Vertices, etc.) is unchanged. Net effect:
noticeably fewer nodes and wires in each saved graph.

No backward-compatibility loader is added for the old `"kind": "operation"`
representation of these eleven operations — consistent with this repo's
existing no-compat-shim convention. A hand-saved graph from before this
change that used Look At/Perspective/etc. would need re-wiring by hand to
load again.

## Tests

- `tests/test_math_graph.py`: every test that builds one of the eleven
  operations via `add_operation` + `connect` (Look At, Perspective, Mat4
  Translate/Scale/Rotate X/Y/Z, Transform, Orthographic, Frustum, Quaternion
  from Axis Angle) is rewritten to use `add_generator` and assert on the
  evaluated result directly. A new test covers `add_operation` rejecting a
  generator operation.
- `tests/test_node_editor.py`: palette/menu node-type assertions for Look
  At, Perspective and Quaternion from Axis Angle switch from
  `OperationNodeItem` to `GeneratorNodeItem`; the semantic-names test reads
  `.parameter_names` instead of `.input_names` for those cases; the MVP
  example test looks for a `GeneratorNodeItem` with
  `operation is Operation.TRANSFORM` instead of an `OperationNodeItem`.
- Existing round-trip tests (`to_dict`/`from_dict`, save/load file) continue
  to exercise the default Vec3-multiply example, which is unaffected by
  this change (Multiply stays a wired operation).

## Docs

`MathNodeEditor/README.md` is updated to describe generator nodes and to
list which operations now skip wiring; a fresh screenshot is captured once
the feature is working, per the project's convention of keeping a demo
screenshot alongside its README.
