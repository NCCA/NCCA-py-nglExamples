# MathNodeEditor architecture

`README.md` covers what the editor does; this is about how it's built, for
anyone (including future me) adding a node type. The short version: there's
a plain-Python calculation graph with no Qt in it, a Qt graphics layer that
mirrors it node-for-node, and a thin canvas that keeps the two in sync. Most
new operations only touch the first of those, plus a handful of lookup
dicts scattered across the other files.

## Module map

| File | Owns |
|---|---|
| `math_graph.py` | The calculation graph: node dataclasses, `MathGraph`, every `Operation`'s maths, no Qt import at all |
| `graph_document.py` | Validates a loaded JSON document completely before `canvas.py` is allowed to rebuild the scene from it |
| `node_visuals.py` | Icon symbol + header colour for every node type, and the code that paints a small `QIcon` from one |
| `mesh_view.py` | OpenGL rendering for the Mesh Viewer node's embedded preview and pop-out window |
| `graphics_items.py` | The `QGraphicsItem` subclasses — one per node shape — plus ports and wires |
| `palette.py` | The node catalogue (what's creatable) and the two UI surfaces that read it: the Tab search menu and the side palette |
| `canvas.py` | `MathNodeScene`/`MathNodeView` — glues graph and graphics items together, owns JSON save/load and the zoom/pan/grid view |
| `node_editor.py` | `MathNodeWindow` — File menu, settings, the top-level widget layout |
| `main.py` | Entry point: argument parsing, `QSurfaceFormat`, `DebugApplication` |

`math_graph.py` depends on nothing else here. `graphics_items.py` depends
on `math_graph`, `node_visuals` and `mesh_view` (a node's editor widgets
need a style and, for Mesh Viewer, a live GL preview). `palette.py` only
depends on `math_graph` and `node_visuals` — it builds factory closures
that call methods on whatever `MathNodeScene` it's given, so it doesn't
need to know about `graphics_items` at all. `canvas.py` is where
everything meets, and `node_editor.py`/`main.py` sit on top of that.
Nothing below `canvas.py` imports anything that shows a window, which is
what keeps `math_graph.py` and `graph_document.py` testable without a
display — see `tests/test_math_graph.py` and `tests/test_graph_document.py`,
which import them directly and never touch Qt.

## The calculation graph (`math_graph.py`)

A node is one of a handful of frozen dataclasses — `ValueNode`,
`GeneratorNode`, `OperationNode`, `OutputNode`, `LiteralNode`,
`MeshViewerNode` — collected in the `GraphNode` union. `MathGraph` is just
`dict[str, GraphNode]` plus the methods to add, edit, connect and evaluate
them; it has no idea any of this is going to be drawn on screen.

`Operation` is a single `Enum`, not a class hierarchy — there's no
`AddOperation` or `CrossOperation` class. Everything specific to one
operation (its input names, its arity, what it actually computes, what
colour it gets) lives in a dict keyed by that one enum member, spread across
this file and two others. That's a deliberate trade against having forty
small operation classes: adding an operation means filling in a few table
rows rather than writing a class, at the cost of the tables needing to
agree with each other. `apply_operation` is the one place that turns an
`Operation` back into a Python callable, via `_OPERATION_HANDLERS`.

Two flavours of operation node exist because not every PyNGL call takes
graph values as input:

- **Wired operations** (`OperationNode`) — Add, Cross, Inverse, and most of
  the rest — take their inputs from other nodes over wires. `OPERATION_ARITY`
  (how many sockets) is derived automatically from `len(OPERATION_INPUT_NAMES[op])`,
  so you never set arity by hand.
- **Generator operations** (`GeneratorNode`) — Look At, Perspective,
  Transform, the axis-angle Quaternion builder — take plain typed-in
  numbers instead, because their PyNGL parameters are just floats and Vec3s
  with no meaningful upstream node to wire from. `GENERATOR_OPERATIONS`
  marks which operations these are; `OPERATION_PARAMETER_TYPES` gives each
  parameter's `MathType` so the UI knows how many spin boxes to draw.

`MathGraph.evaluate(node_id)` walks the graph recursively and lazily —
nothing is computed until an `Output` or `Mesh Viewer` node asks for it,
and nothing is cached, so editing a `Value` node's spin box just needs
`update_outputs()` to be called again. The `active_nodes` set passed through
`_evaluate` is how a cycle gets caught (`GraphError("The graph contains a
cycle")`) instead of a `RecursionError`.

The mesh-array dataclasses (`VertexArray`, `FaceArray`, `UVArray`,
`NormalArray`) exist because a mesh doesn't fit the "one value, one
`MathValue`" shape everything else uses — the `Obj Loader` node produces
four of them from one file, and `Mesh Viewer` merges them back into an
`Obj` via `obj_from_arrays`. This mirrors how `ColourObj` and `Obj2Numpy`
handle Obj data elsewhere in this repo.

## The visual layer (`graphics_items.py`, `node_visuals.py`, `mesh_view.py`)

`BaseNodeItem` is the shared `QGraphicsObject`: it paints the rounded body,
header, icon and input-socket labels, and owns the `PortItem` list. Every
node shape subclasses it and adds whatever editor widgets it needs via a
`QGraphicsProxyWidget`:

- `ValueNodeItem` — a grid of `QDoubleSpinBox`es, one per component.
- `GeneratorNodeItem` — one labelled row of spin boxes per named parameter,
  plus a rotation-order combo box for Transform specifically.
- `OperationNodeItem` — just the wired sockets `BaseNodeItem` already draws;
  nothing extra to build.
- `OutputNodeItem` — a `QGraphicsTextItem` showing `format_value`'s result
  or a `GraphError` message, resized to fit.
- `ObjLoaderNodeItem` — a Load button and four output sockets whose ports
  each carry a different `source_node_id` (see below).
- `MeshViewerNodeItem` — shading/wireframe controls plus an embedded
  `MeshPreviewWidget`.

`PortItem.graph_node_id` normally just returns its owning node's id, but an
output port can be given a different `source_node_id` at construction — this
is how `ObjLoaderNodeItem` gets away with being one visual box wrapping four
separate `LiteralNode`s in the graph (`array_node_ids`). If you're building
another node that produces more than one independent value, this is the
mechanism to reuse rather than inventing a second one. `owned_node_ids()`
is the matching override needed so deletion cleans up all four.

`ConnectionItem` is a cubic Bezier between two ports' scene positions,
recomputed by `update_path()` whenever either node moves —
`BaseNodeItem.itemChange` calls that on every port's connections after a
position change.

`node_visuals.py` is deliberately small and Qt-light: a `NodeVisualStyle`
(icon symbol + `QColor`) per `MathType` and per `Operation`, plus
`catalogue_node_style(label)` which looks a style up by the same string the
palette displays. The palette buttons, the Tab menu entries and the node
headers all call into this one module, so an icon can never drift out of
sync between the sidebar and the node itself — there's only one dict to
edit.

`mesh_view.py` is a separate, smaller world: real OpenGL, not
`QGraphicsItem`s. `MeshRenderState` holds plain data (the evaluated
`MeshViewerInputs`, shading mode, wireframe flag, a version counter) and
nothing GL-specific, because the embedded preview and the pop-out window
are two different GL surfaces that aren't guaranteed to share a context.
`MeshRenderMixin` is the GL-facing half — each view builds its own `Obj`/VAO
from the shared state and rebuilds it only when `state.version` has moved
on. If you add a node that needs its own live GL preview, this
state-versus-per-surface-VAO split is the pattern to copy, not a shortcut
to skip.

## The canvas (`canvas.py`)

`MathNodeScene` is where the graph model and the graphics items actually
meet. It owns both `self.graph: MathGraph` and `self.nodes: dict[str,
BaseNodeItem]`, keyed by the same node id. Every `add_*_node` method
follows the same three steps: ask `self.graph` to create the model node,
build the matching graphics item wired up with `on_change` callbacks that
write back into `self.graph`, then place and register it. `add_value_node`
is the simplest one to read first if you want the pattern;
`add_obj_loader_node` is the one exception, since it creates four
`LiteralNode`s for one visual item.

Dragging a wire is handled by `mousePressEvent`/`mouseMoveEvent`/
`mouseReleaseEvent` on the scene itself (not on the ports), tracking
`self._drag_source` and a dashed preview path; `connect_ports` is what
actually calls `self.graph.connect(...)` and creates the permanent
`ConnectionItem`. `update_outputs()` is the evaluation entry point — it
runs after essentially every edit (a spin box change, a new connection, a
deletion, an Obj file loading) and pushes the result or the caught
`GraphError` onto every `OutputNodeItem`/`MeshViewerNodeItem` in the scene.
There's no dependency tracking to only re-evaluate what changed; the whole
scene re-evaluates every time, which is fine at the size these graphs get
to.

`to_dict`/`from_dict` are the JSON save/load round-trip, gated by
`graph_document.validate_document` on the way in — a malformed file is
rejected *completely* before `from_dict` touches the live scene, and
`node_editor.py`'s `_open_path` restores a snapshot if loading fails partway
through, so a bad file can't leave the canvas half-rebuilt. `SCHEMA_VERSION`
in `graph_document.py` is the version stamp; bump it if a change to a
node's saved shape would break older files, and extend the loader to
still accept the previous shape if you want old graphs to keep opening
(see how `rotation_order` defaults to `xyz` for pre-Transform-rotation-order
files).

`MathNodeView` is the zoom/pan/grid/frame-all camera around the scene, plus
the Tab-key node menu and forwarding mouse-wheel events to a spin box under
the pointer instead of zooming when you're hovering one.

## The window and creation UI (`node_editor.py`, `palette.py`)

`MathNodeWindow` is the outer `QMainWindow`: File menu (New/Open/Save/Save
As), `QSettings`-backed window geometry and "last file open", and loading
the bundled example graph on first run.

`palette.py`'s `NODE_CATALOGUE` is a tuple of `(section title, [(label,
factory), ...])` — one list, built once, consumed by both `NodeCreationMenu`
(the searchable Tab popup) and `NodePalette` (the sidebar buttons). Add a
node to one of the operation-group tuples near the top of the file
(`MATH_OPERATIONS`, `MAT4_OPERATIONS`, `QUATERNION_OPERATIONS`,
`MESH_OPERATIONS`) and it appears in both places for free —
`_operation_catalogue_entries` decides whether to route it through
`add_generator_node` or `add_operation_node` based on whether it's in
`GENERATOR_OPERATIONS`.

## Adding a new operation

`math_graph.py` carries the up-to-date, authoritative checklist as a
comment directly above `class Operation` — read that before making the
change, since it's the thing that gets kept in sync with the code. The
short version, for a **wired** operation like Add or Cross:

1. A member on `Operation` in `math_graph.py`.
2. Its input socket names in `OPERATION_INPUT_NAMES` (also sets its arity).
3. A handler function, registered in `_OPERATION_HANDLERS`.
4. A `NodeVisualStyle` in `node_visuals.OPERATION_NODE_STYLES`.
5. A slot in one of `palette.py`'s operation tuples.

A **generator** operation like Look At or Transform needs those same five
plus:

6. A place in `GENERATOR_OPERATIONS`.
7. Its parameter types in `OPERATION_PARAMETER_TYPES`.
8. Its result type in `GENERATOR_OUTPUT_TYPE`.
9. Starting numbers in `graphics_items.GENERATOR_DEFAULTS`.

Miss one of these and the failure is usually a `KeyError` the first time
the new operation is created or painted, not a silent wrong answer — the
lookup dicts aren't defensively guarded, on purpose, so a gap shows up
immediately in testing rather than shipping. `_normalise`/`_cross` in
`math_graph.py` are a reasonable template for a small wired operation;
`_look_at`/`_transform` for a generator one.

Cover the new operation in `tests/test_math_graph.py` (evaluate it directly,
no Qt needed) and, if it should appear correctly in a saved graph, in
`tests/test_graph_document.py`. `tests/test_node_editor.py` runs under the
offscreen Qt platform (`QT_QPA_PLATFORM=offscreen`, set at the top of that
file) and is where UI-level behaviour — the node actually appearing in the
palette, its sockets, its editors — gets checked.

## Adding a whole new node kind

Rarer, and a bigger job: this is for something structurally different from
an operation, in the shape of `Obj Loader` or `Mesh Viewer` rather than
another entry in the `Operation` enum. It touches:

- A `GraphNode` dataclass and the matching `MathGraph` methods in
  `math_graph.py`.
- A `NODE_KINDS` entry and a `_validate_node`/`_input_count` case in
  `graph_document.py`.
- A `BaseNodeItem` subclass in `graphics_items.py`.
- A style in `node_visuals.py`.
- A catalogue entry in `palette.py`.
- `to_dict`/`from_dict` cases in `canvas.py`.

`Obj Loader` is the template if the new node is a *source* with several
independent outputs; `Mesh Viewer` is the template if it's a *sink* that
renders or otherwise consumes several wired inputs at once.
