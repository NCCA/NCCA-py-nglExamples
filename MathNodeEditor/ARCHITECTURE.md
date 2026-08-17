# MathNodeEditor architecture

This is a quick guide to how I have put the editor together. The
[README](README.md) explains how to use it; this file is mainly for me (or
anyone else) adding another operation later.

The editor has three main parts: a plain Python calculation graph, the Qt
graphics items used to draw it, and a scene which keeps the two in step. I
have kept Qt out of the calculation graph so the maths and saved documents can
be tested without opening a window.

## Files

| File                                     | What it does                                         |
| :--------------------------------------- | :--------------------------------------------------- |
| [`math_graph.py`](math_graph.py)         | Node data, graph connections and all the maths       |
| [`graph_document.py`](graph_document.py) | Checks a saved JSON document before it is loaded     |
| [`node_visuals.py`](node_visuals.py)     | Node colours, symbols and icons                      |
| [`mesh_view.py`](mesh_view.py)           | OpenGL views used by the Mesh Viewer node            |
| [`graphics_items.py`](graphics_items.py) | Qt node, port and connection items                   |
| [`palette.py`](palette.py)               | The side palette and searchable Tab menu             |
| [`canvas.py`](canvas.py)                 | The scene, view, connections and JSON save/load code |
| [`node_editor.py`](node_editor.py)       | The main window, menus and settings                  |
| [`main.py`](main.py)                     | Command line entry point and OpenGL setup            |

The dependency direction is mostly upwards through that table.
`math_graph.py` does not import any of the editor code. `canvas.py` is where
the graph and graphics items meet, then `node_editor.py` puts the finished
scene into a window.

## The graph

[`math_graph.py`](math_graph.py) contains the model. `MathGraph` owns a
dictionary of nodes and has the methods used to create, edit, connect and
evaluate them. It knows nothing about pixels, mouse events or Qt.

The node records are small slotted dataclasses:

- `ValueNode` stores an editable PyNGL value.
- `GeneratorNode` builds a value from numbers typed into the node.
- `OperationNode` takes values from connected nodes.
- `OutputNode` displays the result of one connection.
- `LiteralNode` holds data supplied by something outside the graph, such as an
  Obj file.
- `MeshViewerNode` collects the inputs required by the viewer.

I use one `Operation` enum and a set of lookup tables instead of a class for
every mathematical operation. `OPERATION_INPUT_NAMES` defines the socket
names and is also used to calculate `OPERATION_ARITY`.
`_OPERATION_HANDLERS` maps each enum member to the Python function which does
the work. This does mean the tables need to agree, but I would rather add a
few table entries than forty tiny classes.

There are two sorts of operation node. Normal operations such as Add, Cross
and Inverse get their arguments through wires. Generator operations such as
Look At, Perspective and Transform use spin boxes because their arguments are
usually just a few numbers. `GENERATOR_OPERATIONS` identifies these nodes and
`OPERATION_PARAMETER_TYPES` tells the UI which editors to make.

`MathGraph.evaluate()` walks backwards through the connections when an Output
or Mesh Viewer asks for a value. Results are not cached, so changing a value
and calling `update_outputs()` evaluates the graph again. The `active_nodes`
set catches cycles and turns them into a useful `GraphError` rather than a
`RecursionError`.

Meshes do not fit neatly into the normal one-node, one-value setup. I use
`VertexArray`, `FaceArray`, `UVArray` and `NormalArray` for the four parts of
an Obj mesh. The loader creates them with `arrays_from_obj()` and the viewer
puts them back together with `obj_from_arrays()`.

## Drawing the nodes

[`graphics_items.py`](graphics_items.py) contains the Qt items.
`BaseNodeItem` draws the body, title, icon, labels and ports. The subclasses
only add the controls needed by that particular node:

- `ValueNodeItem` adds a spin box for each component.
- `GeneratorNodeItem` adds an editor row for each parameter (and the rotation
  order menu used by Transform).
- `OperationNodeItem` only needs the ports drawn by the base class.
- `OutputNodeItem` displays the result or a graph error.
- `ObjLoaderNodeItem` has a load button and four outputs.
- `MeshViewerNodeItem` has the render controls and embedded OpenGL preview.

Normally a `PortItem` uses the id of its parent node. The Obj Loader is a
slightly odd case as one box on screen represents four `LiteralNode`s in the
graph. Its output ports therefore have their own `source_node_id`. It also
overrides `owned_node_ids()` so deleting the box removes all four model nodes.
This is the example to follow if I need another node with several independent
outputs.

A `ConnectionItem` is a cubic Bezier path between two ports. When a node moves,
`BaseNodeItem.itemChange()` asks each connected wire to rebuild its path.

The styles live in [`node_visuals.py`](node_visuals.py). Palette buttons, Tab
menu entries and node headers all read the same `NodeVisualStyle` tables, so I
only need to change the colour or symbol in one place.

The Mesh Viewer uses real OpenGL rather than a `QGraphicsItem` painter.
`MeshRenderState` stores the evaluated mesh and display settings, whilst
`MeshRenderMixin` owns the OpenGL work. The embedded preview and pop-out view
are separate GL surfaces and may not share a context, so each builds its own
VAO from the shared state. The version number tells a view when it needs to
rebuild that VAO.

## The scene and window

`MathNodeScene` in [`canvas.py`](canvas.py) owns both the `MathGraph` and a
dictionary of `BaseNodeItem`s. Both dictionaries use the same node ids.
Adding a node normally follows this pattern:

1. Add the model node to `MathGraph`.
2. Make the matching graphics item and connect its callbacks to the model.
3. Put the item into the scene and update the outputs.

`add_value_node()` is the shortest example. `add_obj_loader_node()` is the
exception because it creates four model nodes for one graphics item.

The scene also handles wire dragging. Mouse events create a temporary dashed
path, then `connect_ports()` adds the graph connection and its permanent
`ConnectionItem`. `update_outputs()` evaluates every Output and Mesh Viewer
after an edit. There is no dependency cache; these are small teaching graphs
so evaluating the scene again keeps the code much easier to follow.

`to_dict()` and `from_dict()` handle the JSON representation.
`validate_document()` checks the complete file before the live scene is
changed. The window also restores a snapshot if rebuilding the scene fails
part-way through. `SCHEMA_VERSION` is the saved file version; it needs to move
when a change would make an older node record incompatible. Where it is easy,
I keep old files working (Transform uses `xyz` if `rotation_order` is absent).

`MathNodeView` adds the grid, zooming, panning, frame-all behaviour and the Tab
menu. It forwards wheel events to a spin box under the pointer, otherwise the
wheel zooms the view.

[`palette.py`](palette.py) has one `NODE_CATALOGUE` which is used by the side
palette and the Tab menu. The operation groups near the top of the file decide
where a node appears. `_operation_catalogue_entries()` checks
`GENERATOR_OPERATIONS` and calls the correct scene factory.

Finally, [`node_editor.py`](node_editor.py) contains `MathNodeWindow`. It owns
the File menu, window layout and `QSettings` values. [`main.py`](main.py)
parses the command line, sets the OpenGL surface format and starts the Qt
application.

## Adding an operation

There is a short checklist above `Operation` in
[`math_graph.py`](math_graph.py). That comment should remain the authoritative
one as it sits beside the code, but the process is as follows.

For a wired operation such as Add or Cross I add:

1. A member to `Operation`.
2. Its socket names to `OPERATION_INPUT_NAMES`.
3. A handler and an entry in `_OPERATION_HANDLERS`.
4. A style in `node_visuals.OPERATION_NODE_STYLES`.
5. An entry in one of the operation groups in `palette.py`.

For a generator such as Look At or Transform I also add:

6. The operation to `GENERATOR_OPERATIONS`.
7. Its parameter types to `OPERATION_PARAMETER_TYPES`.
8. Its result type to `GENERATOR_OUTPUT_TYPE`.
9. Useful starting values to `graphics_items.GENERATOR_DEFAULTS`.

Missing entries usually fail with a `KeyError` as soon as the node is created,
which is useful here. `_normalise()` and `_cross()` are small wired examples;
`_look_at()` and `_transform()` show the generator version.

The maths belongs in [`tests/test_math_graph.py`](tests/test_math_graph.py).
Saved-file behaviour belongs in
[`tests/test_graph_document.py`](tests/test_graph_document.py), and anything
which needs the real node UI belongs in
[`tests/test_node_editor.py`](tests/test_node_editor.py). The Qt tests set the
offscreen platform themselves, so they do not need a visible desktop.

## Adding a different node type

A new node type is a larger change. This is only needed for something shaped
differently from the operations above, like Obj Loader or Mesh Viewer. It
needs:

- A dataclass and `MathGraph` methods in `math_graph.py`.
- A `NODE_KINDS` entry plus validation in `graph_document.py`.
- A `BaseNodeItem` subclass in `graphics_items.py`.
- A style in `node_visuals.py`.
- A catalogue entry in `palette.py`.
- Save and load cases in `canvas.py`.

Obj Loader is the useful example for a source with several outputs. Mesh
Viewer is the example for a sink which consumes several inputs and does
something other than display a number.
