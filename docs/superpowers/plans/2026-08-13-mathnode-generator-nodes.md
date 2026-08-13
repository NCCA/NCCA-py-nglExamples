# MathNodeEditor Generator Nodes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the eleven MathNodeEditor operations whose parameters are plain Float/Vec3 numbers (Look At, Perspective, Orthographic, Frustum, Mat4 Translate, Mat4 Scale, Mat4 Rotate X/Y/Z, Transform, Quaternion from Axis Angle) their own inline spin boxes instead of wired input sockets, and update every example graph, test and doc that touches them.

**Architecture:** A new `GeneratorNode` graph-model class (math_graph.py) and matching `GeneratorNodeItem` graphics class (graphics_items.py) sit alongside the existing `OperationNode`/`OperationNodeItem` pair. The eleven operations move to the new pair; everything else (Add, Multiply, Quaternion Slerp, Transform Vertices, ...) is untouched. The palette routes each operation to the right node type based on membership in a new `GENERATOR_OPERATIONS` set.

**Tech Stack:** Python 3.13, PySide6 (Qt Widgets/Graphics View), `ncca.ngl` maths types, pytest, `uv`, `ruff`.

## Global Constraints

- Run everything through `uv` (`uv run pytest`, `uv run ruff ...`) — this project has no other supported entry point.
- Lint/format with `ruff`: `uv run ruff check MathNodeEditor` and `uv run ruff format MathNodeEditor` must both be clean before the final commit.
- No backward-compatibility shim for the old `"kind": "operation"` JSON representation of these eleven operations — this repo doesn't carry compatibility hacks (CLAUDE.md).
- Docs/README prose must go through the `jon-writing-style` skill (mandatory per the user's global CLAUDE.md for all documentation).
- Every demo folder keeps a `README.md` and a preview screenshot in sync with what the demo actually does (project CLAUDE.md convention).
- All work happens on branch `agent/mathnode-generators` inside the worktree at `.worktrees/mathnode-generators` (already created). Never commit to `main` or `Version1.0`.
- Commit after every task with a conventional-commit message; run the affected tests before each commit.

---

## File Map

| File | Responsibility |
|---|---|
| `MathNodeEditor/math_graph.py` | `GeneratorNode` dataclass, `GENERATOR_OPERATIONS`/`OPERATION_PARAMETER_TYPES`/`GENERATOR_OUTPUT_TYPE` tables, `MathGraph.add_generator`/`set_generator_parameter`, `add_operation` guard, `_evaluate` branch, `connect`/`disconnect`/`remove_node` updates |
| `MathNodeEditor/graphics_items.py` | `GeneratorNodeItem` widget, `default_generator_parameters` |
| `MathNodeEditor/canvas.py` | `MathNodeScene.add_generator_node`/`_generator_changed`, `to_dict`/`from_dict` support for `"kind": "generator"` |
| `MathNodeEditor/palette.py` | Routes generator operations to `add_generator_node` in the shared catalogue |
| `MathNodeEditor/node_editor.py` | Re-exports `GeneratorNodeItem` |
| `MathNodeEditor/examples/mvp_demo.json`, `mvp_mesh_demo.json`, `mesh_pipeline_demo.json` | Rewritten to the new node kind |
| `MathNodeEditor/README.md`, `MathNodeEditor/MathNodeEditor.png` | Docs/screenshot refresh |
| `MathNodeEditor/tests/test_math_graph.py`, `tests/test_node_editor.py` | New + migrated tests |
| `docs/agent-sessions/2026-08-13-session.md` | Session summary (CLAUDE.md session-handling rule) |

---

### Task 1: `GeneratorNode` data model and `MathGraph` API

**Files:**
- Modify: `MathNodeEditor/math_graph.py:117-120` (insert new tables after `OPERATION_ARITY`), `math_graph.py:537-543` (insert `GeneratorNode` dataclass after `ValueNode`), `math_graph.py:589-591` (`GraphNode` alias), `math_graph.py:614-616` (`add_operation`), `math_graph.py:645-657` (`connect`/`disconnect`), `math_graph.py:659-667` (`remove_node`), `math_graph.py:712-717` (`_evaluate`)
- Test: `MathNodeEditor/tests/test_math_graph.py`

**Interfaces:**
- Produces: `GeneratorNode(operation: Operation, parameters: list[tuple[float, ...]])`; `GENERATOR_OPERATIONS: frozenset[Operation]`; `OPERATION_PARAMETER_TYPES: dict[Operation, tuple[MathType, ...]]`; `GENERATOR_OUTPUT_TYPE: dict[Operation, MathType]`; `MathGraph.add_generator(operation: Operation, parameters: tuple[tuple[float, ...], ...]) -> str`; `MathGraph.set_generator_parameter(node_id: str, parameter_index: int, components: tuple[float, ...]) -> None`. `MathGraph.add_operation` now raises `ValueError` for any operation in `GENERATOR_OPERATIONS`.

- [ ] **Step 1: Write the failing tests**

Add to `MathNodeEditor/tests/test_math_graph.py`, after `test_incompatible_operation_inputs_report_a_graph_error` (uses the module's existing `look_at`/`perspective`/`Mat4`/`Vec3` imports, already present at the top of the file):

```python
def test_generator_node_builds_a_mat4_from_vec3_parameters() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    operation = graph.add_generator(
        graph_module.Operation.LOOK_AT,
        ((0.0, 2.0, 5.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )

    result = graph.evaluate(operation)

    expected = look_at(Vec3(0.0, 2.0, 5.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
    assert isinstance(result, Mat4)
    assert result.to_list() == pytest.approx(expected.to_list())


def test_generator_node_builds_a_mat4_from_float_parameters() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    operation = graph.add_generator(
        graph_module.Operation.PERSPECTIVE,
        ((45.0,), (16.0 / 9.0,), (0.1,), (100.0,)),
    )

    result = graph.evaluate(operation)

    expected = perspective(45.0, 16.0 / 9.0, 0.1, 100.0)
    assert isinstance(result, Mat4)
    assert result.to_list() == pytest.approx(expected.to_list())


def test_set_generator_parameter_updates_the_result() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    operation = graph.add_generator(graph_module.Operation.MAT4_ROTATE_Y, ((0.0,),))

    graph.set_generator_parameter(operation, 0, (45.0,))

    result = graph.evaluate(operation)
    assert result.to_list() == pytest.approx(Mat4.rotate_y(45.0).to_list())


def test_add_operation_rejects_a_generator_operation() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()

    with pytest.raises(ValueError, match="add_generator"):
        graph.add_operation(graph_module.Operation.LOOK_AT)


def test_generator_node_rejects_wired_connections() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    value_node = graph.add_value(graph_module.MathType.VEC3, (0.0, 0.0, 0.0))
    operation = graph.add_generator(
        graph_module.Operation.LOOK_AT,
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )

    with pytest.raises(ValueError, match="do not have inputs"):
        graph.connect(value_node, operation, 0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest MathNodeEditor/tests/test_math_graph.py -k generator -v`
Expected: FAIL — `AttributeError: 'MathGraph' object has no attribute 'add_generator'`

- [ ] **Step 3: Add the generator tables, just after the `OPERATION_ARITY` block (`math_graph.py:117-120`)**

```python
GENERATOR_OPERATIONS: frozenset[Operation] = frozenset(
    {
        Operation.LOOK_AT,
        Operation.PERSPECTIVE,
        Operation.ORTHO,
        Operation.FRUSTUM,
        Operation.MAT4_TRANSLATE,
        Operation.MAT4_SCALE,
        Operation.MAT4_ROTATE_X,
        Operation.MAT4_ROTATE_Y,
        Operation.MAT4_ROTATE_Z,
        Operation.TRANSFORM,
        Operation.QUATERNION_FROM_AXIS_ANGLE,
    }
)
"""Operations whose parameters are typed in directly, with no wired inputs."""

OPERATION_PARAMETER_TYPES: dict[Operation, tuple[MathType, ...]] = {
    Operation.LOOK_AT: (MathType.VEC3, MathType.VEC3, MathType.VEC3),
    Operation.PERSPECTIVE: (
        MathType.FLOAT,
        MathType.FLOAT,
        MathType.FLOAT,
        MathType.FLOAT,
    ),
    Operation.ORTHO: (MathType.FLOAT,) * 6,
    Operation.FRUSTUM: (MathType.FLOAT,) * 6,
    Operation.MAT4_TRANSLATE: (MathType.FLOAT, MathType.FLOAT, MathType.FLOAT),
    Operation.MAT4_SCALE: (MathType.FLOAT, MathType.FLOAT, MathType.FLOAT),
    Operation.MAT4_ROTATE_X: (MathType.FLOAT,),
    Operation.MAT4_ROTATE_Y: (MathType.FLOAT,),
    Operation.MAT4_ROTATE_Z: (MathType.FLOAT,),
    Operation.TRANSFORM: (MathType.VEC3, MathType.VEC3, MathType.VEC3),
    Operation.QUATERNION_FROM_AXIS_ANGLE: (MathType.VEC3, MathType.FLOAT),
}
"""The MathType of each named parameter, in OPERATION_INPUT_NAMES order."""

GENERATOR_OUTPUT_TYPE: dict[Operation, MathType] = {
    operation: MathType.MAT4 for operation in GENERATOR_OPERATIONS
}
GENERATOR_OUTPUT_TYPE[Operation.QUATERNION_FROM_AXIS_ANGLE] = MathType.QUATERNION
"""The MathType each generator operation's output socket should be coloured as."""
```

- [ ] **Step 4: Add the `GeneratorNode` dataclass, immediately after the `ValueNode` dataclass (`math_graph.py:537-543`)**

```python
@dataclass(slots=True)
class GeneratorNode:
    """An operation node whose parameters are typed in, not wired.

    Suits operations like Look At, Perspective and Transform, whose PyNGL
    inputs are just labelled Float/Vec3 numbers rather than other computed
    values, so there is nothing meaningful to connect a wire to.
    """

    operation: Operation
    parameters: list[tuple[float, ...]]
```

- [ ] **Step 5: Add `GeneratorNode` to the `GraphNode` type alias (`math_graph.py:589-591`)**

```python
GraphNode: TypeAlias = (
    ValueNode | OperationNode | OutputNode | LiteralNode | MeshViewerNode | GeneratorNode
)
```

- [ ] **Step 6: Guard `add_operation` and add `add_generator`/`set_generator_parameter` (`math_graph.py:614-616`, right after `add_value`)**

Replace:

```python
    def add_operation(self, operation: Operation) -> str:
        """Add an operation node to the graph."""
        return self._add_node(OperationNode(operation))
```

with:

```python
    def add_operation(self, operation: Operation) -> str:
        """Add a wired operation node to the graph."""
        if operation in GENERATOR_OPERATIONS:
            raise ValueError(
                f"{operation.value} has no wired inputs; use add_generator instead"
            )
        return self._add_node(OperationNode(operation))

    def add_generator(
        self, operation: Operation, parameters: tuple[tuple[float, ...], ...]
    ) -> str:
        """Add a parameter node (Look At, Perspective, ...) to the graph."""
        parameter_types = OPERATION_PARAMETER_TYPES[operation]
        for parameter_type, components in zip(parameter_types, parameters):
            _validate_components(parameter_type, components)
        return self._add_node(GeneratorNode(operation, [tuple(p) for p in parameters]))

    def set_generator_parameter(
        self, node_id: str, parameter_index: int, components: tuple[float, ...]
    ) -> None:
        """Replace one parameter's components on a generator node."""
        node = self._nodes[node_id]
        if not isinstance(node, GeneratorNode):
            raise ValueError("Only generator nodes store editable parameters")
        parameter_type = OPERATION_PARAMETER_TYPES[node.operation][parameter_index]
        _validate_components(parameter_type, components)
        node.parameters[parameter_index] = tuple(components)
```

- [ ] **Step 7: Extend the "no inputs" checks to `GeneratorNode` (`math_graph.py:645-667`)**

In `connect`, `disconnect` and `remove_node`, change every occurrence of:

```python
        if isinstance(target, (ValueNode, LiteralNode)):
```

(in `connect` and `disconnect`) to:

```python
        if isinstance(target, (ValueNode, LiteralNode, GeneratorNode)):
```

and in `remove_node`, change:

```python
            if isinstance(node, (ValueNode, LiteralNode)):
                continue
```

to:

```python
            if isinstance(node, (ValueNode, LiteralNode, GeneratorNode)):
                continue
```

- [ ] **Step 8: Evaluate `GeneratorNode` (`math_graph.py:712-713`, right after the `LiteralNode` branch)**

Insert immediately after `if isinstance(node, LiteralNode): return node.value`:

```python
            if isinstance(node, GeneratorNode):
                parameter_types = OPERATION_PARAMETER_TYPES[node.operation]
                values = tuple(
                    VALUE_CLASSES[parameter_type](*components)
                    for parameter_type, components in zip(
                        parameter_types, node.parameters
                    )
                )
                return apply_operation(node.operation, *values)
```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `uv run pytest MathNodeEditor/tests/test_math_graph.py -k generator -v`
Expected: PASS (5 passed)

- [ ] **Step 10: Commit**

```bash
cd /Volumes/teaching/Code/PyNGLDemos/.worktrees/mathnode-generators
git add MathNodeEditor/math_graph.py MathNodeEditor/tests/test_math_graph.py
git commit -m "feat: add GeneratorNode model for parameter-only operations"
```

---

### Task 2: Migrate existing wired-input tests for the eleven operations

**Files:**
- Modify: `MathNodeEditor/tests/test_math_graph.py`

**Interfaces:**
- Consumes: `MathGraph.add_generator` from Task 1.

- [ ] **Step 1: Replace `test_look_at_node_builds_a_mat4_from_three_vec3_inputs` (`test_math_graph.py:121-136`) with:**

```python
def test_look_at_node_builds_a_mat4_from_three_vec3_inputs() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    operation = graph.add_generator(
        graph_module.Operation.LOOK_AT,
        ((0.0, 2.0, 5.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )

    result = graph.evaluate(operation)

    expected = look_at(Vec3(0.0, 2.0, 5.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
    assert isinstance(result, Mat4)
    assert result.to_list() == pytest.approx(expected.to_list())
```

- [ ] **Step 2: Delete `test_look_at_node_reports_semantic_input_names` (`test_math_graph.py:139-149`)**

It tested the "missing wired input" error path, which no longer exists for Look At — a generator node's parameters always have values.

- [ ] **Step 3: Replace `test_perspective_node_builds_a_mat4_from_four_float_inputs` (`test_math_graph.py:152-167`) with:**

```python
def test_perspective_node_builds_a_mat4_from_four_float_inputs() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    operation = graph.add_generator(
        graph_module.Operation.PERSPECTIVE,
        ((45.0,), (16.0 / 9.0,), (0.1,), (100.0,)),
    )

    result = graph.evaluate(operation)

    expected = perspective(45.0, 16.0 / 9.0, 0.1, 100.0)
    assert isinstance(result, Mat4)
    assert result.to_list() == pytest.approx(expected.to_list())
```

- [ ] **Step 4: Replace `test_perspective_node_reports_invalid_clip_planes` (`test_math_graph.py:170-182`) with:**

```python
def test_perspective_node_reports_invalid_clip_planes() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    operation = graph.add_generator(
        graph_module.Operation.PERSPECTIVE,
        ((45.0,), (1.0,), (1.0,), (1.0,)),
    )

    with pytest.raises(graph_module.GraphError, match="Perspective failed"):
        graph.evaluate(operation)
```

- [ ] **Step 5: Replace `test_mat4_transform_constructor_nodes` (`test_math_graph.py:185-212`) with:**

```python
@pytest.mark.parametrize(
    ("operation_name", "inputs", "expected"),
    [
        ("MAT4_TRANSLATE", (2.0, 3.0, 4.0), Mat4.translate(2.0, 3.0, 4.0)),
        ("MAT4_SCALE", (2.0, 3.0, 4.0), Mat4.scale(2.0, 3.0, 4.0)),
        ("MAT4_ROTATE_X", (30.0,), Mat4.rotate_x(30.0)),
        ("MAT4_ROTATE_Y", (45.0,), Mat4.rotate_y(45.0)),
        ("MAT4_ROTATE_Z", (60.0,), Mat4.rotate_z(60.0)),
    ],
)
def test_mat4_transform_constructor_nodes(
    operation_name: str,
    inputs: tuple[float, ...],
    expected: Mat4,
) -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    parameters = tuple((component,) for component in inputs)
    operation = graph.add_generator(graph_module.Operation[operation_name], parameters)

    result = graph.evaluate(operation)

    assert result.to_list() == pytest.approx(expected.to_list())
```

- [ ] **Step 6: Replace `test_transform_node_builds_a_model_matrix_from_position_rotation_scale` (`test_math_graph.py:215-233`) with:**

```python
def test_transform_node_builds_a_model_matrix_from_position_rotation_scale() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    operation = graph.add_generator(
        graph_module.Operation.TRANSFORM,
        ((1.0, 2.0, 3.0), (0.0, 30.0, 0.0), (2.0, 2.0, 2.0)),
    )

    result = graph.evaluate(operation)

    expected = Transform()
    expected.set_position(Vec3(1.0, 2.0, 3.0))
    expected.set_rotation(Vec3(0.0, 30.0, 0.0))
    expected.set_scale(Vec3(2.0, 2.0, 2.0))
    assert isinstance(result, Mat4)
    assert result.to_list() == pytest.approx(expected.matrix().to_list())
```

- [ ] **Step 7: Delete `test_transform_node_reports_semantic_input_names` (`test_math_graph.py:236-244`)**

Same rationale as Step 2 — Transform's parameters can no longer be left unwired.

- [ ] **Step 8: Replace `test_ortho_node_builds_an_orthographic_mat4` (`test_math_graph.py:247-261`) with:**

```python
def test_ortho_node_builds_an_orthographic_mat4() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    components = (-10.0, 10.0, -5.0, 5.0, 0.1, 100.0)
    parameters = tuple((component,) for component in components)
    operation = graph.add_generator(graph_module.Operation.ORTHO, parameters)

    result = graph.evaluate(operation)

    assert result.to_list() == pytest.approx(ortho(*components).to_list())
```

- [ ] **Step 9: Replace `test_frustum_node_builds_a_projection_mat4` (`test_math_graph.py:264-278`) with:**

```python
def test_frustum_node_builds_a_projection_mat4() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    components = (-1.0, 1.0, -0.5, 0.5, 0.1, 100.0)
    parameters = tuple((component,) for component in components)
    operation = graph.add_generator(graph_module.Operation.FRUSTUM, parameters)

    result = graph.evaluate(operation)

    assert result.to_list() == pytest.approx(frustum(*components).to_list())
```

- [ ] **Step 10: Replace `test_axis_angle_node_builds_a_quaternion` (`test_math_graph.py:281-294`) with:**

```python
def test_axis_angle_node_builds_a_quaternion() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    operation = graph.add_generator(
        graph_module.Operation.QUATERNION_FROM_AXIS_ANGLE,
        ((0.0, 1.0, 0.0), (90.0,)),
    )

    result = graph.evaluate(operation)

    expected = Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), 90.0)
    assert isinstance(result, Quaternion)
    assert result.to_list() == pytest.approx(expected.to_list())
```

- [ ] **Step 11: Run the full math_graph test file**

Run: `uv run pytest MathNodeEditor/tests/test_math_graph.py -v`
Expected: PASS, no failures, no leftover references to the deleted tests.

- [ ] **Step 12: Commit**

```bash
git add MathNodeEditor/tests/test_math_graph.py
git commit -m "test: migrate generator-operation tests off wired add_operation"
```

---

### Task 3: `GeneratorNodeItem` graphics widget and canvas wiring

**Files:**
- Modify: `MathNodeEditor/graphics_items.py` (imports, new `GENERATOR_DEFAULTS`/`default_generator_parameters`, new `GeneratorNodeItem` class after `ValueNodeItem`)
- Modify: `MathNodeEditor/canvas.py:41-52` (imports), `canvas.py:120-131` (insert `add_generator_node` after `add_operation_node`), `canvas.py:202-205` (insert `_generator_changed` after `_value_changed`)
- Test: `MathNodeEditor/tests/test_node_editor.py`

**Interfaces:**
- Consumes: `OPERATION_INPUT_NAMES`, `OPERATION_PARAMETER_TYPES`, `GENERATOR_OUTPUT_TYPE`, `TYPE_SHAPES`, `TYPE_COLOURS` from `math_graph`/`graphics_items`; `MathGraph.add_generator`/`set_generator_parameter` from Task 1.
- Produces: `GeneratorNodeItem(node_id, operation, parameters, on_change)` with `.operation`, `.parameter_names: tuple[str, ...]`, `.spin_box_rows: list[list[QDoubleSpinBox]]`, `.input_ports == []`; `default_generator_parameters(operation) -> tuple[tuple[float, ...], ...]`; `MathNodeScene.add_generator_node(operation, position=None, parameters=None) -> GeneratorNodeItem`.

- [ ] **Step 1: Write the failing tests**

Add to `MathNodeEditor/tests/test_node_editor.py`, after `test_new_quaternion_node_starts_as_the_identity`:

```python
def test_generator_node_has_no_input_ports(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)

    node = window.canvas.add_generator_node(node_editor.Operation.LOOK_AT)

    assert node.input_ports == []
    assert node.parameter_names == ("Eye", "Target", "Up")
    assert [len(row) for row in node.spin_box_rows] == [3, 3, 3]
    window.close()


def test_generator_node_starts_with_teaching_friendly_defaults(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)

    node = window.canvas.add_generator_node(node_editor.Operation.PERSPECTIVE)

    values = [box.value() for row in node.spin_box_rows for box in row]
    assert values == pytest.approx([45.0, 1.778, 0.1, 100.0])
    window.close()


def test_editing_a_generator_spin_box_updates_downstream_output(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    look_at_node = window.canvas.add_generator_node(node_editor.Operation.LOOK_AT)
    output_node = window.canvas.add_output_node()
    assert look_at_node.output_port is not None
    window.canvas.connect_ports(look_at_node.output_port, output_node.input_ports[0])
    application.processEvents()
    before = window.canvas.output_texts()[0]

    look_at_node.spin_box_rows[0][1].setValue(9.0)
    application.processEvents()

    assert window.canvas.output_texts()[0] != before
    assert window.canvas.output_texts()[0].startswith("Mat4")
    window.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest MathNodeEditor/tests/test_node_editor.py -k generator -v`
Expected: FAIL — `AttributeError: 'MathNodeScene' object has no attribute 'add_generator_node'`

- [ ] **Step 3: Add `QLabel` to the Qt widgets import and `GENERATOR_OUTPUT_TYPE`/`OPERATION_PARAMETER_TYPES` to the math_graph import, at the top of `graphics_items.py`**

Change:

```python
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsProxyWidget,
    QGraphicsSceneHoverEvent,
    QGraphicsTextItem,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QStyle,
    QStyleOptionGraphicsItem,
    QWidget,
)

from .math_graph import (
    MESH_VIEWER_INPUT_NAMES,
    OPERATION_ARITY,
    OPERATION_INPUT_NAMES,
    TYPE_SHAPES,
    MathType,
    Operation,
)
```

to:

```python
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsProxyWidget,
    QGraphicsSceneHoverEvent,
    QGraphicsTextItem,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QStyleOptionGraphicsItem,
    QWidget,
)

from .math_graph import (
    GENERATOR_OUTPUT_TYPE,
    MESH_VIEWER_INPUT_NAMES,
    OPERATION_ARITY,
    OPERATION_INPUT_NAMES,
    OPERATION_PARAMETER_TYPES,
    TYPE_SHAPES,
    MathType,
    Operation,
)
```

- [ ] **Step 4: Add `GENERATOR_DEFAULTS`/`default_generator_parameters`, right after `default_components` in `graphics_items.py`**

```python
GENERATOR_DEFAULTS: dict[Operation, tuple[tuple[float, ...], ...]] = {
    Operation.LOOK_AT: ((0.0, 2.0, 8.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    Operation.PERSPECTIVE: ((45.0,), (1.778,), (0.1,), (100.0,)),
    Operation.ORTHO: ((-10.0,), (10.0,), (-10.0,), (10.0,), (0.1,), (100.0,)),
    Operation.FRUSTUM: ((-1.0,), (1.0,), (-1.0,), (1.0,), (0.1,), (100.0,)),
    Operation.MAT4_TRANSLATE: ((0.0,), (0.0,), (0.0,)),
    Operation.MAT4_SCALE: ((1.0,), (1.0,), (1.0,)),
    Operation.MAT4_ROTATE_X: ((0.0,),),
    Operation.MAT4_ROTATE_Y: ((0.0,),),
    Operation.MAT4_ROTATE_Z: ((0.0,),),
    Operation.TRANSFORM: ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    Operation.QUATERNION_FROM_AXIS_ANGLE: ((0.0, 1.0, 0.0), (0.0,)),
}


def default_generator_parameters(operation: Operation) -> tuple[tuple[float, ...], ...]:
    """Return teaching-friendly default parameters for a new generator node."""
    return GENERATOR_DEFAULTS[operation]
```

- [ ] **Step 5: Add the `GeneratorNodeItem` class, immediately after `ValueNodeItem` (before `class OperationNodeItem`) in `graphics_items.py`**

```python
class GeneratorNodeItem(BaseNodeItem):
    """An operation node whose named parameters are typed in, not wired.

    Used for operations such as Look At, Perspective and Transform, whose
    PyNGL parameters are plain Float/Vec3 numbers rather than other
    computed values (see GeneratorNode in math_graph).
    """

    ROW_HEIGHT = 30.0

    def __init__(
        self,
        node_id: str,
        operation: Operation,
        parameters: tuple[tuple[float, ...], ...],
        on_change: Callable[[int, tuple[float, ...]], None],
    ) -> None:
        """Create one labelled spin-box row per named parameter."""
        parameter_names = OPERATION_INPUT_NAMES[operation]
        parameter_types = OPERATION_PARAMETER_TYPES[operation]
        label_font = QFont()
        label_font.setPointSize(9)
        label_width = max(
            QFontMetrics(label_font).horizontalAdvance(name)
            for name in parameter_names
        )
        max_columns = max(TYPE_SHAPES[math_type][1] for math_type in parameter_types)
        width = max(200.0, label_width + 16.0 + max_columns * 76.0 + 24.0)
        height = NODE_HEADER_HEIGHT + len(parameter_names) * self.ROW_HEIGHT + 20.0
        output_type = GENERATOR_OUTPUT_TYPE[operation]
        super().__init__(
            node_id, operation.value, width, height, (), TYPE_COLOURS[output_type]
        )
        self.operation = operation
        self.parameter_names = parameter_names
        self.on_change = on_change
        self.spin_box_rows: list[list[QDoubleSpinBox]] = []

        editor = QWidget()
        editor.setStyleSheet(
            "QWidget { background: transparent; color: #c2cad7; }"
            "QDoubleSpinBox { background: #151a24; color: #edf1f7; border: 1px solid #46536a; border-radius: 3px; padding: 2px; }"
        )
        layout = QGridLayout(editor)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(4)
        for row_index, (name, components) in enumerate(
            zip(parameter_names, parameters)
        ):
            label = QLabel(name)
            layout.addWidget(label, row_index, 0)
            row_boxes: list[QDoubleSpinBox] = []
            for column_index, component in enumerate(components):
                spin_box = QDoubleSpinBox()
                spin_box.setRange(-1_000_000.0, 1_000_000.0)
                spin_box.setDecimals(3)
                spin_box.setSingleStep(0.1)
                spin_box.setValue(component)
                spin_box.setFixedWidth(68)
                spin_box.valueChanged.connect(
                    lambda _value, row=row_index: self._row_changed(row)
                )
                layout.addWidget(spin_box, row_index, column_index + 1)
                row_boxes.append(spin_box)
            self.spin_box_rows.append(row_boxes)

        self.proxy = QGraphicsProxyWidget(self)
        self.proxy.setWidget(editor)
        self.proxy.setPos(12.0, NODE_HEADER_HEIGHT + 9.0)
        self.proxy.setZValue(2.0)
        if self.output_port is not None:
            self.output_port.setPos(width, NODE_HEADER_HEIGHT + 20.0)

    def _row_changed(self, row_index: int) -> None:
        """Send one parameter's edited components back to the graph model."""
        components = tuple(box.value() for box in self.spin_box_rows[row_index])
        self.on_change(row_index, components)
```

- [ ] **Step 6: Import `GeneratorNodeItem`/`default_generator_parameters` and `Operation` in `canvas.py`**

Change the `.graphics_items` import block (`canvas.py:41-52`):

```python
from .graphics_items import (
    BaseNodeItem,
    ConnectionItem,
    MeshViewerNodeItem,
    ObjLoaderNodeItem,
    OperationNodeItem,
    OutputNodeItem,
    PortItem,
    ValueNodeItem,
    _bezier_path,
    default_components,
)
```

to:

```python
from .graphics_items import (
    BaseNodeItem,
    ConnectionItem,
    GeneratorNodeItem,
    MeshViewerNodeItem,
    ObjLoaderNodeItem,
    OperationNodeItem,
    OutputNodeItem,
    PortItem,
    ValueNodeItem,
    _bezier_path,
    default_components,
    default_generator_parameters,
)
```

- [ ] **Step 7: Add `add_generator_node`, right after `add_operation_node` (`canvas.py:120-131`)**

```python
    def add_generator_node(
        self,
        operation: Operation,
        position: QPointF | None = None,
        parameters: tuple[tuple[float, ...], ...] | None = None,
    ) -> GeneratorNodeItem:
        """Add a parameter node (Look At, Perspective, ...) to the canvas."""
        node_parameters = (
            parameters
            if parameters is not None
            else default_generator_parameters(operation)
        )
        node_id = self.graph.add_generator(operation, node_parameters)
        node = GeneratorNodeItem(
            node_id,
            operation,
            node_parameters,
            lambda parameter_index, values, generator_node_id=node_id: self._generator_changed(
                generator_node_id, parameter_index, values
            ),
        )
        self.addItem(node)
        node.setPos(position if position is not None else self._next_position())
        self.nodes[node_id] = node
        return node
```

- [ ] **Step 8: Add `_generator_changed`, right after `_value_changed` (`canvas.py:202-205`)**

```python
    def _generator_changed(
        self, node_id: str, parameter_index: int, components: tuple[float, ...]
    ) -> None:
        """Update a graph generator parameter after a spin box changes."""
        self.graph.set_generator_parameter(node_id, parameter_index, components)
        self.update_outputs()
```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `uv run pytest MathNodeEditor/tests/test_node_editor.py -k generator -v`
Expected: PASS (3 passed)

- [ ] **Step 10: Commit**

```bash
git add MathNodeEditor/graphics_items.py MathNodeEditor/canvas.py MathNodeEditor/tests/test_node_editor.py
git commit -m "feat: add GeneratorNodeItem widget and canvas wiring"
```

---

### Task 4: Save/load JSON support for generator nodes

**Files:**
- Modify: `MathNodeEditor/canvas.py:312-359` (`to_dict`), `canvas.py:360-414` (`from_dict`)
- Test: `MathNodeEditor/tests/test_node_editor.py`

**Interfaces:**
- Consumes: `GeneratorNodeItem`, `MathNodeScene.add_generator_node` from Task 3.
- Produces: `to_dict()` entries of shape `{"kind": "generator", "operation": str, "parameters": list[list[float]], "id": str, "x": float, "y": float}`; `from_dict()` accepting that shape.

- [ ] **Step 1: Write the failing test**

Add to `MathNodeEditor/tests/test_node_editor.py`, after `test_to_dict_and_from_dict_round_trip_the_example_graph`:

```python
def test_generator_node_round_trips_through_to_dict_and_from_dict(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    look_at_node = window.canvas.add_generator_node(
        node_editor.Operation.LOOK_AT,
        parameters=((0.0, 3.0, 9.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )
    output_node = window.canvas.add_output_node()
    assert look_at_node.output_port is not None
    window.canvas.connect_ports(look_at_node.output_port, output_node.input_ports[0])
    application.processEvents()
    before = window.canvas.output_texts()

    data = window.canvas.to_dict()
    window.canvas.from_dict(data)
    application.processEvents()

    after_node = next(
        node
        for node in window.canvas.nodes.values()
        if isinstance(node, node_editor.GeneratorNodeItem)
    )
    values = [box.value() for row in after_node.spin_box_rows for box in row]
    assert values == pytest.approx([0.0, 3.0, 9.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert window.canvas.output_texts() == before
    window.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest MathNodeEditor/tests/test_node_editor.py -k round_trips_through_to_dict -v`
Expected: FAIL — the round trip drops the node (`from_dict` raises `GraphError: Unknown node kind 'generator'`)

- [ ] **Step 3: Add a `to_dict` branch, right after the `OperationNodeItem` branch (`canvas.py`, inside the `to_dict` node loop)**

```python
            elif isinstance(node, GeneratorNodeItem):
                entry["kind"] = "generator"
                entry["operation"] = node.operation.name
                entry["parameters"] = [
                    [box.value() for box in row] for row in node.spin_box_rows
                ]
```

- [ ] **Step 4: Add a `from_dict` branch, right after the `"operation"` branch (`canvas.py`, inside the `from_dict` node loop)**

```python
            elif kind == "generator":
                node = self.add_generator_node(
                    Operation[entry["operation"]],
                    position,
                    tuple(tuple(p) for p in entry["parameters"]),
                )
                id_map[entry["id"]] = node.node_id
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest MathNodeEditor/tests/test_node_editor.py -k round_trips_through_to_dict -v`
Expected: PASS

- [ ] **Step 6: Run the full node_editor test file to check for regressions**

Run: `uv run pytest MathNodeEditor/tests/test_node_editor.py -v`
Expected: PASS (pre-existing failures from Task 5/6's not-yet-updated tests are expected at this point — see Task 5)

- [ ] **Step 7: Commit**

```bash
git add MathNodeEditor/canvas.py MathNodeEditor/tests/test_node_editor.py
git commit -m "feat: save/load generator nodes as JSON"
```

---

### Task 5: Palette routing and `node_editor` re-export

**Files:**
- Modify: `MathNodeEditor/palette.py:1-97` (import, `_operation_catalogue_entries`)
- Modify: `MathNodeEditor/node_editor.py:19-93` (import `GeneratorNodeItem`, add to `__all__`)
- Modify: `MathNodeEditor/tests/test_node_editor.py` (palette/menu assertions)

**Interfaces:**
- Consumes: `GENERATOR_OPERATIONS` from `math_graph` (Task 1); `MathNodeScene.add_generator_node` (Task 3); `GeneratorNodeItem` (Task 3).

- [ ] **Step 1: Import `GENERATOR_OPERATIONS` in `palette.py`**

Change:

```python
from .math_graph import GraphError, MathType, Operation
```

to:

```python
from .math_graph import GENERATOR_OPERATIONS, GraphError, MathType, Operation
```

- [ ] **Step 2: Route generator operations in `_operation_catalogue_entries` (`palette.py:85-97`)**

Replace:

```python
def _operation_catalogue_entries(
    operations: tuple[Operation, ...],
) -> list[CatalogueEntry]:
    """Return one catalogue entry per operation in the given group."""
    return [
        (
            operation.value,
            lambda canvas, position, value=operation: canvas.add_operation_node(
                value, position
            ),
        )
        for operation in operations
    ]
```

with:

```python
def _operation_catalogue_entries(
    operations: tuple[Operation, ...],
) -> list[CatalogueEntry]:
    """Return one catalogue entry per operation in the given group.

    Operations in GENERATOR_OPERATIONS (Look At, Perspective, Transform,
    ...) route through add_generator_node, since they take typed-in
    Float/Vec3 parameters rather than wired inputs; everything else keeps
    add_operation_node's wired sockets.
    """
    entries: list[CatalogueEntry] = []
    for operation in operations:
        if operation in GENERATOR_OPERATIONS:
            entries.append(
                (
                    operation.value,
                    lambda canvas, position, value=operation: canvas.add_generator_node(
                        value, position
                    ),
                )
            )
        else:
            entries.append(
                (
                    operation.value,
                    lambda canvas, position, value=operation: canvas.add_operation_node(
                        value, position
                    ),
                )
            )
    return entries
```

- [ ] **Step 3: Re-export `GeneratorNodeItem` from `node_editor.py`**

In the `.graphics_items` import block, change:

```python
from .graphics_items import (
    GENERIC_PORT_COLOUR,
    NODE_HEADER_HEIGHT,
    PORT_RADIUS,
    TYPE_COLOURS,
    BaseNodeItem,
    ConnectionItem,
    MeshViewerNodeItem,
    ObjLoaderNodeItem,
    OperationNodeItem,
    OutputNodeItem,
    PortItem,
    ValueNodeItem,
    default_components,
    node_title_font,
)
```

to:

```python
from .graphics_items import (
    GENERIC_PORT_COLOUR,
    NODE_HEADER_HEIGHT,
    PORT_RADIUS,
    TYPE_COLOURS,
    BaseNodeItem,
    ConnectionItem,
    GeneratorNodeItem,
    MeshViewerNodeItem,
    ObjLoaderNodeItem,
    OperationNodeItem,
    OutputNodeItem,
    PortItem,
    ValueNodeItem,
    default_components,
    node_title_font,
)
```

and in `__all__`, insert `"GeneratorNodeItem",` between `"GraphError",` and `"MathGraph",`.

- [ ] **Step 4: Update palette/menu node-type assertions in `test_node_editor.py`**

In `test_palette_buttons_add_the_requested_node_type`'s parametrize list, change:

```python
        ("Matrix Multiply", "OperationNodeItem"),
        ("Look At", "OperationNodeItem"),
        ("Perspective", "OperationNodeItem"),
        ("Output", "OutputNodeItem"),
```

to:

```python
        ("Matrix Multiply", "OperationNodeItem"),
        ("Look At", "GeneratorNodeItem"),
        ("Perspective", "GeneratorNodeItem"),
        ("Output", "OutputNodeItem"),
```

Replace `test_operation_nodes_display_semantic_input_names` with two tests — one for operations that stay wired, one for the new generator nodes:

```python
@pytest.mark.parametrize(
    ("operation_name", "expected_names"),
    [
        ("QUATERNION_SLERP", ("Start", "End", "T")),
    ],
)
def test_operation_nodes_display_semantic_input_names(
    application: QApplication,
    operation_name: str,
    expected_names: tuple[str, ...],
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)

    operation_node = window.canvas.add_operation_node(
        node_editor.Operation[operation_name]
    )

    assert operation_node.input_names == expected_names
    window.close()


@pytest.mark.parametrize(
    ("operation_name", "expected_names"),
    [
        ("LOOK_AT", ("Eye", "Target", "Up")),
        ("PERSPECTIVE", ("FOV", "Aspect", "Near", "Far")),
    ],
)
def test_generator_nodes_display_semantic_parameter_names(
    application: QApplication,
    operation_name: str,
    expected_names: tuple[str, ...],
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)

    generator_node = window.canvas.add_generator_node(
        node_editor.Operation[operation_name]
    )

    assert generator_node.parameter_names == expected_names
    window.close()
```

In `test_long_quaternion_operation_title_fits_inside_node`, change:

```python
    operation_node = window.canvas.add_operation_node(
        node_editor.Operation.QUATERNION_FROM_AXIS_ANGLE
    )
```

to:

```python
    operation_node = window.canvas.add_generator_node(
        node_editor.Operation.QUATERNION_FROM_AXIS_ANGLE
    )
```

In `test_enter_creates_first_filtered_node`, change:

```python
    added_node = next(iter(window.canvas.nodes.values()))
    assert isinstance(added_node, node_editor.OperationNodeItem)
    assert added_node.operation is node_editor.Operation.PERSPECTIVE
```

to:

```python
    added_node = next(iter(window.canvas.nodes.values()))
    assert isinstance(added_node, node_editor.GeneratorNodeItem)
    assert added_node.operation is node_editor.Operation.PERSPECTIVE
```

- [ ] **Step 5: Run the full node_editor test file**

Run: `uv run pytest MathNodeEditor/tests/test_node_editor.py -v`
Expected: PASS except `test_mvp_example_loads_and_evaluates_a_single_matrix` (fixed in Task 6, which also rewrites the example JSON it loads)

- [ ] **Step 6: Commit**

```bash
git add MathNodeEditor/palette.py MathNodeEditor/node_editor.py MathNodeEditor/tests/test_node_editor.py
git commit -m "feat: route generator operations through the new node type in the palette"
```

---

### Task 6: Rewrite the example graphs

**Files:**
- Modify: `MathNodeEditor/examples/mvp_demo.json`, `MathNodeEditor/examples/mvp_mesh_demo.json`, `MathNodeEditor/examples/mesh_pipeline_demo.json`
- Modify: `MathNodeEditor/tests/test_node_editor.py` (`test_mvp_example_loads_and_evaluates_a_single_matrix`)

**Interfaces:**
- Consumes: `"kind": "generator"` JSON support from Task 4.

- [ ] **Step 1: Rewrite `mvp_demo.json` — fold its Position/Rotation/Scale, Eye/Target/Up and FOV/Aspect/Near/Far value nodes into the three generator nodes they used to feed**

Run from the repo root (`/Volumes/teaching/Code/PyNGLDemos/.worktrees/mathnode-generators`):

```bash
uv run python - <<'PYEOF'
import json
from pathlib import Path

path = Path("MathNodeEditor/examples/mvp_demo.json")
data = json.loads(path.read_text())
nodes_by_id = {n["id"]: n for n in data["nodes"]}

def take(*ids):
    return [nodes_by_id[i]["components"] for i in ids]

transform_params = take("node-1", "node-2", "node-3")
look_at_params = take("node-5", "node-6", "node-7")
perspective_params = take("node-9", "node-10", "node-11", "node-12")

drop_ids = {
    "node-1", "node-2", "node-3",
    "node-5", "node-6", "node-7",
    "node-9", "node-10", "node-11", "node-12",
}
data["nodes"] = [n for n in data["nodes"] if n["id"] not in drop_ids]

for node_id, operation, parameters in (
    ("node-4", "TRANSFORM", transform_params),
    ("node-8", "LOOK_AT", look_at_params),
    ("node-13", "PERSPECTIVE", perspective_params),
):
    node = nodes_by_id[node_id]
    node["kind"] = "generator"
    node["operation"] = operation
    node["parameters"] = parameters

data["connections"] = [
    c for c in data["connections"] if c["target"] not in {"node-4", "node-8", "node-13"}
]

path.write_text(json.dumps(data, indent=2) + "\n")
PYEOF
```

The three target nodes (`node-4`, `node-8`, `node-13`) are currently `"kind": "operation"` entries with no `"math_type"` key, so the loop only needs to set `kind`/`operation`/`parameters` — nothing to delete.

- [ ] **Step 2: Rewrite `mvp_mesh_demo.json` — fold its Position/Rotation/Scale value nodes into the Transform node**

```bash
uv run python - <<'PYEOF'
import json
from pathlib import Path

path = Path("MathNodeEditor/examples/mvp_mesh_demo.json")
data = json.loads(path.read_text())
nodes_by_id = {n["id"]: n for n in data["nodes"]}

position = nodes_by_id["node-5"]["components"]
rotation = nodes_by_id["node-6"]["components"]
scale = nodes_by_id["node-7"]["components"]

data["nodes"] = [
    n for n in data["nodes"] if n["id"] not in {"node-5", "node-6", "node-7"}
]
transform_node = nodes_by_id["node-8"]
transform_node["kind"] = "generator"
transform_node["operation"] = "TRANSFORM"
transform_node["parameters"] = [position, rotation, scale]

data["connections"] = [c for c in data["connections"] if c["target"] != "node-8"]

path.write_text(json.dumps(data, indent=2) + "\n")
PYEOF
```

- [ ] **Step 3: Rewrite `mesh_pipeline_demo.json` — fold its Float angle value node into the Mat4 Rotate Y node**

```bash
uv run python - <<'PYEOF'
import json
from pathlib import Path

path = Path("MathNodeEditor/examples/mesh_pipeline_demo.json")
data = json.loads(path.read_text())
nodes_by_id = {n["id"]: n for n in data["nodes"]}

angle = nodes_by_id["node-5"]["components"]

data["nodes"] = [n for n in data["nodes"] if n["id"] != "node-5"]
rotate_node = nodes_by_id["node-6"]
rotate_node["kind"] = "generator"
rotate_node["operation"] = "MAT4_ROTATE_Y"
rotate_node["parameters"] = [angle]

data["connections"] = [c for c in data["connections"] if c["target"] != "node-6"]

path.write_text(json.dumps(data, indent=2) + "\n")
PYEOF
```

- [ ] **Step 4: Update `test_mvp_example_loads_and_evaluates_a_single_matrix` in `test_node_editor.py`**

Change:

```python
    transform_node = next(
        node
        for node in window.canvas.nodes.values()
        if isinstance(node, node_editor.OperationNodeItem)
        and node.operation is node_editor.Operation.TRANSFORM
    )
    assert transform_node.input_names == ("Position", "Rotation", "Scale")
```

to:

```python
    transform_node = next(
        node
        for node in window.canvas.nodes.values()
        if isinstance(node, node_editor.GeneratorNodeItem)
        and node.operation is node_editor.Operation.TRANSFORM
    )
    assert transform_node.parameter_names == ("Position", "Rotation", "Scale")
```

- [ ] **Step 5: Run the whole test suite**

Run: `uv run pytest MathNodeEditor -v`
Expected: PASS, all tests green (this now covers `test_mesh_pipeline_example_loads_without_errors` and `test_mvp_mesh_example_applies_the_model_transform_to_a_displayed_mesh`, which need no code changes but must still pass against the rewritten JSON).

- [ ] **Step 6: Diff-check the example files still parse and evaluate correctly**

Run: `uv run python -c "import json; [json.loads(open(f'MathNodeEditor/examples/{n}').read()) for n in ('mvp_demo.json','mvp_mesh_demo.json','mesh_pipeline_demo.json')]; print('ok')"`
Expected: `ok`

- [ ] **Step 7: Commit**

```bash
git add MathNodeEditor/examples/mvp_demo.json MathNodeEditor/examples/mvp_mesh_demo.json MathNodeEditor/examples/mesh_pipeline_demo.json MathNodeEditor/tests/test_node_editor.py
git commit -m "docs: fold value nodes into generator-node parameters in the example graphs"
```

---

### Task 7: README and screenshot refresh

**Files:**
- Modify: `MathNodeEditor/README.md`
- Modify (binary): `MathNodeEditor/MathNodeEditor.png`

**Interfaces:**
- Consumes: the rewritten `examples/mvp_demo.json` (Task 6).

- [ ] **Step 1: Update `MathNodeEditor/README.md`**

Load the `jon-writing-style` skill before writing any prose (mandatory per the user's global CLAUDE.md for all documentation).

Edit the paragraph starting "The Mat4 nodes cover..." to describe the new spin-box behaviour. Replace:

```
The Mat4 nodes cover translate, scale, rotation about each axis, `look_at`, perspective, orthographic and frustum projection, plus a `Transform` node wrapping PyNGL's `Transform` class (`Position`/`Rotation`/`Scale` `Vec3` inputs combined into one Model matrix, rotation in degrees, xyz order). The quaternion nodes cover axis-angle creation, Hamilton product, vector rotation, conversion to and from `Mat4`, slerp, conjugate and inverse. The original add, subtract, component multiply, matrix multiply, dot, cross, normalise and transpose nodes are still present.
```

with:

```
The Mat4 nodes cover translate, scale, rotation about each axis, `look_at`, perspective, orthographic and frustum projection, plus a `Transform` node wrapping PyNGL's `Transform` class (`Position`/`Rotation`/`Scale`, rotation in degrees, xyz order). These, along with the quaternion axis-angle node, take their parameters from spin boxes built into the node itself rather than wired-in Value nodes — there's nothing to connect, just numbers to type. The remaining quaternion nodes (Hamilton product, vector rotation, conversion to and from `Mat4`, slerp, conjugate, inverse) still take wired inputs, since they combine other quaternions or matrices rather than raw numbers. The original add, subtract, component multiply, matrix multiply, dot, cross, normalise and transpose nodes are unchanged.
```

Update the `mvp_demo.json` description. Replace:

```
`examples/mvp_demo.json` builds a typical Model/View/Projection matrix: a `Transform` node for Model, `Look At` for View, `Perspective` for Projection, combined with two `Matrix Multiply` nodes as `MVP = Projection @ (View @ Model)` — the same order used throughout this repo's OpenGL demos (e.g. `ObjViewer.py`'s `mvp = self.project @ self.view @ self.mouse_global_tx`) — and shown on an `Output` node.
```

with:

```
`examples/mvp_demo.json` builds a typical Model/View/Projection matrix: a `Transform` node for Model, `Look At` for View, `Perspective` for Projection — each carrying its own numbers directly, no Value nodes required — combined with two `Matrix Multiply` nodes as `MVP = Projection @ (View @ Model)` — the same order used throughout this repo's OpenGL demos (e.g. `ObjViewer.py`'s `mvp = self.project @ self.view @ self.mouse_global_tx`) — and shown on an `Output` node.
```

Update the `mvp_mesh_demo.json` description. Replace:

```
`examples/mvp_mesh_demo.json` puts the Model half of that to work on a displayed mesh instead of a plain `Output` node: `Obj Loader` (`examples/cube.obj`) feeds a `Transform` node's `Position`/`Rotation`/`Scale` through `Transform Vertices` and the manual normal-matrix pipeline, landing on a `Mesh Viewer`. There's no `Look At`/`Perspective` in this one — View and Projection are the `Mesh Viewer`'s own arcball camera, which is what you're driving when you drag inside its preview or the `Pop Out` window.
```

with:

```
`examples/mvp_mesh_demo.json` puts the Model half of that to work on a displayed mesh instead of a plain `Output` node: `Obj Loader` (`examples/cube.obj`) feeds a `Transform` node's Position/Rotation/Scale spin boxes through `Transform Vertices` and the manual normal-matrix pipeline, landing on a `Mesh Viewer`. There's no `Look At`/`Perspective` in this one — View and Projection are the `Mesh Viewer`'s own arcball camera, which is what you're driving when you drag inside its preview or the `Pop Out` window.
```

- [ ] **Step 2: Regenerate the screenshot**

Run from the repo root:

```bash
cat > capture_screenshot.py <<'PYEOF'
from pathlib import Path

from PySide6.QtWidgets import QApplication

from MathNodeEditor.node_editor import MathNodeWindow

app = QApplication.instance() or QApplication([])
window = MathNodeWindow(load_example=False)
window.canvas.load_from_file(Path("MathNodeEditor/examples/mvp_demo.json"))
window.resize(1280, 760)
window.show()
app.processEvents()
window.view.frame_all()
app.processEvents()
window.grab().save("MathNodeEditor/MathNodeEditor.png")
window.close()
PYEOF
QT_QPA_PLATFORM=offscreen uv run python capture_screenshot.py
rm capture_screenshot.py
```

- [ ] **Step 3: Visually sanity-check the screenshot**

Open `MathNodeEditor/MathNodeEditor.png` and confirm it shows the Look At / Perspective / Transform nodes with their spin boxes and no stray Value nodes feeding them.

- [ ] **Step 4: Commit**

```bash
git add MathNodeEditor/README.md MathNodeEditor/MathNodeEditor.png
git commit -m "docs: describe generator nodes and refresh the MathNodeEditor screenshot"
```

---

### Task 8: Full verification, session summary, and final commit

**Files:**
- Create: `docs/agent-sessions/2026-08-13-session.md`

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest MathNodeEditor -v`
Expected: PASS, 0 failures.

- [ ] **Step 2: Run the linter and formatter**

Run: `uv run ruff check MathNodeEditor` then `uv run ruff format --check MathNodeEditor`
Expected: both clean. If `ruff format --check` reports files needing formatting, run `uv run ruff format MathNodeEditor` and re-run the test suite.

- [ ] **Step 3: Launch the app once by hand to eyeball it**

Run: `uv run MathNodeEditor/main.py` — add a Look At node from the palette (or press Tab and type "look at"), confirm it shows three labelled Vec3 spin-box rows and no input sockets, and that editing a value updates anything wired to its output. Close the window when done.

- [ ] **Step 4: Write the session summary**

Create `docs/agent-sessions/2026-08-13-session.md`:

```markdown
# 2026-08-13 session: MathNodeEditor generator nodes

## Goal

Give the eleven MathNodeEditor operations that only take Float/Vec3
parameters (Look At, Perspective, Orthographic, Frustum, Mat4 Translate,
Mat4 Scale, Mat4 Rotate X/Y/Z, Transform, Quaternion from Axis Angle) their
own inline spin boxes instead of wired input sockets, and bring every
example graph, test and doc in line with the change.

## Files changed

- `MathNodeEditor/math_graph.py` — `GeneratorNode` model, `GENERATOR_OPERATIONS`/`OPERATION_PARAMETER_TYPES`/`GENERATOR_OUTPUT_TYPE` tables, `add_generator`/`set_generator_parameter`, `add_operation` guard
- `MathNodeEditor/graphics_items.py` — `GeneratorNodeItem`, `default_generator_parameters`
- `MathNodeEditor/canvas.py` — `add_generator_node`/`_generator_changed`, JSON save/load support
- `MathNodeEditor/palette.py` — routes generator operations to the new node type
- `MathNodeEditor/node_editor.py` — re-exports `GeneratorNodeItem`
- `MathNodeEditor/examples/mvp_demo.json`, `mvp_mesh_demo.json`, `mesh_pipeline_demo.json` — rewritten to embed parameters directly
- `MathNodeEditor/README.md`, `MathNodeEditor/MathNodeEditor.png` — docs/screenshot refresh
- `MathNodeEditor/tests/test_math_graph.py`, `tests/test_node_editor.py` — new + migrated tests

## Commands run

- `uv run pytest MathNodeEditor -v`
- `uv run ruff check MathNodeEditor`
- `uv run ruff format MathNodeEditor`
- `uv run MathNodeEditor/main.py` (manual smoke test)
```

- [ ] **Step 5: Final commit**

```bash
git add docs/agent-sessions/2026-08-13-session.md
git commit -m "docs: add session summary for MathNodeEditor generator nodes work"
```

- [ ] **Step 6: Report back**

Summarize to the user: branch `agent/mathnode-generators` in `.worktrees/mathnode-generators` is ready, all tests/lint pass, and ask whether to open a PR / merge into `Version1.0`, per the project's "never commit directly to a protected branch" rule.

---

## Self-Review Notes

- **Spec coverage:** every section of the design spec (data model, graphics, canvas/palette, examples, defaults, tests, docs) maps to a task above; the "no backward-compat loader" decision is honoured by not adding one anywhere.
- **Placeholder scan:** no TBD/TODO; every step has literal code, JSON-transform scripts, or exact run commands.
- **Type consistency:** `GeneratorNodeItem.spin_box_rows`/`.parameter_names`/`.operation` (Task 3) are the exact names used by Task 4's `to_dict`, Task 5's test updates, and Task 6's test updates. `MathGraph.add_generator`/`set_generator_parameter` (Task 1) are the exact names used by `canvas.add_generator_node`/`_generator_changed` (Task 3). `default_generator_parameters` (Task 3, graphics_items.py) is the exact name imported and called by `canvas.add_generator_node`.
- **Caught during drafting:** Task 6 Step 1's first script draft had a stray `del node["math_type"]` that would `KeyError` on the target nodes (they never had that key) — removed before finalizing the step.
