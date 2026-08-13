"""Calculation graph for the PyNGL maths node editor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from ncca.ngl import (
    Mat2,
    Mat3,
    Mat4,
    Quaternion,
    Vec2,
    Vec3,
    Vec4,
    frustum,
    look_at,
    ortho,
    perspective,
)


class MathType(Enum):
    """Maths value types accepted by value nodes."""

    FLOAT = "Float"
    VEC2 = "Vec2"
    VEC3 = "Vec3"
    VEC4 = "Vec4"
    MAT2 = "Mat2"
    MAT3 = "Mat3"
    MAT4 = "Mat4"
    QUATERNION = "Quaternion"


class Operation(Enum):
    """Operations which can be added to a graph."""

    ADD = "Add"
    SUBTRACT = "Subtract"
    MULTIPLY = "Multiply"
    MATRIX_MULTIPLY = "Matrix Multiply"
    DOT = "Dot Product"
    CROSS = "Cross Product"
    NORMALISE = "Normalise"
    TRANSPOSE = "Transpose"
    LOOK_AT = "Look At"
    PERSPECTIVE = "Perspective"
    ORTHO = "Orthographic"
    FRUSTUM = "Frustum"
    MAT4_TRANSLATE = "Mat4 Translate"
    MAT4_SCALE = "Mat4 Scale"
    MAT4_ROTATE_X = "Mat4 Rotate X"
    MAT4_ROTATE_Y = "Mat4 Rotate Y"
    MAT4_ROTATE_Z = "Mat4 Rotate Z"
    QUATERNION_FROM_AXIS_ANGLE = "Quaternion from Axis Angle"
    QUATERNION_PRODUCT = "Quaternion Product"
    QUATERNION_ROTATE_VECTOR = "Quaternion Rotate Vector"
    QUATERNION_TO_MAT4 = "Quaternion to Mat4"
    MAT4_TO_QUATERNION = "Mat4 to Quaternion"
    QUATERNION_SLERP = "Quaternion Slerp"
    QUATERNION_CONJUGATE = "Quaternion Conjugate"
    QUATERNION_INVERSE = "Quaternion Inverse"


OPERATION_INPUT_NAMES: dict[Operation, tuple[str, ...]] = {
    Operation.ADD: ("A", "B"),
    Operation.SUBTRACT: ("A", "B"),
    Operation.MULTIPLY: ("A", "B"),
    Operation.MATRIX_MULTIPLY: ("A", "B"),
    Operation.DOT: ("A", "B"),
    Operation.CROSS: ("A", "B"),
    Operation.NORMALISE: ("Value",),
    Operation.TRANSPOSE: ("Matrix",),
    Operation.LOOK_AT: ("Eye", "Target", "Up"),
    Operation.PERSPECTIVE: ("FOV", "Aspect", "Near", "Far"),
    Operation.ORTHO: ("Left", "Right", "Bottom", "Top", "Near", "Far"),
    Operation.FRUSTUM: ("Left", "Right", "Bottom", "Top", "Near", "Far"),
    Operation.MAT4_TRANSLATE: ("X", "Y", "Z"),
    Operation.MAT4_SCALE: ("X", "Y", "Z"),
    Operation.MAT4_ROTATE_X: ("Angle",),
    Operation.MAT4_ROTATE_Y: ("Angle",),
    Operation.MAT4_ROTATE_Z: ("Angle",),
    Operation.QUATERNION_FROM_AXIS_ANGLE: ("Axis", "Angle"),
    Operation.QUATERNION_PRODUCT: ("A", "B"),
    Operation.QUATERNION_ROTATE_VECTOR: ("Quaternion", "Vector"),
    Operation.QUATERNION_TO_MAT4: ("Quaternion",),
    Operation.MAT4_TO_QUATERNION: ("Matrix",),
    Operation.QUATERNION_SLERP: ("Start", "End", "T"),
    Operation.QUATERNION_CONJUGATE: ("Quaternion",),
    Operation.QUATERNION_INVERSE: ("Quaternion",),
}

OPERATION_ARITY: dict[Operation, int] = {
    operation: len(input_names)
    for operation, input_names in OPERATION_INPUT_NAMES.items()
}


MathValue: TypeAlias = Vec2 | Vec3 | Vec4 | Mat2 | Mat3 | Mat4 | Quaternion | float

VALUE_CLASSES: dict[MathType, Callable[..., MathValue]] = {
    MathType.FLOAT: float,
    MathType.VEC2: Vec2,
    MathType.VEC3: Vec3,
    MathType.VEC4: Vec4,
    MathType.MAT2: Mat2,
    MathType.MAT3: Mat3,
    MathType.MAT4: Mat4,
    MathType.QUATERNION: Quaternion,
}

TYPE_SHAPES: dict[MathType, tuple[int, int]] = {
    MathType.FLOAT: (1, 1),
    MathType.VEC2: (1, 2),
    MathType.VEC3: (1, 3),
    MathType.VEC4: (1, 4),
    MathType.MAT2: (2, 2),
    MathType.MAT3: (3, 3),
    MathType.MAT4: (4, 4),
    MathType.QUATERNION: (1, 4),
}


class GraphError(ValueError):
    """An error which can be shown directly on an output node."""


def _validate_components(math_type: MathType, components: tuple[float, ...]) -> None:
    """Check that a value has the correct number of components."""
    rows, columns = TYPE_SHAPES[math_type]
    expected_count = rows * columns
    if len(components) != expected_count:
        raise GraphError(f"{math_type.value} needs {expected_count} components")


def _format_number(value: float) -> str:
    """Format a component without unnecessary trailing zeroes."""
    return f"{float(value):.5g}"


def format_value(value: MathValue) -> str:
    """Format a PyNGL maths value for an output node."""
    if isinstance(value, float):
        return _format_number(value)

    value_name = type(value).__name__
    data = value.to_numpy()
    if data.ndim == 1:
        components = ", ".join(_format_number(component) for component in data)
        return f"{value_name}({components})"

    rows = [
        "[" + "  ".join(_format_number(component) for component in row) + "]"
        for row in data
    ]
    return f"{value_name}\n" + "\n".join(rows)


def _normalise(value: MathValue) -> MathValue:
    """Return a unit-length copy of a vector or quaternion."""
    return value.normalized()


def _transpose(value: MathValue) -> MathValue:
    """Return the transpose of a matrix."""
    return value.transposed()


def _quaternion_to_mat4(value: MathValue) -> MathValue:
    """Convert a Quaternion input to its Mat4 rotation."""
    if not isinstance(value, Quaternion):
        raise TypeError("Quaternion to Mat4 needs a Quaternion input")
    return value.to_mat4()


def _mat4_to_quaternion(value: MathValue) -> MathValue:
    """Convert a Mat4 input to the equivalent Quaternion."""
    if not isinstance(value, Mat4):
        raise TypeError("Mat4 to Quaternion needs a Mat4 input")
    return Quaternion.from_mat4(value)


def _quaternion_conjugate(value: MathValue) -> MathValue:
    """Return the conjugate of a Quaternion input."""
    if not isinstance(value, Quaternion):
        raise TypeError("Quaternion Conjugate needs a Quaternion input")
    return value.conjugate()


def _quaternion_inverse(value: MathValue) -> MathValue:
    """Return the inverse of a Quaternion input."""
    if not isinstance(value, Quaternion):
        raise TypeError("Quaternion Inverse needs a Quaternion input")
    return value.inverse()


def _quaternion_slerp(start: MathValue, end: MathValue, blend: MathValue) -> MathValue:
    """Spherically interpolate between two Quaternion inputs."""
    if (
        not isinstance(start, Quaternion)
        or not isinstance(end, Quaternion)
        or not isinstance(blend, float)
    ):
        raise TypeError(
            "Quaternion Slerp needs Quaternion, Quaternion and Float inputs"
        )
    return start.slerp(end, blend)


def _look_at(eye: MathValue, target: MathValue, up: MathValue) -> MathValue:
    """Build a view Mat4 from three Vec3 inputs."""
    if not all(isinstance(value, Vec3) for value in (eye, target, up)):
        raise TypeError("Look At needs three Vec3 inputs")
    return look_at(eye, target, up)


def _perspective(*inputs: MathValue) -> MathValue:
    """Build a perspective projection Mat4 from four Float inputs."""
    if not all(isinstance(value, float) for value in inputs):
        raise TypeError("Perspective needs four Float inputs")
    return perspective(*inputs)


def _ortho(*inputs: MathValue) -> MathValue:
    """Build an orthographic projection Mat4 from six Float inputs."""
    if not all(isinstance(value, float) for value in inputs):
        raise TypeError("Orthographic needs six Float inputs")
    return ortho(*inputs)


def _frustum(*inputs: MathValue) -> MathValue:
    """Build a frustum projection Mat4 from six Float inputs."""
    if not all(isinstance(value, float) for value in inputs):
        raise TypeError("Frustum needs six Float inputs")
    return frustum(*inputs)


def _mat4_translate(*inputs: MathValue) -> MathValue:
    """Build a translation Mat4 from three Float inputs."""
    if not all(isinstance(value, float) for value in inputs):
        raise TypeError("Mat4 Translate needs three Float inputs")
    return Mat4.translate(*inputs)


def _mat4_scale(*inputs: MathValue) -> MathValue:
    """Build a scale Mat4 from three Float inputs."""
    if not all(isinstance(value, float) for value in inputs):
        raise TypeError("Mat4 Scale needs three Float inputs")
    return Mat4.scale(*inputs)


def _mat4_rotate_x(angle: MathValue) -> MathValue:
    """Build a Mat4 rotation about X from a Float angle."""
    return Mat4.rotate_x(angle)


def _mat4_rotate_y(angle: MathValue) -> MathValue:
    """Build a Mat4 rotation about Y from a Float angle."""
    return Mat4.rotate_y(angle)


def _mat4_rotate_z(angle: MathValue) -> MathValue:
    """Build a Mat4 rotation about Z from a Float angle."""
    return Mat4.rotate_z(angle)


def _quaternion_from_axis_angle(axis: MathValue, angle: MathValue) -> MathValue:
    """Build a Quaternion from a Vec3 axis and a Float angle."""
    if not isinstance(axis, Vec3) or not isinstance(angle, float):
        raise TypeError("Quaternion from Axis Angle needs Vec3 and Float inputs")
    return Quaternion.from_axis_angle(axis, angle)


def _add(left: MathValue, right: MathValue) -> MathValue:
    """Add two matching PyNGL values."""
    return left + right


def _subtract(left: MathValue, right: MathValue) -> MathValue:
    """Subtract one PyNGL value from another."""
    return left - right


def _matrix_multiply(left: MathValue, right: MathValue) -> MathValue:
    """Apply the PyNGL ``@`` product to two inputs."""
    return left @ right


def _quaternion_product(left: MathValue, right: MathValue) -> MathValue:
    """Return the Hamilton product of two Quaternion inputs."""
    if not isinstance(left, Quaternion) or not isinstance(right, Quaternion):
        raise TypeError("Quaternion Product needs two Quaternion inputs")
    return left @ right


def _quaternion_rotate_vector(left: MathValue, right: MathValue) -> MathValue:
    """Rotate a Vec3 input by a Quaternion input."""
    if not isinstance(left, Quaternion) or not isinstance(right, Vec3):
        raise TypeError("Quaternion Rotate Vector needs Quaternion and Vec3 inputs")
    return left * right


def _dot(left: MathValue, right: MathValue) -> MathValue:
    """Return the scalar dot product of two vector inputs."""
    return float(left.dot(right))


def _cross(left: MathValue, right: MathValue) -> MathValue:
    """Return the cross product of two vector inputs."""
    return left.cross(right)


def _multiply(left: MathValue, right: MathValue) -> MathValue:
    """Multiply two matching inputs component-wise.

    PyNGL reserves ``*`` for scalar multiplication, so this rebuilds the
    result from zipped components rather than using the operator.
    """
    if type(left) is not type(right):
        raise ValueError("component multiply needs matching input types")
    if isinstance(left, float):
        return left * right
    components = tuple(a * b for a, b in zip(left.to_list(), right.to_list()))
    return type(left)(*components)


_OPERATION_HANDLERS: dict[Operation, Callable[..., MathValue]] = {
    Operation.ADD: _add,
    Operation.SUBTRACT: _subtract,
    Operation.MULTIPLY: _multiply,
    Operation.MATRIX_MULTIPLY: _matrix_multiply,
    Operation.DOT: _dot,
    Operation.CROSS: _cross,
    Operation.NORMALISE: _normalise,
    Operation.TRANSPOSE: _transpose,
    Operation.LOOK_AT: _look_at,
    Operation.PERSPECTIVE: _perspective,
    Operation.ORTHO: _ortho,
    Operation.FRUSTUM: _frustum,
    Operation.MAT4_TRANSLATE: _mat4_translate,
    Operation.MAT4_SCALE: _mat4_scale,
    Operation.MAT4_ROTATE_X: _mat4_rotate_x,
    Operation.MAT4_ROTATE_Y: _mat4_rotate_y,
    Operation.MAT4_ROTATE_Z: _mat4_rotate_z,
    Operation.QUATERNION_FROM_AXIS_ANGLE: _quaternion_from_axis_angle,
    Operation.QUATERNION_PRODUCT: _quaternion_product,
    Operation.QUATERNION_ROTATE_VECTOR: _quaternion_rotate_vector,
    Operation.QUATERNION_TO_MAT4: _quaternion_to_mat4,
    Operation.MAT4_TO_QUATERNION: _mat4_to_quaternion,
    Operation.QUATERNION_SLERP: _quaternion_slerp,
    Operation.QUATERNION_CONJUGATE: _quaternion_conjugate,
    Operation.QUATERNION_INVERSE: _quaternion_inverse,
}


def apply_operation(
    operation: Operation,
    *inputs: MathValue,
) -> MathValue:
    """Apply an operation and translate PyNGL errors for the graph UI."""
    try:
        return _OPERATION_HANDLERS[operation](*inputs)
    except GraphError:
        raise
    except (AttributeError, TypeError, ValueError, ZeroDivisionError) as error:
        raise GraphError(f"{operation.value} failed: {error}") from error


@dataclass(slots=True)
class ValueNode:
    """A typed source value in the graph."""

    math_type: MathType
    components: tuple[float, ...]


@dataclass(slots=True)
class OperationNode:
    """A mathematical operation and its input connections."""

    operation: Operation
    inputs: dict[int, str] = field(default_factory=dict)


@dataclass(slots=True)
class OutputNode:
    """A graph result with one input connection."""

    inputs: dict[int, str] = field(default_factory=dict)


GraphNode: TypeAlias = ValueNode | OperationNode | OutputNode


class MathGraph:
    """Own and evaluate value, operation and output nodes."""

    def __init__(self) -> None:
        """Create an empty graph."""
        self._nodes: dict[str, GraphNode] = {}
        self._next_id = 1

    def _add_node(self, node: GraphNode) -> str:
        """Add a graph node and return its stable identifier."""
        node_id = f"node-{self._next_id}"
        self._next_id += 1
        self._nodes[node_id] = node
        return node_id

    def add_value(self, math_type: MathType, components: tuple[float, ...]) -> str:
        """Add a typed value node to the graph."""
        _validate_components(math_type, components)
        return self._add_node(ValueNode(math_type, tuple(components)))

    def add_operation(self, operation: Operation) -> str:
        """Add an operation node to the graph."""
        return self._add_node(OperationNode(operation))

    def set_value(self, node_id: str, components: tuple[float, ...]) -> None:
        """Replace the components stored by a value node."""
        node = self._nodes[node_id]
        if not isinstance(node, ValueNode):
            raise ValueError("Only value nodes store editable components")
        _validate_components(node.math_type, components)
        node.components = tuple(components)

    def add_output(self) -> str:
        """Add an output node to the graph."""
        return self._add_node(OutputNode())

    def connect(self, source_id: str, target_id: str, input_index: int) -> None:
        """Connect a source node to an input on another node."""
        target = self._nodes[target_id]
        if isinstance(target, ValueNode):
            raise ValueError("Value nodes do not have inputs")
        target.inputs[input_index] = source_id

    def disconnect(self, target_id: str, input_index: int) -> None:
        """Remove one input wire without deleting either node."""
        target = self._nodes[target_id]
        if isinstance(target, ValueNode):
            raise ValueError("Value nodes do not have inputs")
        target.inputs.pop(input_index, None)

    def remove_node(self, node_id: str) -> None:
        """Delete a node and clear any downstream inputs that referenced it."""
        del self._nodes[node_id]
        for node in self._nodes.values():
            if isinstance(node, ValueNode):
                continue
            for input_index, source_id in list(node.inputs.items()):
                if source_id == node_id:
                    del node.inputs[input_index]

    def evaluate(self, node_id: str) -> MathValue:
        """Evaluate a graph node and all the inputs below it."""
        return self._evaluate(node_id, set())

    def _evaluate(self, node_id: str, active_nodes: set[str]) -> MathValue:
        """Evaluate a node whilst tracking the current recursion path."""
        if node_id in active_nodes:
            raise GraphError("The graph contains a cycle")
        active_nodes.add(node_id)

        node = self._nodes[node_id]
        try:
            if isinstance(node, ValueNode):
                return VALUE_CLASSES[node.math_type](*node.components)
            if isinstance(node, OutputNode):
                if 0 not in node.inputs:
                    raise GraphError("Output needs input Value")
                return self._evaluate(node.inputs[0], active_nodes)

            for input_index in range(OPERATION_ARITY[node.operation]):
                if input_index not in node.inputs:
                    input_name = OPERATION_INPUT_NAMES[node.operation][input_index]
                    raise GraphError(f"{node.operation.value} needs input {input_name}")

            values = tuple(
                self._evaluate(node.inputs[input_index], active_nodes)
                for input_index in range(OPERATION_ARITY[node.operation])
            )
            return apply_operation(node.operation, *values)
        finally:
            active_nodes.remove(node_id)
