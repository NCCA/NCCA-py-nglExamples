"""Calculation graph for the PyNGL maths node editor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from ncca.ngl import Mat2, Mat3, Mat4, Vec2, Vec3, Vec4


class MathType(Enum):
    """Maths value types accepted by value nodes."""

    VEC2 = "Vec2"
    VEC3 = "Vec3"
    VEC4 = "Vec4"
    MAT2 = "Mat2"
    MAT3 = "Mat3"
    MAT4 = "Mat4"


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


OPERATION_ARITY: dict[Operation, int] = {
    Operation.ADD: 2,
    Operation.SUBTRACT: 2,
    Operation.MULTIPLY: 2,
    Operation.MATRIX_MULTIPLY: 2,
    Operation.DOT: 2,
    Operation.CROSS: 2,
    Operation.NORMALISE: 1,
    Operation.TRANSPOSE: 1,
}


MathValue: TypeAlias = Vec2 | Vec3 | Vec4 | Mat2 | Mat3 | Mat4 | float

VALUE_CLASSES: dict[MathType, Callable[..., MathValue]] = {
    MathType.VEC2: Vec2,
    MathType.VEC3: Vec3,
    MathType.VEC4: Vec4,
    MathType.MAT2: Mat2,
    MathType.MAT3: Mat3,
    MathType.MAT4: Mat4,
}

TYPE_SHAPES: dict[MathType, tuple[int, int]] = {
    MathType.VEC2: (1, 2),
    MathType.VEC3: (1, 3),
    MathType.VEC4: (1, 4),
    MathType.MAT2: (2, 2),
    MathType.MAT3: (3, 3),
    MathType.MAT4: (4, 4),
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


def apply_operation(
    operation: Operation,
    left: MathValue,
    right: MathValue | None = None,
) -> MathValue:
    """Apply an operation and translate PyNGL errors for the graph UI."""
    try:
        if operation is Operation.NORMALISE:
            return left.normalized()
        if operation is Operation.TRANSPOSE:
            return left.transposed()

        if right is None:
            raise GraphError(f"{operation.value} needs input B")
        if operation is Operation.ADD:
            return left + right
        if operation is Operation.SUBTRACT:
            return left - right
        if operation is Operation.MATRIX_MULTIPLY:
            return left @ right
        if operation is Operation.DOT:
            return float(left.dot(right))
        if operation is Operation.CROSS:
            return left.cross(right)

        if type(left) is not type(right):
            raise ValueError("component multiply needs matching input types")
        components = tuple(a * b for a, b in zip(left.to_list(), right.to_list()))
        return type(left)(*components)
    except GraphError:
        raise
    except (AttributeError, TypeError, ValueError) as error:
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
                    input_name = chr(ord("A") + input_index)
                    raise GraphError(f"{node.operation.value} needs input {input_name}")

            left = self._evaluate(node.inputs[0], active_nodes)
            right = None
            if OPERATION_ARITY[node.operation] == 2:
                right = self._evaluate(node.inputs[1], active_nodes)
            return apply_operation(node.operation, left, right)
        finally:
            active_nodes.remove(node_id)
