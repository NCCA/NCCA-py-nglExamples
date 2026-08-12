"""Tests for the maths graph used by the node editor demo."""

from importlib import import_module

import pytest
from ncca.ngl import Mat2, Mat3, Mat4, Vec2, Vec3, Vec4


def _math_graph_module():
    """Load the graph module whilst keeping the first TDD failure readable."""
    try:
        return import_module("MathNodeEditor.math_graph")
    except ModuleNotFoundError:
        pytest.fail("MathNodeEditor.math_graph has not been implemented")


def test_vec3_multiply_is_component_wise() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    left = graph.add_value(graph_module.MathType.VEC3, (1.0, 2.0, 3.0))
    right = graph.add_value(graph_module.MathType.VEC3, (4.0, 5.0, 6.0))
    multiply = graph.add_operation(graph_module.Operation.MULTIPLY)
    output = graph.add_output()
    graph.connect(left, multiply, 0)
    graph.connect(right, multiply, 1)
    graph.connect(multiply, output, 0)

    result = graph.evaluate(output)

    assert result.to_list() == pytest.approx([4.0, 10.0, 18.0])


@pytest.mark.parametrize(
    ("type_name", "value_class", "components"),
    [
        ("VEC2", Vec2, (1.0, 2.0)),
        ("VEC3", Vec3, (1.0, 2.0, 3.0)),
        ("VEC4", Vec4, (1.0, 2.0, 3.0, 4.0)),
        ("MAT2", Mat2, (1.0, 2.0, 3.0, 4.0)),
        ("MAT3", Mat3, tuple(float(value) for value in range(1, 10))),
        ("MAT4", Mat4, tuple(float(value) for value in range(1, 17))),
    ],
)
def test_value_nodes_construct_the_requested_pyngl_type(
    type_name: str,
    value_class: type,
    components: tuple[float, ...],
) -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()

    value_node = graph.add_value(graph_module.MathType[type_name], components)
    result = graph.evaluate(value_node)

    assert isinstance(result, value_class)
    assert result.to_list() == pytest.approx(components)


@pytest.mark.parametrize(
    ("operation_name", "left_type", "left", "right_type", "right", "expected"),
    [
        ("ADD", "VEC3", (1, 2, 3), "VEC3", (4, 5, 6), (5, 7, 9)),
        ("SUBTRACT", "VEC3", (5, 7, 9), "VEC3", (4, 5, 6), (1, 2, 3)),
        (
            "MATRIX_MULTIPLY",
            "MAT2",
            (1, 2, 3, 4),
            "MAT2",
            (1, 2, 3, 4),
            (7, 10, 15, 22),
        ),
        ("MATRIX_MULTIPLY", "MAT2", (1, 2, 3, 4), "VEC2", (5, 6), (17, 39)),
        ("DOT", "VEC3", (1, 2, 3), "VEC3", (4, 5, 6), 32.0),
        ("CROSS", "VEC3", (1, 2, 3), "VEC3", (4, 5, 6), (-3, 6, -3)),
    ],
)
def test_binary_operation_nodes_use_pyngl_math(
    operation_name: str,
    left_type: str,
    left: tuple[float, ...],
    right_type: str,
    right: tuple[float, ...],
    expected: tuple[float, ...] | float,
) -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    left_node = graph.add_value(graph_module.MathType[left_type], left)
    right_node = graph.add_value(graph_module.MathType[right_type], right)
    operation = graph.add_operation(graph_module.Operation[operation_name])
    graph.connect(left_node, operation, 0)
    graph.connect(right_node, operation, 1)

    result = graph.evaluate(operation)

    if isinstance(expected, tuple):
        assert result.to_list() == pytest.approx(expected)
    else:
        assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    ("operation_name", "math_type", "components", "expected"),
    [
        ("NORMALISE", "VEC3", (3.0, 0.0, 4.0), (0.6, 0.0, 0.8)),
        ("TRANSPOSE", "MAT2", (1.0, 2.0, 3.0, 4.0), (1.0, 3.0, 2.0, 4.0)),
    ],
)
def test_unary_operation_nodes_use_pyngl_math(
    operation_name: str,
    math_type: str,
    components: tuple[float, ...],
    expected: tuple[float, ...],
) -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    value_node = graph.add_value(graph_module.MathType[math_type], components)
    operation = graph.add_operation(graph_module.Operation[operation_name])
    graph.connect(value_node, operation, 0)

    result = graph.evaluate(operation)

    assert result.to_list() == pytest.approx(expected)


def test_changing_a_value_updates_downstream_results() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    left = graph.add_value(graph_module.MathType.VEC2, (1.0, 2.0))
    right = graph.add_value(graph_module.MathType.VEC2, (10.0, 20.0))
    add = graph.add_operation(graph_module.Operation.ADD)
    output = graph.add_output()
    graph.connect(left, add, 0)
    graph.connect(right, add, 1)
    graph.connect(add, output, 0)

    graph.set_value(left, (3.0, 4.0))

    assert graph.evaluate(output).to_list() == pytest.approx([13.0, 24.0])


def test_incomplete_operation_reports_the_missing_input() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    value_node = graph.add_value(graph_module.MathType.VEC3, (1.0, 2.0, 3.0))
    add = graph.add_operation(graph_module.Operation.ADD)
    graph.connect(value_node, add, 0)

    with pytest.raises(graph_module.GraphError, match="input B"):
        graph.evaluate(add)


def test_unconnected_output_reports_the_missing_value() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    output = graph.add_output()

    with pytest.raises(graph_module.GraphError, match="Output needs input Value"):
        graph.evaluate(output)


def test_evaluation_rejects_cycles() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    add = graph.add_operation(graph_module.Operation.ADD)
    graph.connect(add, add, 0)
    graph.connect(add, add, 1)

    with pytest.raises(graph_module.GraphError, match="cycle"):
        graph.evaluate(add)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Vec3(1.0, 2.5, 3.0), "Vec3(1, 2.5, 3)"),
        (Mat2(1.0, 2.0, 3.0, 4.0), "Mat2\n[1  2]\n[3  4]"),
        (32.0, "32"),
    ],
)
def test_values_are_formatted_for_output_nodes(value: object, expected: str) -> None:
    graph_module = _math_graph_module()

    result = graph_module.format_value(value)

    assert result == expected


def test_value_nodes_reject_the_wrong_number_of_components() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()

    with pytest.raises(graph_module.GraphError, match="Vec4 needs 4 components"):
        graph.add_value(graph_module.MathType.VEC4, (1.0, 2.0, 3.0))


@pytest.mark.parametrize("operation_name", ["ADD", "MULTIPLY"])
def test_incompatible_operation_inputs_report_a_graph_error(
    operation_name: str,
) -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    vec2 = graph.add_value(graph_module.MathType.VEC2, (1.0, 2.0))
    vec3 = graph.add_value(graph_module.MathType.VEC3, (1.0, 2.0, 3.0))
    operation = graph.add_operation(graph_module.Operation[operation_name])
    graph.connect(vec2, operation, 0)
    graph.connect(vec3, operation, 1)

    with pytest.raises(graph_module.GraphError, match="failed"):
        graph.evaluate(operation)
