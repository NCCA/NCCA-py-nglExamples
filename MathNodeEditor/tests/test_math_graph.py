"""Tests for the maths graph used by the node editor demo."""

from importlib import import_module

import pytest
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


def test_quaternion_value_node_uses_scalar_then_vector_component_order() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()

    value_node = graph.add_value(
        graph_module.MathType.QUATERNION,
        (1.0, 0.25, 0.5, 0.75),
    )
    result = graph.evaluate(value_node)

    assert isinstance(result, Quaternion)
    assert result.to_list() == pytest.approx([1.0, 0.25, 0.5, 0.75])


def test_float_value_node_supplies_scalar_operation_inputs() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()

    value_node = graph.add_value(graph_module.MathType.FLOAT, (45.0,))
    result = graph.evaluate(value_node)

    assert result == pytest.approx(45.0)


def test_float_nodes_can_use_the_basic_arithmetic_operations() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    left = graph.add_value(graph_module.MathType.FLOAT, (3.0,))
    right = graph.add_value(graph_module.MathType.FLOAT, (4.0,))
    multiply = graph.add_operation(graph_module.Operation.MULTIPLY)
    graph.connect(left, multiply, 0)
    graph.connect(right, multiply, 1)

    result = graph.evaluate(multiply)

    assert result == pytest.approx(12.0)


def test_look_at_node_builds_a_mat4_from_three_vec3_inputs() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    eye = graph.add_value(graph_module.MathType.VEC3, (0.0, 2.0, 5.0))
    target = graph.add_value(graph_module.MathType.VEC3, (0.0, 0.0, 0.0))
    up = graph.add_value(graph_module.MathType.VEC3, (0.0, 1.0, 0.0))
    operation = graph.add_operation(graph_module.Operation.LOOK_AT)
    graph.connect(eye, operation, 0)
    graph.connect(target, operation, 1)
    graph.connect(up, operation, 2)

    result = graph.evaluate(operation)

    expected = look_at(Vec3(0.0, 2.0, 5.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
    assert isinstance(result, Mat4)
    assert result.to_list() == pytest.approx(expected.to_list())


def test_look_at_node_reports_semantic_input_names() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    eye = graph.add_value(graph_module.MathType.VEC3, (0.0, 2.0, 5.0))
    target = graph.add_value(graph_module.MathType.VEC3, (0.0, 0.0, 0.0))
    operation = graph.add_operation(graph_module.Operation.LOOK_AT)
    graph.connect(eye, operation, 0)
    graph.connect(target, operation, 1)

    with pytest.raises(graph_module.GraphError, match="input Up"):
        graph.evaluate(operation)


def test_perspective_node_builds_a_mat4_from_four_float_inputs() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    values = [
        graph.add_value(graph_module.MathType.FLOAT, (component,))
        for component in (45.0, 16.0 / 9.0, 0.1, 100.0)
    ]
    operation = graph.add_operation(graph_module.Operation.PERSPECTIVE)
    for input_index, value_node in enumerate(values):
        graph.connect(value_node, operation, input_index)

    result = graph.evaluate(operation)

    expected = perspective(45.0, 16.0 / 9.0, 0.1, 100.0)
    assert isinstance(result, Mat4)
    assert result.to_list() == pytest.approx(expected.to_list())


def test_perspective_node_reports_invalid_clip_planes() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    values = [
        graph.add_value(graph_module.MathType.FLOAT, (component,))
        for component in (45.0, 1.0, 1.0, 1.0)
    ]
    operation = graph.add_operation(graph_module.Operation.PERSPECTIVE)
    for input_index, value_node in enumerate(values):
        graph.connect(value_node, operation, input_index)

    with pytest.raises(graph_module.GraphError, match="Perspective failed"):
        graph.evaluate(operation)


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
    value_nodes = [
        graph.add_value(graph_module.MathType.FLOAT, (component,))
        for component in inputs
    ]
    operation = graph.add_operation(graph_module.Operation[operation_name])
    for input_index, value_node in enumerate(value_nodes):
        graph.connect(value_node, operation, input_index)

    result = graph.evaluate(operation)

    assert result.to_list() == pytest.approx(expected.to_list())


def test_ortho_node_builds_an_orthographic_mat4() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    components = (-10.0, 10.0, -5.0, 5.0, 0.1, 100.0)
    value_nodes = [
        graph.add_value(graph_module.MathType.FLOAT, (component,))
        for component in components
    ]
    operation = graph.add_operation(graph_module.Operation.ORTHO)
    for input_index, value_node in enumerate(value_nodes):
        graph.connect(value_node, operation, input_index)

    result = graph.evaluate(operation)

    assert result.to_list() == pytest.approx(ortho(*components).to_list())


def test_frustum_node_builds_a_projection_mat4() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    components = (-1.0, 1.0, -0.5, 0.5, 0.1, 100.0)
    value_nodes = [
        graph.add_value(graph_module.MathType.FLOAT, (component,))
        for component in components
    ]
    operation = graph.add_operation(graph_module.Operation.FRUSTUM)
    for input_index, value_node in enumerate(value_nodes):
        graph.connect(value_node, operation, input_index)

    result = graph.evaluate(operation)

    assert result.to_list() == pytest.approx(frustum(*components).to_list())


def test_axis_angle_node_builds_a_quaternion() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    axis = graph.add_value(graph_module.MathType.VEC3, (0.0, 1.0, 0.0))
    angle = graph.add_value(graph_module.MathType.FLOAT, (90.0,))
    operation = graph.add_operation(graph_module.Operation.QUATERNION_FROM_AXIS_ANGLE)
    graph.connect(axis, operation, 0)
    graph.connect(angle, operation, 1)

    result = graph.evaluate(operation)

    expected = Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), 90.0)
    assert isinstance(result, Quaternion)
    assert result.to_list() == pytest.approx(expected.to_list())


def test_quaternion_product_node_uses_the_hamilton_product() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    left_value = Quaternion.from_axis_angle(Vec3(1.0, 0.0, 0.0), 30.0)
    right_value = Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), 45.0)
    left = graph.add_value(graph_module.MathType.QUATERNION, left_value.to_tuple())
    right = graph.add_value(graph_module.MathType.QUATERNION, right_value.to_tuple())
    operation = graph.add_operation(graph_module.Operation.QUATERNION_PRODUCT)
    graph.connect(left, operation, 0)
    graph.connect(right, operation, 1)

    result = graph.evaluate(operation)

    assert isinstance(result, Quaternion)
    assert result.to_list() == pytest.approx((left_value @ right_value).to_list())


def test_quaternion_rotate_vector_node_returns_a_vec3() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    quaternion = Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), 90.0)
    quaternion_node = graph.add_value(
        graph_module.MathType.QUATERNION,
        quaternion.to_tuple(),
    )
    vector_node = graph.add_value(graph_module.MathType.VEC3, (1.0, 0.0, 0.0))
    operation = graph.add_operation(graph_module.Operation.QUATERNION_ROTATE_VECTOR)
    graph.connect(quaternion_node, operation, 0)
    graph.connect(vector_node, operation, 1)

    result = graph.evaluate(operation)

    assert isinstance(result, Vec3)
    assert result.to_list() == pytest.approx(
        (quaternion * Vec3(1.0, 0.0, 0.0)).to_list()
    )


def test_quaternion_and_mat4_conversion_nodes_round_trip() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    quaternion = Quaternion.from_axis_angle(Vec3(0.0, 0.0, 1.0), 35.0)
    quaternion_node = graph.add_value(
        graph_module.MathType.QUATERNION,
        quaternion.to_tuple(),
    )
    to_matrix = graph.add_operation(graph_module.Operation.QUATERNION_TO_MAT4)
    to_quaternion = graph.add_operation(graph_module.Operation.MAT4_TO_QUATERNION)
    graph.connect(quaternion_node, to_matrix, 0)
    graph.connect(to_matrix, to_quaternion, 0)

    matrix_result = graph.evaluate(to_matrix)
    quaternion_result = graph.evaluate(to_quaternion)

    assert matrix_result.to_list() == pytest.approx(quaternion.to_mat4().to_list())
    assert quaternion_result.to_list() == pytest.approx(
        Quaternion.from_mat4(quaternion.to_mat4()).to_list()
    )


def test_quaternion_slerp_node_uses_a_float_blend_value() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    start_value = Quaternion()
    end_value = Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), 90.0)
    start = graph.add_value(graph_module.MathType.QUATERNION, start_value.to_tuple())
    end = graph.add_value(graph_module.MathType.QUATERNION, end_value.to_tuple())
    blend = graph.add_value(graph_module.MathType.FLOAT, (0.5,))
    operation = graph.add_operation(graph_module.Operation.QUATERNION_SLERP)
    graph.connect(start, operation, 0)
    graph.connect(end, operation, 1)
    graph.connect(blend, operation, 2)

    result = graph.evaluate(operation)

    assert isinstance(result, Quaternion)
    assert result.to_list() == pytest.approx(
        start_value.slerp(end_value, 0.5).to_list()
    )


@pytest.mark.parametrize(
    ("operation_name", "expected_method"),
    [
        ("QUATERNION_CONJUGATE", "conjugate"),
        ("QUATERNION_INVERSE", "inverse"),
    ],
)
def test_quaternion_unary_operation_nodes(
    operation_name: str,
    expected_method: str,
) -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    quaternion = Quaternion.from_axis_angle(Vec3(1.0, 2.0, 3.0).normalized(), 47.0)
    value_node = graph.add_value(
        graph_module.MathType.QUATERNION,
        quaternion.to_tuple(),
    )
    operation = graph.add_operation(graph_module.Operation[operation_name])
    graph.connect(value_node, operation, 0)

    result = graph.evaluate(operation)

    expected = getattr(quaternion, expected_method)()
    assert result.to_list() == pytest.approx(expected.to_list())


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
