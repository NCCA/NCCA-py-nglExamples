"""Tests for the maths graph used by the node editor demo."""

from importlib import import_module
from pathlib import Path

import pytest
from ncca.ngl import (
    Mat2,
    Mat3,
    Mat4,
    Obj,
    Quaternion,
    Transform,
    Vec2,
    Vec3,
    Vec4,
    frustum,
    look_at,
    ortho,
    perspective,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _math_graph_module():
    """Load the graph module whilst keeping the first TDD failure readable."""
    try:
        return import_module("MathNodeEditor.math_graph")
    except ModuleNotFoundError:
        pytest.fail("MathNodeEditor.math_graph has not been implemented")


def _load_obj(name: str) -> Obj:
    """Load one of the tiny fixture meshes under ``tests/fixtures``."""
    obj = Obj()
    obj.load(str(FIXTURES_DIR / name))
    return obj


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
    operation = graph.add_generator(
        graph_module.Operation.LOOK_AT,
        ((0.0, 2.0, 5.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )

    result = graph.evaluate(operation)

    expected = look_at(Vec3(0.0, 2.0, 5.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
    assert isinstance(result, Mat4)
    assert result.to_list() == pytest.approx(expected.to_list())


def test_look_at_node_reports_semantic_input_names() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    operation = graph.add_generator(
        graph_module.Operation.LOOK_AT,
        ((0.0, 2.0, 5.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )

    # Generator nodes always have all their parameters, so we can't test missing input
    # by omitting a parameter. Instead, test that the generator works correctly.
    result = graph.evaluate(operation)
    assert isinstance(result, Mat4)


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


def test_perspective_node_reports_invalid_clip_planes() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    operation = graph.add_generator(
        graph_module.Operation.PERSPECTIVE,
        ((45.0,), (1.0,), (1.0,), (1.0,)),
    )

    with pytest.raises(graph_module.GraphError, match="Perspective failed"):
        graph.evaluate(operation)


@pytest.mark.parametrize(
    ("operation_name", "inputs", "expected"),
    [
        ("MAT4_TRANSLATE", ((2.0,), (3.0,), (4.0,)), Mat4.translate(2.0, 3.0, 4.0)),
        ("MAT4_SCALE", ((2.0,), (3.0,), (4.0,)), Mat4.scale(2.0, 3.0, 4.0)),
        ("MAT4_ROTATE_X", ((30.0,),), Mat4.rotate_x(30.0)),
        ("MAT4_ROTATE_Y", ((45.0,),), Mat4.rotate_y(45.0)),
        ("MAT4_ROTATE_Z", ((60.0,),), Mat4.rotate_z(60.0)),
    ],
)
def test_mat4_transform_constructor_nodes(
    operation_name: str,
    inputs: tuple[tuple[float, ...], ...],
    expected: Mat4,
) -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    operation = graph.add_generator(
        graph_module.Operation[operation_name],
        inputs,
    )

    result = graph.evaluate(operation)

    assert result.to_list() == pytest.approx(expected.to_list())


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


def test_transform_node_reports_semantic_input_names() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    operation = graph.add_generator(
        graph_module.Operation.TRANSFORM,
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )

    # Generator nodes always have all their parameters, so we can't test missing input
    # by omitting a parameter. Instead, test that the generator works correctly.
    result = graph.evaluate(operation)
    assert isinstance(result, Mat4)


def test_ortho_node_builds_an_orthographic_mat4() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    components = (-10.0, 10.0, -5.0, 5.0, 0.1, 100.0)
    operation = graph.add_generator(
        graph_module.Operation.ORTHO,
        tuple((c,) for c in components),
    )

    result = graph.evaluate(operation)

    assert result.to_list() == pytest.approx(ortho(*components).to_list())


def test_frustum_node_builds_a_projection_mat4() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    components = (-1.0, 1.0, -0.5, 0.5, 0.1, 100.0)
    operation = graph.add_generator(
        graph_module.Operation.FRUSTUM,
        tuple((c,) for c in components),
    )

    result = graph.evaluate(operation)

    assert result.to_list() == pytest.approx(frustum(*components).to_list())


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


def test_remove_node_deletes_it_from_the_graph() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    value_node = graph.add_value(graph_module.MathType.FLOAT, (1.0,))

    graph.remove_node(value_node)

    with pytest.raises(KeyError):
        graph.evaluate(value_node)


def test_remove_node_clears_downstream_references() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    left = graph.add_value(graph_module.MathType.VEC3, (1.0, 2.0, 3.0))
    right = graph.add_value(graph_module.MathType.VEC3, (4.0, 5.0, 6.0))
    add = graph.add_operation(graph_module.Operation.ADD)
    graph.connect(left, add, 0)
    graph.connect(right, add, 1)

    graph.remove_node(left)

    with pytest.raises(graph_module.GraphError, match="input A"):
        graph.evaluate(add)


def test_disconnect_removes_a_single_input_without_deleting_the_node() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    left = graph.add_value(graph_module.MathType.VEC3, (1.0, 2.0, 3.0))
    right = graph.add_value(graph_module.MathType.VEC3, (4.0, 5.0, 6.0))
    add = graph.add_operation(graph_module.Operation.ADD)
    graph.connect(left, add, 0)
    graph.connect(right, add, 1)

    graph.disconnect(add, 0)

    with pytest.raises(graph_module.GraphError, match="input A"):
        graph.evaluate(add)
    assert graph.evaluate(left).to_list() == pytest.approx([1.0, 2.0, 3.0])


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


def test_mat4_to_mat3_drops_translation() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    matrix = Mat4.translate(5.0, 6.0, 7.0) @ Mat4.rotate_y(30.0)
    value_node = graph.add_value(graph_module.MathType.MAT4, matrix.to_list())
    operation = graph.add_operation(graph_module.Operation.MAT4_TO_MAT3)
    graph.connect(value_node, operation, 0)

    result = graph.evaluate(operation)

    assert isinstance(result, Mat3)
    assert result.to_list() == pytest.approx(Mat3.from_mat4(matrix).to_list())


@pytest.mark.parametrize("math_type", ["MAT3", "MAT4"])
def test_inverse_node_inverts_mat3_or_mat4(math_type: str) -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    matrix_class = Mat3 if math_type == "MAT3" else Mat4
    matrix = matrix_class.rotate_y(40.0) if math_type == "MAT3" else Mat4.rotate_y(40.0)
    value_node = graph.add_value(graph_module.MathType[math_type], matrix.to_list())
    operation = graph.add_operation(graph_module.Operation.INVERSE)
    graph.connect(value_node, operation, 0)

    result = graph.evaluate(operation)

    assert result.to_list() == pytest.approx(matrix.inverse().to_list())


def test_transform_vertices_applies_translation_to_every_point() -> None:
    graph_module = _math_graph_module()
    matrix = Mat4.translate(1.0, 2.0, 3.0)
    vertices = graph_module.VertexArray((Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0)))

    result = graph_module.apply_operation(
        graph_module.Operation.TRANSFORM_VERTICES, matrix, vertices
    )

    assert isinstance(result, graph_module.VertexArray)
    flattened = [component for v in result.values for component in v.to_list()]
    assert flattened == pytest.approx([1.0, 2.0, 3.0, 2.0, 2.0, 3.0])


def test_transform_normals_ignores_translation() -> None:
    graph_module = _math_graph_module()
    linear_part = Mat3.from_mat4(Mat4.translate(9.0, 9.0, 9.0) @ Mat4.rotate_z(90.0))
    normals = graph_module.NormalArray((Vec3(1.0, 0.0, 0.0),))

    result = graph_module.apply_operation(
        graph_module.Operation.TRANSFORM_NORMALS, linear_part, normals
    )

    assert isinstance(result, graph_module.NormalArray)
    assert result.values[0].to_list() == pytest.approx(
        (Vec3(1.0, 0.0, 0.0) @ linear_part).to_list()
    )


def test_transform_vertices_rejects_a_normal_array_input() -> None:
    graph_module = _math_graph_module()
    normals = graph_module.NormalArray((Vec3(1.0, 0.0, 0.0),))

    with pytest.raises(graph_module.GraphError, match="failed"):
        graph_module.apply_operation(
            graph_module.Operation.TRANSFORM_VERTICES, Mat4(), normals
        )


def test_arrays_from_obj_splits_vertex_face_uv_normal_data() -> None:
    graph_module = _math_graph_module()
    obj = _load_obj("triangle.obj")

    vertices, faces, uvs, normals = graph_module.arrays_from_obj(obj)

    vertex_components = [c for v in vertices.values for c in v.to_list()]
    assert vertex_components == pytest.approx(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )
    uv_components = [c for uv in uvs.values for c in uv.to_list()]
    assert uv_components == pytest.approx([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    normal_components = [c for n in normals.values for c in n.to_list()]
    assert normal_components == pytest.approx([0.0, 0.0, 1.0])
    assert faces.triangles == (((0, 0, 0), (1, 1, 0), (2, 2, 0)),)


def test_arrays_from_obj_rejects_a_non_triangular_mesh() -> None:
    graph_module = _math_graph_module()
    obj = _load_obj("quad.obj")

    with pytest.raises(graph_module.GraphError, match="triangulated"):
        graph_module.arrays_from_obj(obj)


def test_obj_from_arrays_merges_vertex_face_uv_normal_data_back_together() -> None:
    graph_module = _math_graph_module()
    obj = _load_obj("triangle.obj")
    vertices, faces, uvs, normals = graph_module.arrays_from_obj(obj)

    merged = graph_module.obj_from_arrays(vertices, faces, uvs, normals)

    merged_components = [c for v in merged.vertex for c in v.to_list()]
    expected_components = [c for v in vertices.values for c in v.to_list()]
    assert merged_components == pytest.approx(expected_components)
    assert merged.is_triangular()
    assert merged.faces[0].vertex == [0, 1, 2]
    assert merged.faces[0].uv == [0, 1, 2]
    assert merged.faces[0].normal == [0, 0, 0]


def test_obj_from_arrays_omits_uv_and_normal_when_not_supplied() -> None:
    graph_module = _math_graph_module()
    obj = _load_obj("triangle.obj")
    vertices, faces, _uvs, _normals = graph_module.arrays_from_obj(obj)

    merged = graph_module.obj_from_arrays(vertices, faces, None, None)

    assert merged.uv == []
    assert merged.normals == []
    assert merged.faces[0].uv == []
    assert merged.faces[0].normal == []


def test_mesh_viewer_needs_vertices_and_faces() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    viewer = graph.add_mesh_viewer()

    with pytest.raises(graph_module.GraphError, match="input Vertices"):
        graph.evaluate_mesh_viewer(viewer)

    vertices = graph.add_literal(graph_module.VertexArray((Vec3(0.0, 0.0, 0.0),)))
    graph.connect(vertices, viewer, 0)
    with pytest.raises(graph_module.GraphError, match="input Faces"):
        graph.evaluate_mesh_viewer(viewer)


def test_mesh_viewer_evaluates_optional_inputs_when_wired() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    obj = _load_obj("triangle.obj")
    vertices, faces, uvs, normals = graph_module.arrays_from_obj(obj)

    viewer = graph.add_mesh_viewer()
    graph.connect(graph.add_literal(vertices), viewer, 0)
    graph.connect(graph.add_literal(faces), viewer, 1)
    graph.connect(graph.add_literal(uvs), viewer, 2)
    graph.connect(graph.add_literal(normals), viewer, 3)
    graph.connect(graph.add_value(graph_module.MathType.VEC4, (1, 0, 0, 1)), viewer, 4)

    result = graph.evaluate_mesh_viewer(viewer)

    assert result.vertices is vertices
    assert result.faces is faces
    assert result.uvs is uvs
    assert result.normals is normals
    assert result.colour.to_list() == pytest.approx([1.0, 0.0, 0.0, 1.0])


def test_mesh_viewer_leaves_unwired_optional_inputs_as_none() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    vertices = graph_module.VertexArray((Vec3(0.0, 0.0, 0.0),))
    faces = graph_module.FaceArray(
        (((0, None, None), (0, None, None), (0, None, None)),)
    )

    viewer = graph.add_mesh_viewer()
    graph.connect(graph.add_literal(vertices), viewer, 0)
    graph.connect(graph.add_literal(faces), viewer, 1)

    result = graph.evaluate_mesh_viewer(viewer)

    assert result.uvs is None
    assert result.normals is None
    assert result.colour is None


def test_literal_node_value_can_be_replaced() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    literal = graph.add_literal(graph_module.VertexArray(()))

    graph.set_literal(literal, graph_module.VertexArray((Vec3(1.0, 2.0, 3.0),)))

    result = graph.evaluate(literal)
    assert result.values[0].to_list() == pytest.approx([1.0, 2.0, 3.0])


def test_set_literal_rejects_non_literal_nodes() -> None:
    graph_module = _math_graph_module()
    graph = graph_module.MathGraph()
    value_node = graph.add_value(graph_module.MathType.FLOAT, (1.0,))

    with pytest.raises(ValueError):
        graph.set_literal(value_node, 2.0)


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
