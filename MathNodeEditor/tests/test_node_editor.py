"""Qt integration tests for the maths node editor."""

import os
from importlib import import_module

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import QApplication, QGroupBox, QPushButton


def _node_editor_module():
    """Load the editor module whilst keeping the first TDD failure readable."""
    try:
        return import_module("MathNodeEditor.node_editor")
    except ModuleNotFoundError:
        pytest.fail("MathNodeEditor.node_editor has not been implemented")


@pytest.fixture(scope="module")
def application() -> QApplication:
    """Return the shared Qt application used by the window test."""
    return QApplication.instance() or QApplication([])


def test_example_graph_displays_the_vec3_multiply_result(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)

    application.processEvents()

    assert window.canvas.output_texts() == ["Vec3(4, 10, 18)"]
    window.close()


@pytest.mark.parametrize(
    ("button_text", "node_class_name"),
    [
        ("Float", "ValueNodeItem"),
        ("Vec2", "ValueNodeItem"),
        ("Quaternion", "ValueNodeItem"),
        ("Matrix Multiply", "OperationNodeItem"),
        ("Look At", "OperationNodeItem"),
        ("Perspective", "OperationNodeItem"),
        ("Output", "OutputNodeItem"),
    ],
)
def test_palette_buttons_add_the_requested_node_type(
    application: QApplication,
    button_text: str,
    node_class_name: str,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    button = next(
        child
        for child in window.palette.findChildren(QPushButton)
        if child.text() == button_text
    )

    button.click()
    application.processEvents()

    assert len(window.canvas.nodes) == 1
    added_node = next(iter(window.canvas.nodes.values()))
    assert type(added_node).__name__ == node_class_name
    window.close()


def test_mat4_result_fits_inside_the_output_node(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    value_node = window.canvas.add_value_node(node_editor.MathType.MAT4)
    output_node = window.canvas.add_output_node()
    assert value_node.output_port is not None
    window.canvas.connect_ports(value_node.output_port, output_node.input_ports[0])

    application.processEvents()

    text_bounds = output_node.result_text.mapRectToParent(
        output_node.result_text.boundingRect()
    )
    assert text_bounds.bottom() <= output_node.height - 8.0
    window.close()


def test_new_quaternion_node_starts_as_the_identity(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)

    quaternion_node = window.canvas.add_value_node(node_editor.MathType.QUATERNION)
    result = window.canvas.graph.evaluate(quaternion_node.node_id)

    assert result.to_list() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    window.close()


def test_quaternion_node_labels_its_component_order(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)

    quaternion_node = window.canvas.add_value_node(node_editor.MathType.QUATERNION)

    assert quaternion_node.title == "Quaternion (s, x, y, z)"
    window.close()


@pytest.mark.parametrize(
    ("operation_name", "expected_names"),
    [
        ("LOOK_AT", ("Eye", "Target", "Up")),
        ("PERSPECTIVE", ("FOV", "Aspect", "Near", "Far")),
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


def test_palette_scrolls_when_all_extended_nodes_are_available(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    window.show()

    application.processEvents()

    assert window.palette_scroll.verticalScrollBar().maximum() > 0
    window.close()


def test_long_quaternion_operation_title_fits_inside_node(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    operation_node = window.canvas.add_operation_node(
        node_editor.Operation.QUATERNION_FROM_AXIS_ANGLE
    )
    title_font = QFont()
    title_font.setPointSize(10)
    title_font.setBold(True)

    required_width = (
        QFontMetrics(title_font).horizontalAdvance(operation_node.title) + 24
    )

    assert operation_node.width >= required_width
    window.close()


def test_mat4_output_keeps_each_matrix_row_on_one_line(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    components = tuple(float(value) + 0.12345 for value in range(1, 17))
    matrix_node = window.canvas.add_value_node(
        node_editor.MathType.MAT4,
        components=components,
    )
    output_node = window.canvas.add_output_node()
    assert matrix_node.output_port is not None
    window.canvas.connect_ports(matrix_node.output_port, output_node.input_ports[0])

    application.processEvents()

    document = output_node.result_text.document()
    visual_line_count = 0
    block = document.begin()
    while block.isValid():
        visual_line_count += block.layout().lineCount()
        block = block.next()
    expected_line_count = output_node.value_text().count("\n") + 1
    assert visual_line_count == expected_line_count
    window.close()


def test_long_quaternion_palette_label_is_not_clipped(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    window.show()
    button = next(
        child
        for child in window.palette.findChildren(QPushButton)
        if child.text() == "Quaternion from Axis Angle"
    )

    application.processEvents()

    assert button.width() >= button.sizeHint().width()
    window.close()


def test_palette_groups_extended_operations_by_domain(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)

    group_titles = [group.title() for group in window.palette.findChildren(QGroupBox)]

    assert group_titles == ["Values", "Maths", "Mat4", "Quaternion", "Result"]
    window.close()
