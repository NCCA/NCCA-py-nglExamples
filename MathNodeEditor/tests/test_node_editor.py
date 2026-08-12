"""Qt integration tests for the maths node editor."""

import os
from importlib import import_module

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QPushButton


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
        ("Vec2", "ValueNodeItem"),
        ("Matrix Multiply", "OperationNodeItem"),
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
