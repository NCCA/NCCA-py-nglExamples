"""Qt integration tests for the maths node editor."""

import json
import os
import sys
from importlib import import_module
from itertools import combinations
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from ncca.ngl import Vec3
from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QCloseEvent, QFont, QFontMetrics, QKeySequence, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QInputDialog,
    QMessageBox,
    QPushButton,
)


def _wheel_event(delta_y: int, position: QPointF | None = None) -> QWheelEvent:
    """Build a synthetic wheel event scrolling up (positive) or down."""
    event_position = position if position is not None else QPointF(0.0, 0.0)
    return QWheelEvent(
        event_position,
        event_position,
        QPoint(0, 0),
        QPoint(0, delta_y),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def _node_editor_module():
    """Load the editor module whilst keeping the first TDD failure readable."""
    try:
        return import_module("node_editor")
    except ModuleNotFoundError:
        pytest.fail("node_editor has not been implemented")


@pytest.fixture(scope="module")
def application() -> QApplication:
    """Return the shared Qt application used by the window test."""
    return QApplication.instance() or QApplication([])


def _isolated_settings(tmp_path: Path) -> QSettings:
    """Build a throwaway QSettings store so tests never touch real user prefs."""
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture(autouse=True)
def _redirect_default_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point every bare QSettings() at a throwaway file for this test only.

    ``MathNodeWindow(load_example=...)`` calls elsewhere in this file (there
    are dozens, from before this task) never pass ``settings=`` explicitly,
    so without this autouse fixture they'd fall through to
    ``node_editor._default_settings()`` -> ``QSettings("NCCA",
    "MathNodeEditor")``, which resolves to the same real, persistent
    preferences store the actual application uses, instead of a clean temp
    file.
    """
    node_editor = _node_editor_module()
    ini_path = str(tmp_path / "default-settings.ini")
    monkeypatch.setattr(
        node_editor,
        "_default_settings",
        lambda: QSettings(ini_path, QSettings.Format.IniFormat),
    )


@pytest.fixture(autouse=True)
def _default_discard_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer "Discard" to any unmocked save-changes prompt.

    Dozens of tests in this file add nodes and then call ``window.close()``
    without caring about the unsaved-changes prompt added in this task. Left
    unmocked, ``closeEvent``'s call to ``QMessageBox.question()`` would raise
    a real modal dialog and hang the test under the offscreen QPA platform.
    Tests that exercise the prompt itself override this with their own
    ``monkeypatch.setattr(QMessageBox, "question", ...)`` call.
    """
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Discard,
    )


def test_example_graph_displays_the_vec3_multiply_result(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)

    application.processEvents()

    assert window.canvas.output_texts() == ["Vec3(4, 10, 18)"]
    window.close()


def test_code_view_action_controls_a_hidden_read_only_dock(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)
    window.show()
    application.processEvents()

    assert window.code_dock.isHidden()
    assert window.code_editor.isReadOnly()
    assert window.code_highlighter.document() is window.code_editor.document()

    window.action_code_view.trigger()
    application.processEvents()

    assert window.code_dock.isVisible()
    assert window.action_code_view.isChecked()
    assert "output_node_4" in window.code_editor.toPlainText()
    window.close()


def test_code_view_refreshes_when_a_graph_connection_changes(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    value = window.canvas.add_value_node(node_editor.MathType.FLOAT, components=(3.0,))
    output = window.canvas.add_output_node()

    window.canvas.connect_ports(value.output_port, output.input_ports[0])
    application.processEvents()

    assert "node_1 = 3.0" in window.code_editor.toPlainText()
    assert "output_node_2 = node_2" in window.code_editor.toPlainText()
    window.close()


def test_code_view_copy_and_save_buttons_export_generated_python(
    application: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)
    expected = window.code_editor.toPlainText()

    window.copy_code_button.click()

    assert application.clipboard().text() == expected

    destination = tmp_path / "generated_graph.py"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(destination), "Python Files (*.py)"),
    )
    window.save_code_button.click()

    assert destination.read_text(encoding="utf-8") == expected
    window.close()


@pytest.mark.parametrize(
    ("button_text", "node_class_name"),
    [
        ("Float", "ValueNodeItem"),
        ("Vec2", "ValueNodeItem"),
        ("Quaternion", "ValueNodeItem"),
        ("Matrix Multiply", "OperationNodeItem"),
        ("Look At", "GeneratorNodeItem"),
        ("Perspective", "GeneratorNodeItem"),
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


def test_transform_generator_exposes_every_rotation_order(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)

    node = window.canvas.add_generator_node(node_editor.Operation.TRANSFORM)

    combo = node.rotation_order_combo
    assert combo is not None
    assert [combo.itemText(index) for index in range(combo.count())] == [
        "xyz",
        "yzx",
        "zxy",
        "xzy",
        "yxz",
        "zyx",
    ]
    assert combo.currentText() == "xyz"
    window.close()


def test_changing_transform_rotation_order_updates_downstream_output(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    transform_node = window.canvas.add_generator_node(
        node_editor.Operation.TRANSFORM,
        parameters=(
            (0.0, 0.0, 0.0),
            (30.0, 45.0, 60.0),
            (1.0, 1.0, 1.0),
        ),
    )
    output_node = window.canvas.add_output_node()
    assert transform_node.output_port is not None
    window.canvas.connect_ports(transform_node.output_port, output_node.input_ports[0])
    application.processEvents()
    before = window.canvas.output_texts()[0]

    assert transform_node.rotation_order_combo is not None
    transform_node.rotation_order_combo.setCurrentText("zyx")
    application.processEvents()

    assert window.canvas.graph.generator_rotation_order(transform_node.node_id) == "zyx"
    assert window.canvas.output_texts()[0] != before
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


def test_quaternion_node_labels_its_component_order(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)

    quaternion_node = window.canvas.add_value_node(node_editor.MathType.QUATERNION)

    assert quaternion_node.title == "Quaternion (s, x, y, z)"
    window.close()


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
    operation_node = window.canvas.add_generator_node(
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


def test_double_clicking_node_header_renames_node(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    node = window.canvas.add_value_node(node_editor.MathType.VEC3)
    window.canvas.mark_clean()
    window.show()
    window.view.centerOn(node)
    monkeypatch.setattr(
        QInputDialog,
        "getText",
        lambda *args, **kwargs: ("Camera Position", True),
    )
    application.processEvents()
    header_position = window.view.mapFromScene(
        node.mapToScene(QPointF(node.width / 2.0, node_editor.NODE_HEADER_HEIGHT / 2.0))
    )

    QTest.mouseDClick(
        window.view.viewport(),
        Qt.MouseButton.LeftButton,
        pos=header_position,
    )
    application.processEvents()

    assert node.title == "Camera Position"
    assert window.canvas.modified is True
    window.close()


def test_long_node_name_expands_node_and_moves_output_port(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    node = window.canvas.add_operation_node(node_editor.Operation.MULTIPLY)
    node.set_name("Projection Matrix @ View Model Matrix")

    required_width = (
        QFontMetrics(node_editor.node_title_font()).horizontalAdvance(node.title) + 24
    )

    assert node.width >= required_width
    assert node.output_port is not None
    assert node.output_port.pos().x() == node.width
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

    assert group_titles == ["Values", "Maths", "Mat4", "Quaternion", "Mesh", "Result"]
    window.close()


def test_palette_and_menu_expose_the_same_catalogue_labels(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    window.show()

    catalogue_labels = {
        label
        for _title, entries in node_editor.NODE_CATALOGUE
        for label, _factory in entries
    }
    palette_labels = {
        button.text()
        for button in window.palette.findChildren(QPushButton)
        if button.text() in catalogue_labels
    }
    menu_labels = {action.text() for action in window.view.node_menu.creation_actions}

    assert palette_labels == catalogue_labels
    assert menu_labels == catalogue_labels
    window.close()


def test_value_nodes_have_type_specific_icons_and_header_colours(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)

    nodes = [
        window.canvas.add_value_node(math_type) for math_type in node_editor.MathType
    ]

    assert all(node.icon_symbol for node in nodes)
    assert len({node.icon_symbol for node in nodes}) == len(node_editor.MathType)
    assert len({node.header_colour.name() for node in nodes}) == len(
        node_editor.MathType
    )
    window.close()


def test_every_operation_has_an_icon_and_domain_colour() -> None:
    node_editor = _node_editor_module()

    styles = {
        operation: node_editor.operation_node_style(operation)
        for operation in node_editor.Operation
    }

    assert all(style.icon_symbol for style in styles.values())
    assert (
        styles[node_editor.Operation.ADD].header_colour
        == styles[node_editor.Operation.DOT].header_colour
    )
    assert (
        styles[node_editor.Operation.LOOK_AT].header_colour
        != styles[node_editor.Operation.ADD].header_colour
    )
    assert (
        styles[node_editor.Operation.QUATERNION_PRODUCT].header_colour
        != styles[node_editor.Operation.LOOK_AT].header_colour
    )
    assert (
        styles[node_editor.Operation.TRANSFORM_VERTICES].header_colour
        != styles[node_editor.Operation.QUATERNION_PRODUCT].header_colour
    )


def test_palette_and_menu_show_an_icon_for_every_catalogue_entry(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)

    catalogue_labels = {
        label
        for _title, entries in node_editor.NODE_CATALOGUE
        for label, _factory in entries
    }
    palette_buttons = {
        button.text(): button
        for button in window.palette.findChildren(QPushButton)
        if button.text() in catalogue_labels
    }
    menu_actions = {
        action.text(): action for action in window.view.node_menu.creation_actions
    }

    assert all(not button.icon().isNull() for button in palette_buttons.values())
    assert all(not action.icon().isNull() for action in menu_actions.values())
    window.close()


def test_operation_groupings_cover_every_operation() -> None:
    node_editor = _node_editor_module()
    grouped_operations = (
        set(node_editor.MATH_OPERATIONS)
        | set(node_editor.MAT4_OPERATIONS)
        | set(node_editor.QUATERNION_OPERATIONS)
        | set(node_editor.MESH_OPERATIONS)
    )
    assert grouped_operations == set(node_editor.Operation)


@pytest.mark.parametrize("key_target", ["view", "viewport"])
def test_pressing_tab_on_canvas_opens_node_creation_menu(
    application: QApplication,
    key_target: str,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    window.show()
    window.view.setFocus()

    target = window.view if key_target == "view" else window.view.viewport()
    QTest.keyClick(target, Qt.Key.Key_Tab)
    application.processEvents()

    assert window.view.node_menu.isVisible()
    assert window.view.node_menu.search_edit.hasFocus()
    window.view.node_menu.close()
    window.close()


def test_node_creation_menu_adds_selected_node_at_requested_position(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    window.show()
    viewport_position = QPoint(340, 210)
    expected_position = window.view.mapToScene(viewport_position)

    menu = window.view.open_node_menu(viewport_position)
    vec3_action = next(
        action for action in menu.creation_actions if action.text() == "Vec3"
    )
    vec3_action.trigger()
    application.processEvents()

    added_node = next(iter(window.canvas.nodes.values()))
    assert isinstance(added_node, node_editor.ValueNodeItem)
    assert added_node.math_type is node_editor.MathType.VEC3
    assert added_node.pos().x() == pytest.approx(expected_position.x())
    assert added_node.pos().y() == pytest.approx(expected_position.y())
    window.close()


def test_node_creation_menu_filters_entries_as_text_is_typed(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    window.show()
    menu = window.view.open_node_menu(QPoint(300, 200))

    menu.search_edit.setText("axis angle")
    application.processEvents()

    visible_labels = [
        action.text() for action in menu.creation_actions if action.isVisible()
    ]
    assert visible_labels == ["Quaternion from Axis Angle"]
    menu.close()
    window.close()


def test_enter_creates_first_filtered_node(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    window.show()
    menu = window.view.open_node_menu(QPoint(300, 200))
    menu.search_edit.setText("perspective")

    QTest.keyClick(menu.search_edit, Qt.Key.Key_Return)
    application.processEvents()

    added_node = next(iter(window.canvas.nodes.values()))
    assert isinstance(added_node, node_editor.GeneratorNodeItem)
    assert added_node.operation is node_editor.Operation.PERSPECTIVE
    assert not menu.isVisible()
    window.close()


@pytest.mark.parametrize("key", [Qt.Key.Key_Delete, Qt.Key.Key_Backspace])
def test_delete_key_removes_selected_node_and_its_wires(
    application: QApplication,
    key: Qt.Key,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)
    window.show()
    window.view.setFocus()
    multiply_node = next(
        node
        for node in window.canvas.nodes.values()
        if isinstance(node, node_editor.OperationNodeItem)
    )
    multiply_node.setSelected(True)

    QTest.keyClick(window.view, key)
    application.processEvents()

    assert multiply_node.node_id not in window.canvas.nodes
    assert len(window.canvas.connections) == 0
    assert window.canvas.modified is True
    window.close()


@pytest.mark.parametrize("key", [Qt.Key.Key_Delete, Qt.Key.Key_Backspace])
def test_delete_key_with_no_selection_does_not_mark_the_scene_modified(
    application: QApplication,
    key: Qt.Key,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)
    window.show()
    window.view.setFocus()
    assert window.canvas.modified is False

    QTest.keyClick(window.view, key)
    application.processEvents()

    assert window.canvas.modified is False
    window.close()


@pytest.mark.parametrize("key", [Qt.Key.Key_Delete, Qt.Key.Key_Backspace])
def test_delete_key_edits_a_focused_spin_box_instead_of_deleting_its_node(
    application: QApplication,
    key: Qt.Key,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)
    window.show()
    window.view.setFocus()
    value_node = next(
        node
        for node in window.canvas.nodes.values()
        if isinstance(node, node_editor.ValueNodeItem)
    )
    value_node.setSelected(True)
    value_node.spin_boxes[0].setFocus(Qt.FocusReason.MouseFocusReason)
    application.processEvents()

    QTest.keyClick(window.view, key)
    application.processEvents()

    assert value_node.node_id in window.canvas.nodes
    window.close()


def test_delete_key_removes_selected_connection_without_deleting_its_nodes(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)
    window.show()
    window.view.setFocus()
    connection = window.canvas.connections[0]
    connection.setSelected(True)

    QTest.keyClick(window.view, Qt.Key.Key_Delete)
    application.processEvents()

    assert connection not in window.canvas.connections
    assert len(window.canvas.nodes) == 4
    window.close()


def test_context_menu_target_resolves_a_port_to_its_owning_node(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)
    window.show()
    multiply_node = next(
        node
        for node in window.canvas.nodes.values()
        if isinstance(node, node_editor.OperationNodeItem)
    )
    viewport_position = window.view.mapFromScene(
        multiply_node.input_ports[0].scene_centre()
    )

    target = window.view._deletable_item_at(viewport_position)

    assert target is multiply_node
    window.close()


def test_context_menu_target_resolves_a_click_on_a_wire(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)
    window.show()
    connection = window.canvas.connections[0]
    midpoint = connection.path().pointAtPercent(0.5)
    viewport_position = window.view.mapFromScene(midpoint)

    target = window.view._deletable_item_at(viewport_position)

    assert target is connection
    window.close()


def test_right_click_menu_deletes_the_targeted_node(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)
    window.show()
    multiply_node = next(
        node
        for node in window.canvas.nodes.values()
        if isinstance(node, node_editor.OperationNodeItem)
    )
    viewport_position = window.view.mapFromScene(
        multiply_node.input_ports[0].scene_centre()
    )

    window.view._delete_item_at(viewport_position)
    application.processEvents()

    assert multiply_node.node_id not in window.canvas.nodes
    window.close()


def test_to_dict_and_from_dict_round_trip_the_example_graph(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)
    original_output = window.canvas.output_texts()

    data = window.canvas.to_dict()
    window.canvas.from_dict(data)
    application.processEvents()

    assert window.canvas.output_texts() == original_output
    assert len(window.canvas.nodes) == 4
    assert len(window.canvas.connections) == 3
    window.close()


def test_node_name_round_trips_through_graph_document(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    node = window.canvas.add_value_node(node_editor.MathType.VEC3)
    node.set_name("Camera Position")

    document = window.canvas.to_dict()
    saved_node = next(
        entry for entry in document["nodes"] if entry["id"] == node.node_id
    )
    window.canvas.from_dict(document)
    restored_node = next(iter(window.canvas.nodes.values()))

    assert saved_node.get("name") == "Camera Position"
    assert restored_node.title == "Camera Position"
    window.close()


def test_serialized_graph_includes_its_schema_version(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)

    document = window.canvas.to_dict()

    assert document["schema_version"] == 1
    window.close()


def test_from_dict_does_not_replace_the_graph_when_validation_fails(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)
    before = window.canvas.to_dict()
    invalid_document = {
        "nodes": [{"id": "node-1", "x": 0.0, "y": 0.0, "kind": "unknown"}],
        "connections": [],
    }

    with pytest.raises(node_editor.GraphError, match="Unknown node kind"):
        window.canvas.from_dict(invalid_document)

    assert window.canvas.to_dict() == before
    window.close()


def test_from_dict_rejects_an_unsupported_schema_version(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)
    document = window.canvas.to_dict()
    document["schema_version"] = 99

    with pytest.raises(node_editor.GraphError, match="schema version 99"):
        window.canvas.from_dict(document)

    assert window.canvas.output_texts() == ["Vec3(4, 10, 18)"]
    window.close()


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


def test_transform_rotation_order_round_trips_through_the_document(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    transform_node = window.canvas.add_generator_node(node_editor.Operation.TRANSFORM)
    assert transform_node.rotation_order_combo is not None
    transform_node.rotation_order_combo.setCurrentText("zyx")

    document = window.canvas.to_dict()
    saved_node = next(
        entry for entry in document["nodes"] if entry["id"] == transform_node.node_id
    )
    window.canvas.from_dict(document)

    loaded_node = next(iter(window.canvas.nodes.values()))
    assert isinstance(loaded_node, node_editor.GeneratorNodeItem)
    assert saved_node["rotation_order"] == "zyx"
    assert loaded_node.rotation_order_combo is not None
    assert loaded_node.rotation_order_combo.currentText() == "zyx"
    assert window.canvas.graph.generator_rotation_order(loaded_node.node_id) == "zyx"
    window.close()


def test_serialization_preserves_value_precision_beyond_the_spin_box_display(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    precise_value = 1.23456789
    node = window.canvas.add_value_node(
        node_editor.MathType.FLOAT, components=(precise_value,)
    )

    document = window.canvas.to_dict()
    saved_node = next(
        entry for entry in document["nodes"] if entry["id"] == node.node_id
    )

    assert saved_node["components"] == [precise_value]
    window.close()


def test_serialization_preserves_generator_parameter_precision(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    precise_value = 1.23456789
    node = window.canvas.add_generator_node(
        node_editor.Operation.MAT4_TRANSLATE,
        parameters=((precise_value,), (2.0,), (3.0,)),
    )

    document = window.canvas.to_dict()
    saved_node = next(
        entry for entry in document["nodes"] if entry["id"] == node.node_id
    )

    assert saved_node["parameters"] == [[precise_value], [2.0], [3.0]]
    window.close()


@pytest.mark.parametrize("node_kind", ["value", "generator"])
def test_numeric_editors_support_precise_large_values(
    application: QApplication, node_kind: str
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    if node_kind == "value":
        node = window.canvas.add_value_node(
            node_editor.MathType.FLOAT, components=(1.23456789,)
        )
        spin_box = node.spin_boxes[0]
    else:
        node = window.canvas.add_generator_node(
            node_editor.Operation.MAT4_TRANSLATE,
            parameters=((1.23456789,), (2.0,), (3.0,)),
        )
        spin_box = node.spin_box_rows[0][0]

    assert spin_box.value() == pytest.approx(1.23456789, abs=1e-7)
    spin_box.setValue(1.0e12)
    assert spin_box.value() == pytest.approx(1.0e12)
    window.close()


def test_generator_tables_agree_on_operations_and_arities() -> None:
    math_graph = import_module("math_graph")
    from graphics_items import GENERATOR_DEFAULTS

    assert set(math_graph.OPERATION_PARAMETER_TYPES) == set(
        math_graph.GENERATOR_OPERATIONS
    )
    assert set(math_graph.GENERATOR_OUTPUT_TYPE) == set(math_graph.GENERATOR_OPERATIONS)
    assert set(GENERATOR_DEFAULTS) == set(math_graph.GENERATOR_OPERATIONS)

    for operation in math_graph.GENERATOR_OPERATIONS:
        parameter_types = math_graph.OPERATION_PARAMETER_TYPES[operation]
        defaults = GENERATOR_DEFAULTS[operation]
        assert len(parameter_types) == len(defaults), operation
        for parameter_type, default_components in zip(
            parameter_types, defaults, strict=True
        ):
            expected_count = math_graph.TYPE_SHAPES[parameter_type][1]
            assert len(default_components) == expected_count, operation


def test_save_to_file_and_load_from_file_round_trip(
    application: QApplication,
    tmp_path: Path,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)
    file_path = tmp_path / "graph.json"

    window.canvas.save_to_file(file_path)
    window.canvas.load_from_file(file_path)
    application.processEvents()

    assert window.canvas.output_texts() == ["Vec3(4, 10, 18)"]
    window.close()


def test_failed_atomic_save_preserves_the_existing_file(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_editor = _node_editor_module()
    canvas_module = import_module("canvas")
    window = node_editor.MathNodeWindow(load_example=True)
    file_path = tmp_path / "graph.json"
    file_path.write_text("original document")

    class RejectingSaveFile:
        def __init__(self, path: str) -> None:
            self.path = path

        def open(self, _mode: object) -> bool:
            return True

        def write(self, payload: bytes) -> int:
            return len(payload)

        def commit(self) -> bool:
            return False

        def errorString(self) -> str:
            return "simulated commit failure"

    monkeypatch.setattr(canvas_module, "QSaveFile", RejectingSaveFile, raising=False)

    with pytest.raises(OSError, match="simulated commit failure"):
        window.canvas.save_to_file(file_path)

    assert file_path.read_text() == "original document"
    window.close()


def test_save_action_writes_a_json_file(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=True, settings=_isolated_settings(tmp_path)
    )
    file_path = tmp_path / "graph.json"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(file_path), ""),
    )

    window.action_save.trigger()
    application.processEvents()

    assert file_path.exists()
    window.close()


def test_open_action_replaces_the_current_graph(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=True, settings=_isolated_settings(tmp_path)
    )
    file_path = tmp_path / "graph.json"
    window.canvas.save_to_file(file_path)
    window.canvas.clear_graph()
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(file_path), ""),
    )

    window.action_open.trigger()
    application.processEvents()

    assert window.canvas.output_texts() == ["Vec3(4, 10, 18)"]
    window.close()


def test_open_action_reports_a_malformed_file_instead_of_crashing(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=True, settings=_isolated_settings(tmp_path)
    )
    file_path = tmp_path / "broken.json"
    file_path.write_text("not valid json")
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(file_path), ""),
    )
    warnings: list[object] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    window.action_open.trigger()
    application.processEvents()

    assert len(warnings) == 1
    assert window.canvas.output_texts() == ["Vec3(4, 10, 18)"]
    window.close()


def test_open_action_restores_the_previous_graph_after_a_schema_error(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema-invalid (but JSON-valid) file must not wreck the open graph.

    ``MathNodeScene.from_dict`` clears the graph before rebuilding it, so a
    file that's valid JSON but fails schema validation (an unknown node
    ``kind``, here) leaves the canvas half-built rather than untouched. Left
    unhandled, the user would see a wrecked canvas under a clean (non-dirty)
    title bar, and a following ``Ctrl+S`` would silently overwrite
    ``good_file`` on disk with that wreckage.
    """
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=True, settings=_isolated_settings(tmp_path)
    )
    good_file = tmp_path / "good.json"
    window.canvas.save_to_file(good_file)
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(good_file), ""),
    )
    window.action_open.trigger()
    application.processEvents()
    assert window.current_file == good_file

    broken_file = tmp_path / "bad_schema.json"
    broken_file.write_text(
        json.dumps(
            {
                "nodes": [{"id": "n1", "x": 0.0, "y": 0.0, "kind": "not_a_real_kind"}],
                "connections": [],
            }
        )
    )
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(broken_file), ""),
    )
    warnings: list[object] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    window.action_open.trigger()
    application.processEvents()

    assert len(warnings) == 1
    assert window.canvas.output_texts() == ["Vec3(4, 10, 18)"]
    assert window.current_file == good_file
    window.close()


def test_failed_open_preserves_a_dirty_document(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=False, settings=_isolated_settings(tmp_path)
    )
    window.canvas.add_value_node(node_editor.MathType.VEC3)
    current_file = tmp_path / "current.json"
    window.current_file = current_file
    broken_file = tmp_path / "broken.json"
    broken_file.write_text("not valid json")
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)

    opened = window._open_path(broken_file)

    assert opened is False
    assert len(window.canvas.nodes) == 1
    assert window.canvas.modified is True
    assert window.current_file == current_file
    window.close()


def test_file_menu_has_the_expected_actions_and_shortcuts(
    application: QApplication, tmp_path: Path
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=False, settings=_isolated_settings(tmp_path)
    )

    assert window.action_new.shortcut() == QKeySequence.StandardKey.New
    assert window.action_open.shortcut() == QKeySequence.StandardKey.Open
    assert window.action_save.shortcut() == QKeySequence.StandardKey.Save
    assert window.action_save_as.shortcut() == QKeySequence.StandardKey.SaveAs
    window.close()


def test_new_clears_the_graph_and_forgets_the_current_file(
    application: QApplication, tmp_path: Path
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=True, settings=_isolated_settings(tmp_path)
    )

    window.action_new.trigger()

    assert window.canvas.nodes == {}
    assert window.current_file is None
    window.close()


def test_save_as_writes_a_file_and_becomes_the_current_file(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=True, settings=_isolated_settings(tmp_path)
    )
    file_path = tmp_path / "graph.json"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(file_path), "")
    )

    window.action_save_as.trigger()

    assert file_path.exists()
    assert window.current_file == file_path
    window.close()


def test_save_writes_to_the_current_file_without_prompting(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=True, settings=_isolated_settings(tmp_path)
    )
    window.current_file = tmp_path / "graph.json"

    def _fail_if_called(*_args: object, **_kwargs: object) -> tuple[str, str]:
        raise AssertionError("Save must not prompt when a current file is set")

    monkeypatch.setattr(QFileDialog, "getSaveFileName", _fail_if_called)

    window.action_save.trigger()

    assert window.current_file.exists()
    window.close()


def test_save_without_a_current_file_behaves_like_save_as(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=True, settings=_isolated_settings(tmp_path)
    )
    file_path = tmp_path / "graph.json"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(file_path), "")
    )

    window.action_save.trigger()

    assert window.current_file == file_path
    window.close()


def test_open_replaces_the_graph_and_updates_recent_file_setting(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_editor = _node_editor_module()
    settings = _isolated_settings(tmp_path)
    window = node_editor.MathNodeWindow(load_example=True, settings=settings)
    file_path = tmp_path / "graph.json"
    window.canvas.save_to_file(file_path)
    window.action_new.trigger()
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(file_path), "")
    )

    window.action_open.trigger()
    application.processEvents()

    assert window.canvas.output_texts() == ["Vec3(4, 10, 18)"]
    assert settings.value("recentFile") == str(file_path)
    window.close()


def test_startup_reopens_the_recent_file_when_present(
    application: QApplication, tmp_path: Path
) -> None:
    node_editor = _node_editor_module()
    settings = _isolated_settings(tmp_path)
    seed_window = node_editor.MathNodeWindow(load_example=False, settings=settings)
    seed_window.canvas.add_value_node(node_editor.MathType.VEC3)
    seed_window.canvas.add_output_node()
    file_path = tmp_path / "graph.json"
    seed_window.canvas.save_to_file(file_path)
    seed_window.close()
    settings.setValue("recentFile", str(file_path))

    window = node_editor.MathNodeWindow(load_example=True, settings=settings)
    application.processEvents()

    assert window.current_file == file_path
    assert len(window.canvas.nodes) == 2
    window.close()


def test_startup_falls_back_to_the_bundled_demo_without_a_recent_file(
    application: QApplication, tmp_path: Path
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=True, settings=_isolated_settings(tmp_path)
    )
    application.processEvents()

    assert window.current_file is None
    assert window.canvas.output_texts() == ["Vec3(4, 10, 18)"]
    window.close()


def test_startup_falls_back_when_the_recent_file_is_missing(
    application: QApplication, tmp_path: Path
) -> None:
    node_editor = _node_editor_module()
    settings = _isolated_settings(tmp_path)
    settings.setValue("recentFile", str(tmp_path / "does-not-exist.json"))

    window = node_editor.MathNodeWindow(load_example=True, settings=settings)
    application.processEvents()

    assert window.current_file is None
    assert window.canvas.output_texts() == ["Vec3(4, 10, 18)"]
    window.close()


def test_window_geometry_round_trips_through_settings(
    application: QApplication, tmp_path: Path
) -> None:
    # 700x480 rather than a larger size: the offscreen QPA platform used for
    # these tests has a fixed 800x800 virtual screen, and QWidget.restoreGeometry
    # clamps a restored size down to the available screen area, so anything
    # wider/taller than that can never round-trip exactly under this platform.
    node_editor = _node_editor_module()
    settings = _isolated_settings(tmp_path)
    first = node_editor.MathNodeWindow(load_example=False, settings=settings)
    first.resize(700, 480)
    first.show()
    application.processEvents()
    first.close()

    second = node_editor.MathNodeWindow(load_example=False, settings=settings)

    assert second.size().width() == 700
    assert second.size().height() == 480
    second.close()


def test_title_shows_the_current_file_name_and_dirty_marker(
    application: QApplication, tmp_path: Path
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=False, settings=_isolated_settings(tmp_path)
    )
    assert window.windowTitle().endswith("Untitled")

    window.canvas.add_value_node(node_editor.MathType.VEC3)
    application.processEvents()

    assert window.windowTitle().endswith("Untitled*")
    window.close()


def test_mesh_pipeline_example_loads_without_errors(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    example_path = Path(__file__).parent.parent / "examples" / "mesh_pipeline_demo.json"

    window.canvas.load_from_file(example_path)
    application.processEvents()

    mesh_viewer = next(
        node
        for node in window.canvas.nodes.values()
        if isinstance(node, node_editor.MeshViewerNodeItem)
    )
    obj_loader = next(
        node
        for node in window.canvas.nodes.values()
        if isinstance(node, node_editor.ObjLoaderNodeItem)
    )
    assert mesh_viewer.status_text_item.toPlainText() == ""
    assert obj_loader.status_text_item.toPlainText() == "cube.obj: 8 verts, 12 faces"
    assert mesh_viewer.render_state.mesh_inputs is not None
    assert len(mesh_viewer.render_state.mesh_inputs.vertices.values) == 8
    window.close()


def test_diffuse_mesh_viewer_rejects_an_empty_normal_array(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    graph_module = import_module("math_graph")
    window = node_editor.MathNodeWindow(load_example=False)
    loader = window.canvas.add_obj_loader_node()
    vertices_id, faces_id, _uvs_id, normals_id = loader.array_node_ids
    window.canvas.graph.set_literal(
        vertices_id,
        graph_module.VertexArray(
            (
                Vec3(0.0, 0.0, 0.0),
                Vec3(1.0, 0.0, 0.0),
                Vec3(0.0, 1.0, 0.0),
            )
        ),
    )
    window.canvas.graph.set_literal(
        faces_id,
        graph_module.FaceArray((((0, None, None), (1, None, None), (2, None, None)),)),
    )
    window.canvas.graph.set_literal(normals_id, graph_module.NormalArray(()))
    viewer = window.canvas.add_mesh_viewer_node(shading_mode="Diffuse")
    window.canvas.connect_ports(loader.output_ports[0], viewer.input_ports[0])
    window.canvas.connect_ports(loader.output_ports[1], viewer.input_ports[1])
    window.canvas.connect_ports(loader.output_ports[3], viewer.input_ports[3])

    assert "needs input Normals" in viewer.status_text_item.toPlainText()
    window.close()


def test_mesh_display_controls_do_not_rebuild_geometry(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    example_path = Path(__file__).parent.parent / "examples" / "mesh_pipeline_demo.json"
    window.canvas.load_from_file(example_path)
    viewer = next(
        node
        for node in window.canvas.nodes.values()
        if isinstance(node, node_editor.MeshViewerNodeItem)
    )
    geometry_version = viewer.render_state.version

    viewer.wireframe_check.toggle()
    viewer.shading_combo.setCurrentText("Diffuse")
    application.processEvents()

    assert viewer.render_state.version == geometry_version
    window.close()


def test_closed_mesh_popup_can_be_opened_again(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_editor = _node_editor_module()
    graphics_items = import_module("graphics_items")

    class Popup:
        def __init__(self, *_args: object) -> None:
            self.visible = False
            self.show_calls = 0
            self.activation_calls = 0

        def show(self) -> None:
            self.visible = True
            self.show_calls += 1

        def close(self) -> None:
            self.visible = False

        def isVisible(self) -> bool:
            return self.visible

        def requestActivate(self) -> None:
            self.activation_calls += 1

        def update(self) -> None:
            pass

    monkeypatch.setattr(graphics_items, "MeshPopupWindow", Popup)
    window = node_editor.MathNodeWindow(load_example=False)
    viewer = window.canvas.add_mesh_viewer_node()

    viewer._pop_out()
    popup = viewer._popup
    assert popup is not None
    popup.close()
    viewer._pop_out()

    assert popup.show_calls == 2
    window.close()


def test_clearing_the_graph_closes_mesh_popups(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    viewer = window.canvas.add_mesh_viewer_node()

    class Popup:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    popup = Popup()
    viewer._popup = popup

    window.canvas.clear_graph()

    assert popup.close_calls == 1
    window.close()


def test_default_example_file_loads_the_vec3_multiply_result(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    example_path = Path(__file__).parent.parent / "examples" / "vec3_multiply_demo.json"

    window.canvas.load_from_file(example_path)
    application.processEvents()

    assert window.canvas.output_texts() == ["Vec3(4, 10, 18)"]
    window.close()


def test_mvp_example_loads_and_evaluates_a_single_matrix(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    example_path = Path(__file__).parent.parent / "examples" / "mvp_demo.json"

    window.canvas.load_from_file(example_path)
    application.processEvents()

    assert len(window.canvas.output_texts()) == 1
    assert window.canvas.output_texts()[0].startswith("Mat4")
    transform_node = next(
        node
        for node in window.canvas.nodes.values()
        if isinstance(node, node_editor.GeneratorNodeItem)
        and node.operation is node_editor.Operation.TRANSFORM
    )
    assert transform_node.parameter_names == ("Position", "Rotation", "Scale")
    window.close()


def test_mvp_mesh_example_applies_the_model_transform_to_a_displayed_mesh(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    example_path = Path(__file__).parent.parent / "examples" / "mvp_mesh_demo.json"

    window.canvas.load_from_file(example_path)
    application.processEvents()

    mesh_viewer = next(
        node
        for node in window.canvas.nodes.values()
        if isinstance(node, node_editor.MeshViewerNodeItem)
    )
    assert mesh_viewer.status_text_item.toPlainText() == ""
    assert mesh_viewer.render_state.mesh_inputs is not None
    vertices = mesh_viewer.render_state.mesh_inputs.vertices.values
    assert len(vertices) == 8
    # The cube is Model-transformed (position, rotation, scale all non-identity),
    # so its rendered vertices should differ from the untouched loader output.
    unit_cube_vertex = max(vertices, key=lambda v: v.x**2 + v.y**2 + v.z**2)
    assert unit_cube_vertex.to_list() != pytest.approx([1.0, 1.0, 1.0])
    window.close()


@pytest.mark.parametrize(
    ("filename", "expected_outputs"),
    [
        (
            "vector_arithmetic_demo.json",
            [
                "Vec3(2.5, 1, -4)",
                "Vec3(3, 4, 4)",
                "41",
                "Vec3(0.46852, 0.6247, 0.6247)",
            ],
        ),
        (
            "triangle_normal_demo.json",
            ["Vec3(0, 6, 0)", "Vec3(0, 1, 0)"],
        ),
        ("lambert_diffuse_demo.json", ["0.80178"]),
        (
            "mat2_rotation_demo.json",
            ["Vec2(-1, 2)", "Vec2(2, 1)"],
        ),
        (
            "homogeneous_coordinates_demo.json",
            ["Vec4(6, 0, 4, 1)", "Vec4(1, 2, 3, 0)"],
        ),
        (
            "transform_order_demo.json",
            ["Vec4(12, 0, 0, 1)", "Vec4(7, 0, 0, 1)"],
        ),
        (
            "normal_matrix_demo.json",
            [
                "Vec3(0.89443, 0.44721, 0)",
                "Vec3(0.44721, 0.89443, 0)",
            ],
        ),
        (
            "quaternion_rotation_demo.json",
            [
                "Quaternion(0.70711, 0, 0.70711, 0)",
                "Vec3(0, 0, -1)",
            ],
        ),
        (
            "quaternion_slerp_demo.json",
            [
                "Quaternion(0.70711, 0, 0.70711, 0)",
                "Vec3(0, 0, -1)",
            ],
        ),
    ],
)
def test_teaching_examples_load_and_evaluate_expected_results(
    application: QApplication,
    filename: str,
    expected_outputs: list[str],
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    example_path = Path(__file__).parent.parent / "examples" / filename

    window.canvas.load_from_file(example_path)
    application.processEvents()

    assert window.canvas.output_texts() == expected_outputs
    window.close()


def test_projection_comparison_example_evaluates_three_mat4_values(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    example_path = (
        Path(__file__).parent.parent / "examples" / "projection_comparison_demo.json"
    )

    window.canvas.load_from_file(example_path)
    application.processEvents()

    assert len(window.canvas.output_texts()) == 3
    assert all(text.startswith("Mat4") for text in window.canvas.output_texts())
    window.close()


@pytest.mark.parametrize(
    "filename",
    [
        "vector_arithmetic_demo.json",
        "triangle_normal_demo.json",
        "lambert_diffuse_demo.json",
        "mat2_rotation_demo.json",
        "homogeneous_coordinates_demo.json",
        "transform_order_demo.json",
        "normal_matrix_demo.json",
        "quaternion_rotation_demo.json",
        "quaternion_slerp_demo.json",
        "projection_comparison_demo.json",
    ],
)
def test_teaching_example_nodes_do_not_overlap(
    application: QApplication,
    filename: str,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    example_path = Path(__file__).parent.parent / "examples" / filename

    window.canvas.load_from_file(example_path)
    application.processEvents()
    visual_nodes = list(
        {id(node): node for node in window.canvas.nodes.values()}.values()
    )

    overlaps = []
    for left, right in combinations(visual_nodes, 2):
        intersection = left.sceneBoundingRect().intersected(right.sceneBoundingRect())
        if intersection.width() > 1.0 and intersection.height() > 1.0:
            overlaps.append((left.title, right.title))

    assert overlaps == []
    window.close()


def test_wheel_zoom_step_is_gentle(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)

    window.view.wheelEvent(_wheel_event(120))

    assert window.view.transform().m11() == pytest.approx(
        node_editor.MathNodeView.ZOOM_STEP
    )
    window.close()


def test_wheel_zoom_is_clamped_to_the_maximum(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)

    for _ in range(200):
        window.view.wheelEvent(_wheel_event(120))

    assert window.view.transform().m11() == pytest.approx(
        node_editor.MathNodeView.MAX_ZOOM
    )
    window.close()


def test_wheel_zoom_is_clamped_to_the_minimum(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)

    for _ in range(200):
        window.view.wheelEvent(_wheel_event(-120))

    assert window.view.transform().m11() == pytest.approx(
        node_editor.MathNodeView.MIN_ZOOM
    )
    window.close()


def test_wheel_over_a_spin_box_edits_it_instead_of_zooming(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    value_node = window.canvas.add_value_node(node_editor.MathType.FLOAT)
    application.processEvents()
    spin_box = value_node.spin_boxes[0]
    starting_value = spin_box.value()
    scene_position = value_node.proxy.mapToScene(QPointF(spin_box.geometry().center()))
    view_position = QPointF(window.view.mapFromScene(scene_position))

    window.view.wheelEvent(_wheel_event(120, view_position))

    assert spin_box.value() == pytest.approx(starting_value + spin_box.singleStep())
    assert window.view.transform().m11() == pytest.approx(1.0)
    window.close()


def test_frame_all_fits_every_node_within_the_viewport(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)
    window.resize(900, 600)
    window.show()
    application.processEvents()
    window.view.resetTransform()
    window.view.centerOn(2000.0, 2000.0)

    window.view.frame_all()
    application.processEvents()

    visible_rect = window.view.mapToScene(window.view.viewport().rect()).boundingRect()
    for node in window.canvas.nodes.values():
        assert visible_rect.contains(node.sceneBoundingRect())
    window.close()


def test_startup_frames_every_loaded_node(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)
    window.resize(900, 600)
    window.show()
    application.processEvents()

    visible_rect = window.view.mapToScene(window.view.viewport().rect()).boundingRect()

    for node in window.canvas.nodes.values():
        assert visible_rect.contains(node.sceneBoundingRect())
    window.close()


def test_frame_all_does_nothing_on_an_empty_graph(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    window.show()
    application.processEvents()
    before = window.view.transform()

    window.view.frame_all()

    assert window.view.transform() == before
    window.close()


def test_h_key_triggers_frame_all(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)
    window.show()
    window.view.setFocus()
    calls: list[bool] = []
    window.view.frame_all = lambda: calls.append(True)

    QTest.keyClick(window.view, Qt.Key.Key_H)
    application.processEvents()

    assert calls == [True]
    window.close()


def test_frame_all_button_calls_view_frame_all(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)
    calls: list[bool] = []
    window.view.frame_all = lambda: calls.append(True)
    frame_button = next(
        child
        for child in window.palette.findChildren(QPushButton)
        if child.text() == "Frame All"
    )

    frame_button.click()

    assert calls == [True]
    window.close()


def test_adding_a_node_marks_the_scene_modified(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)

    window.canvas.add_value_node(node_editor.MathType.VEC3)

    assert window.canvas.modified is True
    window.close()


@pytest.mark.parametrize(
    "node_kind",
    ["value", "operation", "generator", "output", "obj_loader", "mesh_viewer"],
)
def test_adding_a_node_at_the_scene_origin_marks_the_scene_modified(
    application: QApplication,
    node_kind: str,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    origin = QPointF(0.0, 0.0)

    if node_kind == "value":
        window.canvas.add_value_node(node_editor.MathType.VEC3, origin)
    elif node_kind == "operation":
        window.canvas.add_operation_node(node_editor.Operation.ADD, origin)
    elif node_kind == "generator":
        window.canvas.add_generator_node(node_editor.Operation.PERSPECTIVE, origin)
    elif node_kind == "output":
        window.canvas.add_output_node(origin)
    elif node_kind == "obj_loader":
        window.canvas.add_obj_loader_node(origin)
    else:
        window.canvas.add_mesh_viewer_node(origin)

    assert window.canvas.modified is True
    window.close()


def test_editing_a_value_marks_the_scene_modified(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    node = window.canvas.add_value_node(node_editor.MathType.VEC3)
    window.canvas.modified = False

    node.spin_boxes[0].setValue(9.0)

    assert window.canvas.modified is True
    window.close()


def test_dragging_a_node_marks_the_scene_modified(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    node = window.canvas.add_value_node(node_editor.MathType.VEC3)
    window.canvas.modified = False

    node.setPos(QPointF(50.0, 50.0))

    assert window.canvas.modified is True
    window.close()


def test_loading_a_file_leaves_the_scene_unmodified(
    application: QApplication, tmp_path: Path
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    window.canvas.add_value_node(node_editor.MathType.VEC3)
    file_path = tmp_path / "graph.json"
    window.canvas.save_to_file(file_path)

    window.canvas.load_from_file(file_path)

    assert window.canvas.modified is False
    window.close()


def test_clear_graph_resets_modified(application: QApplication) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    window.canvas.add_value_node(node_editor.MathType.VEC3)
    assert window.canvas.modified is True

    window.canvas.clear_graph()

    assert window.canvas.modified is False
    window.close()


def test_modified_changed_signal_fires_once_on_transition(
    application: QApplication,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=False)
    seen: list[bool] = []
    window.canvas.modifiedChanged.connect(seen.append)

    window.canvas.add_value_node(node_editor.MathType.VEC3)
    window.canvas.add_value_node(node_editor.MathType.VEC3)

    assert seen == [True]
    window.close()


def test_saving_emits_the_clean_document_state(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=False, settings=_isolated_settings(tmp_path)
    )
    window.canvas.add_value_node(node_editor.MathType.VEC3)
    seen: list[bool] = []
    window.canvas.modifiedChanged.connect(seen.append)
    file_path = tmp_path / "graph.json"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(file_path), ""),
    )

    window.action_save.trigger()

    assert seen == [False]
    window.close()


def test_new_prompts_and_cancels_when_discarding_is_declined(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=False, settings=_isolated_settings(tmp_path)
    )
    window.canvas.add_value_node(node_editor.MathType.VEC3)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Cancel,
    )

    window.action_new.trigger()

    assert len(window.canvas.nodes) == 1
    window.close()


def test_new_discards_without_saving_when_chosen(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=False, settings=_isolated_settings(tmp_path)
    )
    window.canvas.add_value_node(node_editor.MathType.VEC3)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Discard,
    )

    window.action_new.trigger()

    assert window.canvas.nodes == {}
    window.close()


def test_new_saves_first_when_save_is_chosen(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=False, settings=_isolated_settings(tmp_path)
    )
    window.canvas.add_value_node(node_editor.MathType.VEC3)
    file_path = tmp_path / "graph.json"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(file_path), "")
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Save,
    )

    window.action_new.trigger()

    assert file_path.exists()
    assert window.canvas.nodes == {}
    window.close()


def test_new_does_not_prompt_when_the_graph_is_unmodified(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=False, settings=_isolated_settings(tmp_path)
    )

    def _fail_if_called(
        *_args: object, **_kwargs: object
    ) -> QMessageBox.StandardButton:
        raise AssertionError("Must not prompt on a clean graph")

    monkeypatch.setattr(QMessageBox, "question", _fail_if_called)

    window.action_new.trigger()

    assert window.canvas.nodes == {}
    window.close()


def test_open_prompts_before_replacing_a_modified_graph(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=False, settings=_isolated_settings(tmp_path)
    )
    window.canvas.add_value_node(node_editor.MathType.VEC3)
    other_path = tmp_path / "other.json"
    seed_window = node_editor.MathNodeWindow(
        load_example=True, settings=_isolated_settings(tmp_path)
    )
    seed_window.canvas.save_to_file(other_path)
    seed_window.close()
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(other_path), "")
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Cancel,
    )

    window.action_open.trigger()

    assert len(window.canvas.nodes) == 1
    window.close()


def test_close_event_ignored_when_discard_is_cancelled(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(
        load_example=False, settings=_isolated_settings(tmp_path)
    )
    window.canvas.add_value_node(node_editor.MathType.VEC3)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Cancel,
    )
    close_event = QCloseEvent()

    window.closeEvent(close_event)

    assert close_event.isAccepted() is False
    window.canvas.clear_graph()
    window.close()
