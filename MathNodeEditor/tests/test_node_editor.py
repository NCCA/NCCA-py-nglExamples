"""Qt integration tests for the maths node editor."""

import os
from importlib import import_module
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QFont, QFontMetrics, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QMessageBox,
    QPushButton,
)


def _wheel_event(delta_y: int) -> QWheelEvent:
    """Build a synthetic wheel event scrolling up (positive) or down."""
    return QWheelEvent(
        QPointF(0.0, 0.0),
        QPointF(0.0, 0.0),
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


def test_save_graph_button_writes_a_json_file(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)
    file_path = tmp_path / "graph.json"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(file_path), ""),
    )
    save_button = next(
        child
        for child in window.palette.findChildren(QPushButton)
        if child.text() == "Save graph..."
    )

    save_button.click()
    application.processEvents()

    assert file_path.exists()
    window.close()


def test_load_graph_button_replaces_the_current_graph(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)
    file_path = tmp_path / "graph.json"
    window.canvas.save_to_file(file_path)
    window.canvas.clear_graph()
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(file_path), ""),
    )
    load_button = next(
        child
        for child in window.palette.findChildren(QPushButton)
        if child.text() == "Load graph..."
    )

    load_button.click()
    application.processEvents()

    assert window.canvas.output_texts() == ["Vec3(4, 10, 18)"]
    window.close()


def test_load_graph_button_reports_a_malformed_file_instead_of_crashing(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_editor = _node_editor_module()
    window = node_editor.MathNodeWindow(load_example=True)
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
    load_button = next(
        child
        for child in window.palette.findChildren(QPushButton)
        if child.text() == "Load graph..."
    )

    load_button.click()
    application.processEvents()

    assert len(warnings) == 1
    assert window.canvas.output_texts() == ["Vec3(4, 10, 18)"]
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
