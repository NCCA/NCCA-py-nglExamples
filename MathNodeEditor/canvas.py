"""Scene and view for the PyNGL maths node editor canvas."""

from __future__ import annotations

import json
from pathlib import Path

from graph_document import SCHEMA_VERSION, validate_document
from graphics_items import (
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
from math_graph import (
    FaceArray,
    GraphError,
    MathGraph,
    MathType,
    NormalArray,
    Operation,
    UVArray,
    VertexArray,
    arrays_from_obj,
    format_value,
)
from mesh_view import SHADING_DIFFUSE, SHADING_SOLID
from ncca.ngl import (
    Obj,
    ObjParseFaceError,
    ObjParseNormalError,
    ObjParseUVError,
    ObjParseVertexError,
    Vec2,
    Vec3,
)
from palette import NodeCreationMenu
from PySide6.QtCore import (
    QEvent,
    QIODevice,
    QLineF,
    QPoint,
    QPointF,
    QRectF,
    QSaveFile,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QContextMenuEvent,
    QCursor,
    QKeyEvent,
    QPainter,
    QPen,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QFileDialog,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsProxyWidget,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QGraphicsView,
    QMenu,
    QWidget,
)

_OBJ_PARSE_ERRORS = (
    ObjParseVertexError,
    ObjParseNormalError,
    ObjParseUVError,
    ObjParseFaceError,
)

DEFAULT_EXAMPLE_PATH = (
    Path(__file__).resolve().parent / "examples" / "vec3_multiply_demo.json"
)


class MathNodeScene(QGraphicsScene):
    """Graphics scene which keeps visible wires and the maths graph in sync."""

    modifiedChanged = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty node graph canvas."""
        super().__init__(parent)
        self.setSceneRect(-1800.0, -1200.0, 3600.0, 2400.0)
        self.graph = MathGraph()
        self.nodes: dict[str, BaseNodeItem] = {}
        self.connections: list[ConnectionItem] = []
        self._drag_source: PortItem | None = None
        self._preview_connection: QGraphicsPathItem | None = None
        self._insertion_index = 0
        self.modified = False
        self._loading = False

    def set_modified(self, modified: bool) -> None:
        """Set the document state and notify listeners when it changes."""
        if (modified and self._loading) or self.modified == modified:
            return
        self.modified = modified
        self.modifiedChanged.emit(modified)

    def mark_modified(self) -> None:
        """Flag the graph as having unsaved changes."""
        self.set_modified(True)

    def mark_clean(self) -> None:
        """Flag the graph as matching the document on disk."""
        self.set_modified(False)

    def _next_position(self) -> QPointF:
        """Return a slightly offset position for the next palette node."""
        offset = float((self._insertion_index % 8) * 34)
        self._insertion_index += 1
        return QPointF(-300.0 + offset, -190.0 + offset)

    def add_value_node(
        self,
        math_type: MathType,
        position: QPointF | None = None,
        components: tuple[float, ...] | None = None,
    ) -> ValueNodeItem:
        """Add an editable maths value node to the canvas."""
        node_components = (
            components if components is not None else default_components(math_type)
        )
        node_id = self.graph.add_value(math_type, node_components)
        node = ValueNodeItem(
            node_id,
            math_type,
            node_components,
            lambda values, value_node_id=node_id: self._value_changed(
                value_node_id, values
            ),
        )
        self.addItem(node)
        node.setPos(position if position is not None else self._next_position())
        self.nodes[node_id] = node
        self.mark_modified()
        return node

    def add_operation_node(
        self,
        operation: Operation,
        position: QPointF | None = None,
    ) -> OperationNodeItem:
        """Add an operation node to the canvas."""
        node_id = self.graph.add_operation(operation)
        node = OperationNodeItem(node_id, operation)
        self.addItem(node)
        node.setPos(position if position is not None else self._next_position())
        self.nodes[node_id] = node
        self.mark_modified()
        return node

    def add_generator_node(
        self,
        operation: Operation,
        position: QPointF | None = None,
        parameters: tuple[tuple[float, ...], ...] | None = None,
        rotation_order: str | None = None,
    ) -> GeneratorNodeItem:
        """Add a parameter node (Look At, Perspective, ...) to the canvas."""
        node_parameters = (
            parameters
            if parameters is not None
            else default_generator_parameters(operation)
        )
        node_id = self.graph.add_generator(
            operation, node_parameters, rotation_order=rotation_order
        )
        node = GeneratorNodeItem(
            node_id,
            operation,
            node_parameters,
            lambda parameter_index,
            values,
            generator_node_id=node_id: self._generator_changed(
                generator_node_id, parameter_index, values
            ),
            rotation_order=(
                self.graph.generator_rotation_order(node_id)
                if operation is Operation.TRANSFORM
                else None
            ),
            on_rotation_order_change=lambda order,
            generator_node_id=node_id: self._generator_rotation_order_changed(
                generator_node_id, order
            ),
        )
        self.addItem(node)
        node.setPos(position if position is not None else self._next_position())
        self.nodes[node_id] = node
        self.mark_modified()
        return node

    def add_output_node(self, position: QPointF | None = None) -> OutputNodeItem:
        """Add a result node to the canvas."""
        node_id = self.graph.add_output()
        node = OutputNodeItem(node_id)
        self.addItem(node)
        node.setPos(position if position is not None else self._next_position())
        self.nodes[node_id] = node
        self.mark_modified()
        return node

    def add_obj_loader_node(self, position: QPointF | None = None) -> ObjLoaderNodeItem:
        """Add an Obj Loader node with four empty array outputs."""
        array_node_ids = (
            self.graph.add_literal(VertexArray(())),
            self.graph.add_literal(FaceArray(())),
            self.graph.add_literal(UVArray(())),
            self.graph.add_literal(NormalArray(())),
        )
        node = ObjLoaderNodeItem(
            array_node_ids[0], array_node_ids, self._load_obj_into_node
        )
        self.addItem(node)
        node.setPos(position if position is not None else self._next_position())
        self.nodes[node.node_id] = node
        self.mark_modified()
        return node

    def _load_obj_into_node(self, item: ObjLoaderNodeItem) -> None:
        """Prompt for an Obj file and load it into an existing loader node."""
        parent_widget = self.views()[0] if self.views() else None
        path, _name_filter = QFileDialog.getOpenFileName(
            parent_widget, "Load Obj", "", "Obj Files (*.obj)"
        )
        if not path:
            return
        try:
            obj = Obj()
            if not obj.load(path):
                raise GraphError(f"Could not parse {Path(path).name}")
            vertices, faces, uvs, normals = arrays_from_obj(obj)
        except (GraphError, OSError, ValueError, *_OBJ_PARSE_ERRORS) as error:
            item.set_status(str(error) or type(error).__name__, is_error=True)
            return

        vertices_id, faces_id, uv_id, normals_id = item.array_node_ids
        self.graph.set_literal(vertices_id, vertices)
        self.graph.set_literal(faces_id, faces)
        self.graph.set_literal(uv_id, uvs)
        self.graph.set_literal(normals_id, normals)
        item.set_status(
            f"{Path(path).name}: {len(vertices.values)} verts, "
            f"{len(faces.triangles)} faces"
        )
        self.update_outputs()

    def add_mesh_viewer_node(
        self,
        position: QPointF | None = None,
        shading_mode: str = SHADING_SOLID,
        wireframe: bool = False,
    ) -> MeshViewerNodeItem:
        """Add a mesh viewer sink node to the canvas."""
        node_id = self.graph.add_mesh_viewer()
        node = MeshViewerNodeItem(
            node_id,
            lambda viewer_id=node_id: self._mesh_viewer_config_changed(viewer_id),
            shading_mode=shading_mode,
            wireframe=wireframe,
        )
        self.addItem(node)
        node.setPos(position if position is not None else self._next_position())
        self.nodes[node_id] = node
        self.mark_modified()
        return node

    def _mesh_viewer_config_changed(self, node_id: str) -> None:
        """Refresh display-only settings without rebuilding valid geometry."""
        self.mark_modified()
        node = self.nodes.get(node_id)
        if not isinstance(node, MeshViewerNodeItem):
            return
        mesh_inputs = node.render_state.mesh_inputs
        diffuse_needs_normals = node.render_state.shading_mode == SHADING_DIFFUSE and (
            mesh_inputs is None
            or mesh_inputs.normals is None
            or not mesh_inputs.normals.values
        )
        if mesh_inputs is None or diffuse_needs_normals:
            self.update_outputs()

    def _value_changed(self, node_id: str, components: tuple[float, ...]) -> None:
        """Update a graph value after a spin box changes."""
        self.graph.set_value(node_id, components)
        self.update_outputs()

    def _generator_changed(
        self, node_id: str, parameter_index: int, components: tuple[float, ...]
    ) -> None:
        """Update a graph generator parameter after a spin box changes."""
        self.graph.set_generator_parameter(node_id, parameter_index, components)
        self.update_outputs()

    def _generator_rotation_order_changed(
        self, node_id: str, rotation_order: str
    ) -> None:
        """Update a Transform generator after its rotation order changes."""
        self.graph.set_generator_rotation_order(node_id, rotation_order)
        self.update_outputs()

    def _item_for_node_id(self, graph_node_id: str) -> BaseNodeItem:
        """Return the graphics item that owns a graph node id."""
        for node in self.nodes.values():
            if graph_node_id in node.owned_node_ids():
                return node
        raise KeyError(graph_node_id)

    def _output_port_for_node_id(self, graph_node_id: str) -> PortItem:
        """Return the output socket that sources a given graph node id."""
        item = self._item_for_node_id(graph_node_id)
        for port in item.output_ports:
            if port.graph_node_id == graph_node_id:
                return port
        raise KeyError(graph_node_id)

    def connect_ports(self, source: PortItem, target: PortItem) -> ConnectionItem:
        """Connect an output socket to an input, replacing its previous wire."""
        if not source.is_output or target.is_output or target.input_index is None:
            raise GraphError("Connections must run from an output to an input")
        if source.node is target.node:
            raise GraphError("A node cannot connect directly to itself")

        for old_connection in list(target.connections):
            self._remove_connection(old_connection)
        self.graph.connect(
            source.graph_node_id, target.node.node_id, target.input_index
        )
        connection = ConnectionItem(source, target)
        self.addItem(connection)
        self.connections.append(connection)
        target.colour = source.colour
        target.setBrush(QBrush(source.colour))
        connection.update_path()
        self.update_outputs()
        return connection

    def _remove_connection(self, connection: ConnectionItem) -> None:
        """Remove a visible wire before replacing an input connection."""
        connection.detach()
        self.connections.remove(connection)
        self.removeItem(connection)

    def delete_selected(self) -> None:
        """Remove the selected nodes, then any additionally selected wires."""
        selected_nodes = [
            item for item in self.selectedItems() if isinstance(item, BaseNodeItem)
        ]
        selected_connections = [
            item for item in self.selectedItems() if isinstance(item, ConnectionItem)
        ]
        if not selected_nodes and not selected_connections:
            return
        for node in selected_nodes:
            self._delete_node(node)
        for connection in selected_connections:
            if connection in self.connections:
                self._disconnect(connection)
        self.update_outputs()

    def _delete_node(self, node: BaseNodeItem) -> None:
        """Remove a node, detaching its wires and dropping it from the graph."""
        if isinstance(node, MeshViewerNodeItem):
            node.close_popup()
        ports = [*node.input_ports, *node.output_ports]
        for port in ports:
            for connection in list(port.connections):
                self._remove_connection(connection)
        for owned_id in node.owned_node_ids():
            self.graph.remove_node(owned_id)
        del self.nodes[node.node_id]
        self.removeItem(node)

    def _disconnect(self, connection: ConnectionItem) -> None:
        """Remove one wire and its underlying graph connection."""
        self.graph.disconnect(
            connection.target.node.node_id, connection.target.input_index
        )
        self._remove_connection(connection)

    def clear_graph(self) -> None:
        """Remove all visible items and start a fresh calculation graph."""
        for node in self.nodes.values():
            if isinstance(node, MeshViewerNodeItem):
                node.close_popup()
        self.clear()
        self.graph = MathGraph()
        self.nodes.clear()
        self.connections.clear()
        self._drag_source = None
        self._preview_connection = None
        self._insertion_index = 0
        self.mark_clean()

    def load_example(self) -> None:
        """Load the bundled Vec3 component-multiply example graph."""
        self.load_from_file(DEFAULT_EXAMPLE_PATH)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable snapshot of the graph and node layout."""
        nodes: list[dict[str, object]] = []
        for node_id, node in self.nodes.items():
            position = node.pos()
            entry: dict[str, object] = {
                "id": node_id,
                "name": node.title,
                "x": position.x(),
                "y": position.y(),
            }
            if isinstance(node, ValueNodeItem):
                entry["kind"] = "value"
                entry["math_type"] = node.math_type.name
                entry["components"] = list(self.graph.value_components(node_id))
            elif isinstance(node, OperationNodeItem):
                entry["kind"] = "operation"
                entry["operation"] = node.operation.name
            elif isinstance(node, GeneratorNodeItem):
                entry["kind"] = "generator"
                entry["operation"] = node.operation.name
                entry["parameters"] = [
                    list(components)
                    for components in self.graph.generator_parameters(node_id)
                ]
                if node.operation is Operation.TRANSFORM:
                    entry["rotation_order"] = self.graph.generator_rotation_order(
                        node_id
                    )
            elif isinstance(node, ObjLoaderNodeItem):
                entry["kind"] = "obj_loader"
                entry["array_ids"] = list(node.array_node_ids)
                vertices, faces, uvs, normals = (
                    self.graph.evaluate(array_id) for array_id in node.array_node_ids
                )
                entry["vertices"] = [v.to_list() for v in vertices.values]
                entry["faces"] = [
                    [list(corner) for corner in triangle]
                    for triangle in faces.triangles
                ]
                entry["uvs"] = [uv.to_list() for uv in uvs.values]
                entry["normals"] = [n.to_list() for n in normals.values]
                entry["status"] = node.status_text_item.toPlainText()
            elif isinstance(node, MeshViewerNodeItem):
                entry["kind"] = "mesh_viewer"
                entry["shading_mode"] = node.render_state.shading_mode
                entry["wireframe"] = node.render_state.wireframe
            else:
                entry["kind"] = "output"
            nodes.append(entry)
        connections = [
            {
                "source": connection.source.graph_node_id,
                "target": connection.target.node.node_id,
                "input": connection.target.input_index,
            }
            for connection in self.connections
        ]
        return {
            "schema_version": SCHEMA_VERSION,
            "nodes": nodes,
            "connections": connections,
        }

    def from_dict(self, data: dict[str, object]) -> None:
        """Replace the current graph with one built from ``to_dict`` data."""
        validate_document(data)
        self._loading = True
        try:
            self.clear_graph()
            id_map: dict[str, str] = {}
            for entry in data["nodes"]:
                position = QPointF(entry["x"], entry["y"])
                kind = entry["kind"]
                if kind == "value":
                    node = self.add_value_node(
                        MathType[entry["math_type"]],
                        position,
                        tuple(entry["components"]),
                    )
                    id_map[entry["id"]] = node.node_id
                elif kind == "operation":
                    node = self.add_operation_node(
                        Operation[entry["operation"]], position
                    )
                    id_map[entry["id"]] = node.node_id
                elif kind == "generator":
                    node = self.add_generator_node(
                        Operation[entry["operation"]],
                        position,
                        tuple(tuple(p) for p in entry["parameters"]),
                        rotation_order=entry.get("rotation_order"),
                    )
                    id_map[entry["id"]] = node.node_id
                elif kind == "output":
                    node = self.add_output_node(position)
                    id_map[entry["id"]] = node.node_id
                elif kind == "obj_loader":
                    node = self.add_obj_loader_node(position)
                    node.set_status(entry.get("status", ""))
                    vertices = VertexArray(tuple(Vec3(*v) for v in entry["vertices"]))
                    faces = FaceArray(
                        tuple(
                            tuple(
                                (corner[0], corner[1], corner[2]) for corner in triangle
                            )
                            for triangle in entry["faces"]
                        )
                    )
                    uvs = UVArray(tuple(Vec2(*uv) for uv in entry["uvs"]))
                    normals = NormalArray(tuple(Vec3(*n) for n in entry["normals"]))
                    self.graph.set_literal(node.array_node_ids[0], vertices)
                    self.graph.set_literal(node.array_node_ids[1], faces)
                    self.graph.set_literal(node.array_node_ids[2], uvs)
                    self.graph.set_literal(node.array_node_ids[3], normals)
                    for old_id, new_id in zip(entry["array_ids"], node.array_node_ids):
                        id_map[old_id] = new_id
                elif kind == "mesh_viewer":
                    node = self.add_mesh_viewer_node(
                        position,
                        shading_mode=entry.get("shading_mode", SHADING_SOLID),
                        wireframe=bool(entry.get("wireframe", False)),
                    )
                    id_map[entry["id"]] = node.node_id
                else:
                    raise GraphError(f"Unknown node kind {kind!r}")
                node.set_name(entry.get("name", node.title))
            for connection in data["connections"]:
                source_port = self._output_port_for_node_id(
                    id_map[connection["source"]]
                )
                target_node = self.nodes[id_map[connection["target"]]]
                self.connect_ports(
                    source_port,
                    target_node.input_ports[connection["input"]],
                )
            self.update_outputs()
        finally:
            self._loading = False

    def save_to_file(self, path: str | Path) -> None:
        """Write the current graph to a JSON file."""
        destination = Path(path)
        payload = (json.dumps(self.to_dict(), indent=2) + "\n").encode("utf-8")
        save_file = QSaveFile(str(destination))
        if not save_file.open(QIODevice.OpenModeFlag.WriteOnly):
            raise OSError(save_file.errorString())
        if save_file.write(payload) != len(payload):
            error_message = save_file.errorString()
            save_file.cancelWriting()
            raise OSError(error_message)
        if not save_file.commit():
            raise OSError(save_file.errorString())

    def load_from_file(self, path: str | Path) -> None:
        """Replace the current graph with one loaded from a JSON file."""
        self.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def update_outputs(self) -> None:
        """Evaluate and refresh every output and mesh viewer node in the scene."""
        self.mark_modified()
        for node_id, node in self.nodes.items():
            if isinstance(node, OutputNodeItem):
                try:
                    node.set_result(format_value(self.graph.evaluate(node_id)))
                except (
                    GraphError,
                    KeyError,
                    TypeError,
                    ValueError,
                    AttributeError,
                ) as error:
                    node.set_result(str(error) or type(error).__name__, is_error=True)
            elif isinstance(node, MeshViewerNodeItem):
                try:
                    mesh_inputs = self.graph.evaluate_mesh_viewer(node_id)
                    if node.render_state.shading_mode == SHADING_DIFFUSE and (
                        mesh_inputs.normals is None or not mesh_inputs.normals.values
                    ):
                        raise GraphError(
                            "Mesh Viewer needs input Normals for Diffuse shading"
                        )
                    node.set_mesh_inputs(mesh_inputs)
                except (
                    GraphError,
                    KeyError,
                    TypeError,
                    ValueError,
                    AttributeError,
                ) as error:
                    node.show_error(str(error) or type(error).__name__)

    def output_texts(self) -> list[str]:
        """Return the current output strings in graph insertion order."""
        return [
            node.value_text()
            for node in self.nodes.values()
            if isinstance(node, OutputNodeItem)
        ]

    def _set_preview_path(self, end: QPointF) -> None:
        """Update the temporary wire shown whilst making a connection."""
        if self._drag_source is None or self._preview_connection is None:
            return
        self._preview_connection.setPath(
            _bezier_path(self._drag_source.scene_centre(), end)
        )

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Start a wire drag when an output socket is pressed."""
        item = self.itemAt(event.scenePos(), QTransform())
        if isinstance(item, PortItem) and item.is_output:
            self._drag_source = item
            self._preview_connection = QGraphicsPathItem()
            self._preview_connection.setPen(
                QPen(item.colour, 3.0, Qt.PenStyle.DashLine)
            )
            self._preview_connection.setZValue(-1.0)
            self.addItem(self._preview_connection)
            self._set_preview_path(event.scenePos())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Move the temporary wire with the pointer."""
        if self._drag_source is not None:
            self._set_preview_path(event.scenePos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Finish a wire when it is released over an input socket."""
        if self._drag_source is not None:
            target = self.itemAt(event.scenePos(), QTransform())
            if isinstance(target, PortItem) and not target.is_output:
                try:
                    self.connect_ports(self._drag_source, target)
                except GraphError:
                    pass
            if self._preview_connection is not None:
                self.removeItem(self._preview_connection)
            self._drag_source = None
            self._preview_connection = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


class MathNodeView(QGraphicsView):
    """Zoomable view with a subdued node-editor grid."""

    MIN_ZOOM = 0.15
    MAX_ZOOM = 3.0
    ZOOM_STEP = 1.1
    FRAME_ALL_MARGIN = 60.0

    def __init__(self, scene: MathNodeScene, parent: QWidget | None = None) -> None:
        """Configure rendering and navigation for the graph scene."""
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor("#111620")))
        self.node_menu = NodeCreationMenu(scene, self)
        self.viewport().installEventFilter(self)

    def _open_node_menu_at_cursor(self) -> None:
        """Open the creation menu at the pointer or the view centre."""
        viewport_position = self.viewport().mapFromGlobal(QCursor.pos())
        if not self.viewport().rect().contains(viewport_position):
            viewport_position = self.viewport().rect().center()
        self.open_node_menu(viewport_position)

    def open_node_menu(self, viewport_position: QPoint) -> NodeCreationMenu:
        """Open the node menu and remember where its node should be placed."""
        scene_position = self.mapToScene(viewport_position)
        global_position = self.viewport().mapToGlobal(viewport_position)
        self.node_menu.open_at(scene_position, global_position)
        return self.node_menu

    def _handle_key_press(self, event: QKeyEvent) -> bool:
        """Handle a key press shared between the view and its viewport."""
        if isinstance(self.scene().focusItem(), QGraphicsProxyWidget):
            return False
        if event.key() == Qt.Key.Key_Tab:
            self._open_node_menu_at_cursor()
            return True
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.scene().delete_selected()
            return True
        if event.key() == Qt.Key.Key_H:
            self.frame_all()
            return True
        return False

    def event(self, event: QEvent) -> bool:
        """Catch Tab and Delete before Qt uses them for other purposes."""
        if event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            if self._handle_key_press(event):
                event.accept()
                return True
        return super().event(event)

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        """Catch Tab, Delete and right-clicks on the graphics view viewport."""
        if watched is not self.viewport():
            return super().eventFilter(watched, event)
        if event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            if self._handle_key_press(event):
                event.accept()
                return True
        if event.type() == QEvent.Type.ContextMenu and isinstance(
            event, QContextMenuEvent
        ):
            self._show_context_menu(event.pos(), event.globalPos())
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def _deletable_item_at(
        self, viewport_position: QPoint
    ) -> BaseNodeItem | ConnectionItem | None:
        """Return the node or connection under a viewport position, if any."""
        scene_position = self.mapToScene(viewport_position)
        item: QGraphicsItem | None = self.scene().itemAt(scene_position, QTransform())
        while item is not None:
            if isinstance(item, (BaseNodeItem, ConnectionItem)):
                return item
            item = item.parentItem()
        return None

    def _delete_item_at(self, viewport_position: QPoint) -> None:
        """Select and delete the node or wire at a viewport position."""
        target = self._deletable_item_at(viewport_position)
        if target is None:
            return
        self.scene().clearSelection()
        target.setSelected(True)
        self.scene().delete_selected()

    def _show_context_menu(
        self, viewport_position: QPoint, global_position: QPoint
    ) -> None:
        """Offer to delete the node or wire under the cursor."""
        if self._deletable_item_at(viewport_position) is None:
            return
        menu = QMenu(self)
        menu.addAction("Delete", lambda: self._delete_item_at(viewport_position))
        menu.exec(global_position)

    def _nodes_bounding_rect(self) -> QRectF | None:
        """Return the scene rect spanning every node, or None if there are none."""
        nodes = list(self.scene().nodes.values())
        if not nodes:
            return None
        rect = nodes[0].sceneBoundingRect()
        for node in nodes[1:]:
            rect = rect.united(node.sceneBoundingRect())
        return rect

    def _clamp_zoom(self) -> None:
        """Rescale the view so its zoom stays within the configured limits."""
        current_scale = self.transform().m11()
        clamped_scale = min(max(current_scale, self.MIN_ZOOM), self.MAX_ZOOM)
        if clamped_scale != current_scale:
            self.scale(clamped_scale / current_scale, clamped_scale / current_scale)

    def frame_all(self) -> None:
        """Fit every node into the viewport, within the configured zoom limits."""
        nodes_rect = self._nodes_bounding_rect()
        if nodes_rect is None:
            return
        margin = self.FRAME_ALL_MARGIN
        self.fitInView(
            nodes_rect.adjusted(-margin, -margin, margin, margin),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self._clamp_zoom()

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        """Draw minor and major grid lines behind the nodes."""
        super().drawBackground(painter, rect)
        minor_step = 20
        major_step = 100
        left = int(rect.left()) - (int(rect.left()) % minor_step)
        top = int(rect.top()) - (int(rect.top()) % minor_step)
        minor_lines: list[QLineF] = []
        major_lines: list[QLineF] = []
        for x_position in range(left, int(rect.right()), minor_step):
            line = QLineF(
                float(x_position), rect.top(), float(x_position), rect.bottom()
            )
            (major_lines if x_position % major_step == 0 else minor_lines).append(line)
        for y_position in range(top, int(rect.bottom()), minor_step):
            line = QLineF(
                rect.left(), float(y_position), rect.right(), float(y_position)
            )
            (major_lines if y_position % major_step == 0 else minor_lines).append(line)
        painter.setPen(QPen(QColor("#18202d"), 1.0))
        painter.drawLines(minor_lines)
        painter.setPen(QPen(QColor("#222d3e"), 1.0))
        painter.drawLines(major_lines)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Adjust a spin box under the pointer, or zoom the canvas around it."""
        if self._forward_wheel_to_spin_box(event):
            return
        factor = self.ZOOM_STEP if event.angleDelta().y() > 0 else 1.0 / self.ZOOM_STEP
        current_scale = self.transform().m11()
        clamped_scale = min(max(current_scale * factor, self.MIN_ZOOM), self.MAX_ZOOM)
        self.scale(clamped_scale / current_scale, clamped_scale / current_scale)

    def _forward_wheel_to_spin_box(self, event: QWheelEvent) -> bool:
        """Forward the event to a spin box under the pointer; report whether it was."""
        scene_position = self.mapToScene(event.position().toPoint())
        item = self.scene().itemAt(scene_position, QTransform())
        if not isinstance(item, QGraphicsProxyWidget):
            return False
        widget = item.widget()
        if widget is None:
            return False
        local_position = item.mapFromScene(scene_position).toPoint()
        child = widget.childAt(local_position)
        while child is not None and not isinstance(child, QAbstractSpinBox):
            child = child.parentWidget()
        if child is None:
            return False
        QApplication.sendEvent(child, event)
        return True
