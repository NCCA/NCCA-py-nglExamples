"""PySide6 node editor for experimenting with PyNGL maths types."""

from __future__ import annotations

import sys
from collections.abc import Callable

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsProxyWidget,
    QGraphicsScene,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
    QGraphicsTextItem,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyleOptionGraphicsItem,
    QVBoxLayout,
    QWidget,
)

from .math_graph import (
    OPERATION_ARITY,
    OPERATION_INPUT_NAMES,
    TYPE_SHAPES,
    GraphError,
    MathGraph,
    MathType,
    Operation,
    format_value,
)

NODE_HEADER_HEIGHT = 32.0
PORT_RADIUS = 6.0

TYPE_COLOURS: dict[MathType, QColor] = {
    MathType.FLOAT: QColor("#d7dce5"),
    MathType.VEC2: QColor("#42c9c2"),
    MathType.VEC3: QColor("#63d471"),
    MathType.VEC4: QColor("#4ea5ff"),
    MathType.MAT2: QColor("#f4bf55"),
    MathType.MAT3: QColor("#f2935c"),
    MathType.MAT4: QColor("#ea6f91"),
    MathType.QUATERNION: QColor("#b58cff"),
}
GENERIC_PORT_COLOUR = QColor("#ad8cff")

MATH_OPERATIONS = (
    Operation.ADD,
    Operation.SUBTRACT,
    Operation.MULTIPLY,
    Operation.MATRIX_MULTIPLY,
    Operation.DOT,
    Operation.CROSS,
    Operation.NORMALISE,
    Operation.TRANSPOSE,
)
MAT4_OPERATIONS = (
    Operation.LOOK_AT,
    Operation.PERSPECTIVE,
    Operation.ORTHO,
    Operation.FRUSTUM,
    Operation.MAT4_TRANSLATE,
    Operation.MAT4_SCALE,
    Operation.MAT4_ROTATE_X,
    Operation.MAT4_ROTATE_Y,
    Operation.MAT4_ROTATE_Z,
)
QUATERNION_OPERATIONS = (
    Operation.QUATERNION_FROM_AXIS_ANGLE,
    Operation.QUATERNION_PRODUCT,
    Operation.QUATERNION_ROTATE_VECTOR,
    Operation.QUATERNION_TO_MAT4,
    Operation.MAT4_TO_QUATERNION,
    Operation.QUATERNION_SLERP,
    Operation.QUATERNION_CONJUGATE,
    Operation.QUATERNION_INVERSE,
)


def node_title_font() -> QFont:
    """Return the font shared by node titles and width calculations."""
    title_font = QFont()
    title_font.setPointSize(10)
    title_font.setBold(True)
    return title_font


def default_components(math_type: MathType) -> tuple[float, ...]:
    """Return sensible default components for a new value node."""
    if math_type is MathType.QUATERNION:
        return (1.0, 0.0, 0.0, 0.0)
    rows, columns = TYPE_SHAPES[math_type]
    if rows == 1:
        return (0.0,) * columns
    return tuple(
        1.0 if row == column else 0.0
        for row in range(rows)
        for column in range(columns)
    )


class PortItem(QGraphicsEllipseItem):
    """A connection socket belonging to a graphics node.

    Attributes
    ----------
        node : BaseNodeItem
            node which owns this port
        input_index : int | None
            graph input index, or ``None`` for an output port
        connections : list[ConnectionItem]
            visible wires attached to the port
    """

    def __init__(
        self,
        node: BaseNodeItem,
        input_index: int | None,
        colour: QColor,
    ) -> None:
        """Create an input or output socket on a node."""
        super().__init__(
            -PORT_RADIUS,
            -PORT_RADIUS,
            PORT_RADIUS * 2.0,
            PORT_RADIUS * 2.0,
            node,
        )
        self.node = node
        self.input_index = input_index
        self.colour = colour
        self.connections: list[ConnectionItem] = []
        self.setBrush(QBrush(colour))
        self.setPen(QPen(QColor("#10141d"), 2.0))
        self.setZValue(3.0)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setToolTip("Output" if self.is_output else f"Input {input_index}")

    @property
    def is_output(self) -> bool:
        """Return whether this socket sends values."""
        return self.input_index is None

    def scene_centre(self) -> QPointF:
        """Return the socket centre in scene coordinates."""
        return self.mapToScene(QPointF(0.0, 0.0))

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        """Highlight the socket under the pointer."""
        self.setBrush(QBrush(self.colour.lighter(140)))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        """Restore the normal socket colour."""
        self.setBrush(QBrush(self.colour))
        super().hoverLeaveEvent(event)


class ConnectionItem(QGraphicsPathItem):
    """A curved wire joining an output socket to an input socket."""

    def __init__(self, source: PortItem, target: PortItem) -> None:
        """Create a visible graph connection."""
        super().__init__()
        self.source = source
        self.target = target
        self.source.connections.append(self)
        self.target.connections.append(self)
        self.setPen(QPen(source.colour, 3.0))
        self.setZValue(-1.0)
        self.update_path()

    def update_path(self) -> None:
        """Rebuild the Bezier path after either node moves."""
        start = self.source.scene_centre()
        end = self.target.scene_centre()
        control_distance = max(70.0, abs(end.x() - start.x()) * 0.5)
        path = QPainterPath(start)
        path.cubicTo(
            QPointF(start.x() + control_distance, start.y()),
            QPointF(end.x() - control_distance, end.y()),
            end,
        )
        self.setPath(path)

    def detach(self) -> None:
        """Remove the wire from both sockets."""
        self.source.connections.remove(self)
        self.target.connections.remove(self)


class BaseNodeItem(QGraphicsObject):
    """Shared painted body and sockets for all node types."""

    def __init__(
        self,
        node_id: str,
        title: str,
        width: float,
        height: float,
        input_names: tuple[str, ...],
        output_colour: QColor | None,
    ) -> None:
        """Create a movable graphics node."""
        super().__init__()
        self.node_id = node_id
        self.title = title
        self.width = width
        self.height = height
        self.input_names = input_names
        self.input_ports: list[PortItem] = []
        self.output_port: PortItem | None = None
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.setZValue(1.0)

        for input_index, _name in enumerate(input_names):
            port = PortItem(self, input_index, GENERIC_PORT_COLOUR)
            port.setPos(0.0, NODE_HEADER_HEIGHT + 28.0 + input_index * 30.0)
            self.input_ports.append(port)
        if output_colour is not None:
            self.output_port = PortItem(self, None, output_colour)
            self.output_port.setPos(width, NODE_HEADER_HEIGHT + 28.0)

    def boundingRect(self) -> QRectF:
        """Return the painted node bounds."""
        margin = PORT_RADIUS + 2.0
        return QRectF(
            -margin, -margin, self.width + margin * 2.0, self.height + margin * 2.0
        )

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint the node body, header, title and socket labels."""
        del widget
        body = QRectF(0.0, 0.0, self.width, self.height)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(QColor("#202735")))
        border_colour = (
            QColor("#74a7ff")
            if option.state & QStyle.StateFlag.State_Selected
            else QColor("#3d495e")
        )
        painter.setPen(QPen(border_colour, 2.0))
        painter.drawRoundedRect(body, 8.0, 8.0)

        painter.setBrush(QBrush(QColor("#303c52")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(
            QRectF(1.0, 1.0, self.width - 2.0, NODE_HEADER_HEIGHT), 7.0, 7.0
        )
        painter.drawRect(QRectF(1.0, NODE_HEADER_HEIGHT - 7.0, self.width - 2.0, 8.0))

        painter.setPen(QPen(QColor("#f0f3f8")))
        painter.setFont(node_title_font())
        painter.drawText(
            QRectF(12.0, 0.0, self.width - 24.0, NODE_HEADER_HEIGHT),
            Qt.AlignmentFlag.AlignVCenter,
            self.title,
        )

        label_font = QFont()
        label_font.setPointSize(9)
        painter.setFont(label_font)
        painter.setPen(QPen(QColor("#c2cad7")))
        for input_index, input_name in enumerate(self.input_names):
            y_position = NODE_HEADER_HEIGHT + 20.0 + input_index * 30.0
            painter.drawText(
                QRectF(12.0, y_position, self.width - 24.0, 18.0), input_name
            )

    def itemChange(
        self,
        change: QGraphicsItem.GraphicsItemChange,
        value: object,
    ) -> object:
        """Keep attached wires aligned when the node moves."""
        if change is QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            ports = [*self.input_ports]
            if self.output_port is not None:
                ports.append(self.output_port)
            for port in ports:
                for connection in port.connections:
                    connection.update_path()
        return super().itemChange(change, value)


class ValueNodeItem(BaseNodeItem):
    """A maths value node with editable component values."""

    def __init__(
        self,
        node_id: str,
        math_type: MathType,
        components: tuple[float, ...],
        on_change: Callable[[tuple[float, ...]], None],
    ) -> None:
        """Create the numeric editors for a PyNGL value."""
        rows, columns = TYPE_SHAPES[math_type]
        width = max(190.0, columns * 76.0 + 24.0)
        height = NODE_HEADER_HEIGHT + rows * 38.0 + 20.0
        title = (
            "Quaternion (s, x, y, z)"
            if math_type is MathType.QUATERNION
            else math_type.value
        )
        super().__init__(node_id, title, width, height, (), TYPE_COLOURS[math_type])
        self.math_type = math_type
        self.on_change = on_change
        self.spin_boxes: list[QDoubleSpinBox] = []

        editor = QWidget()
        editor.setStyleSheet(
            "QWidget { background: transparent; } QDoubleSpinBox { background: #151a24; color: #edf1f7; border: 1px solid #46536a; border-radius: 3px; padding: 2px; }"
        )
        layout = QGridLayout(editor)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(4)
        for component_index, component in enumerate(components):
            spin_box = QDoubleSpinBox()
            spin_box.setRange(-1_000_000.0, 1_000_000.0)
            spin_box.setDecimals(3)
            spin_box.setSingleStep(0.1)
            spin_box.setValue(component)
            spin_box.setFixedWidth(68)
            spin_box.valueChanged.connect(self._values_changed)
            layout.addWidget(
                spin_box, component_index // columns, component_index % columns
            )
            self.spin_boxes.append(spin_box)

        self.proxy = QGraphicsProxyWidget(self)
        self.proxy.setWidget(editor)
        self.proxy.setPos(12.0, NODE_HEADER_HEIGHT + 9.0)
        self.proxy.setZValue(2.0)
        if self.output_port is not None:
            self.output_port.setPos(width, NODE_HEADER_HEIGHT + 20.0)

    def _values_changed(self, _value: float) -> None:
        """Send the edited components back to the graph model."""
        self.on_change(tuple(spin_box.value() for spin_box in self.spin_boxes))


class OperationNodeItem(BaseNodeItem):
    """A PyNGL operation node with one or more named inputs."""

    def __init__(self, node_id: str, operation: Operation) -> None:
        """Create the named sockets needed by an operation."""
        arity = OPERATION_ARITY[operation]
        input_names = OPERATION_INPUT_NAMES[operation]
        height = NODE_HEADER_HEIGHT + 38.0 + max(0, arity - 1) * 30.0
        title_width = QFontMetrics(node_title_font()).horizontalAdvance(operation.value)
        width = max(180.0, float(title_width + 24))
        super().__init__(
            node_id, operation.value, width, height, input_names, GENERIC_PORT_COLOUR
        )
        self.operation = operation
        if self.output_port is not None:
            self.output_port.setPos(self.width, NODE_HEADER_HEIGHT + 28.0)


class OutputNodeItem(BaseNodeItem):
    """A node which displays the final formatted result or graph error."""

    def __init__(self, node_id: str) -> None:
        """Create a result node with one input socket."""
        super().__init__(node_id, "Output", 260.0, 132.0, ("Value",), None)
        self.result_text = QGraphicsTextItem("Waiting for input", self)
        self.result_text.setDefaultTextColor(QColor("#e8edf5"))
        self.result_text.setFont(QFont("Monaco", 10))
        self.result_text.setTextWidth(225.0)
        self.result_text.setPos(18.0, NODE_HEADER_HEIGHT + 42.0)

    def set_result(self, text: str, is_error: bool = False) -> None:
        """Update the visible result string."""
        colour = QColor("#ff8a8a") if is_error else QColor("#d9f99d")
        self.result_text.setDefaultTextColor(colour)
        self.result_text.setPlainText(text)
        text_metrics = QFontMetrics(self.result_text.font())
        longest_line = max(
            text_metrics.horizontalAdvance(line) for line in text.splitlines()
        )
        required_width = max(260.0, float(longest_line + 48))
        if required_width != self.width:
            self.prepareGeometryChange()
            self.width = required_width
            self.result_text.setTextWidth(self.width - 35.0)
        required_height = max(132.0, 82.0 + self.result_text.boundingRect().height())
        if required_height != self.height:
            self.prepareGeometryChange()
            self.height = required_height
            self.update()

    def value_text(self) -> str:
        """Return the plain result string for tests and accessibility."""
        return self.result_text.toPlainText()


class MathNodeScene(QGraphicsScene):
    """Graphics scene which keeps visible wires and the maths graph in sync."""

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
        return node

    def add_output_node(self, position: QPointF | None = None) -> OutputNodeItem:
        """Add a result node to the canvas."""
        node_id = self.graph.add_output()
        node = OutputNodeItem(node_id)
        self.addItem(node)
        node.setPos(position if position is not None else self._next_position())
        self.nodes[node_id] = node
        return node

    def _value_changed(self, node_id: str, components: tuple[float, ...]) -> None:
        """Update a graph value after a spin box changes."""
        self.graph.set_value(node_id, components)
        self.update_outputs()

    def connect_ports(self, source: PortItem, target: PortItem) -> ConnectionItem:
        """Connect an output socket to an input, replacing its previous wire."""
        if not source.is_output or target.is_output or target.input_index is None:
            raise GraphError("Connections must run from an output to an input")
        if source.node is target.node:
            raise GraphError("A node cannot connect directly to itself")

        for old_connection in list(target.connections):
            self._remove_connection(old_connection)
        self.graph.connect(source.node.node_id, target.node.node_id, target.input_index)
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

    def clear_graph(self) -> None:
        """Remove all visible items and start a fresh calculation graph."""
        self.clear()
        self.graph = MathGraph()
        self.nodes.clear()
        self.connections.clear()
        self._drag_source = None
        self._preview_connection = None
        self._insertion_index = 0

    def load_example(self) -> None:
        """Build the requested component-wise Vec3 multiplication example."""
        self.clear_graph()
        left = self.add_value_node(
            MathType.VEC3, QPointF(-520.0, -130.0), (1.0, 2.0, 3.0)
        )
        right = self.add_value_node(
            MathType.VEC3, QPointF(-520.0, 100.0), (4.0, 5.0, 6.0)
        )
        multiply = self.add_operation_node(Operation.MULTIPLY, QPointF(-140.0, 0.0))
        output = self.add_output_node(QPointF(220.0, 0.0))
        assert left.output_port is not None
        assert right.output_port is not None
        assert multiply.output_port is not None
        self.connect_ports(left.output_port, multiply.input_ports[0])
        self.connect_ports(right.output_port, multiply.input_ports[1])
        self.connect_ports(multiply.output_port, output.input_ports[0])

    def update_outputs(self) -> None:
        """Evaluate and refresh every output node in the scene."""
        for node_id, node in self.nodes.items():
            if not isinstance(node, OutputNodeItem):
                continue
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
        start = self._drag_source.scene_centre()
        control_distance = max(70.0, abs(end.x() - start.x()) * 0.5)
        path = QPainterPath(start)
        path.cubicTo(
            QPointF(start.x() + control_distance, start.y()),
            QPointF(end.x() - control_distance, end.y()),
            end,
        )
        self._preview_connection.setPath(path)

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

    def __init__(self, scene: MathNodeScene, parent: QWidget | None = None) -> None:
        """Configure rendering and navigation for the graph scene."""
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor("#111620")))

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
        """Zoom the canvas around the pointer."""
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)


class NodePalette(QWidget):
    """Buttons used to add value, operation and output nodes."""

    def __init__(self, canvas: MathNodeScene, parent: QWidget | None = None) -> None:
        """Build the node palette for a canvas."""
        super().__init__(parent)
        self.canvas = canvas
        self.setFixedWidth(220)
        self.setStyleSheet(
            "QWidget { background: #171d28; color: #e8edf5; }"
            "QGroupBox { border: 1px solid #344056; border-radius: 5px; margin-top: 9px; padding-top: 8px; font-weight: bold; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }"
            "QPushButton { background: #29354a; border: 1px solid #40516d; border-radius: 4px; padding: 6px; text-align: left; }"
            "QPushButton:hover { background: #354763; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        heading = QLabel("PyNGL Maths")
        heading_font = heading.font()
        heading_font.setPointSize(15)
        heading_font.setBold(True)
        heading.setFont(heading_font)
        layout.addWidget(heading)
        help_label = QLabel(
            "Click to add nodes. Drag from an output socket to an input socket to connect them."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #aeb9c9;")
        layout.addWidget(help_label)

        value_actions = [
            (math_type.value, lambda value=math_type: self.canvas.add_value_node(value))
            for math_type in MathType
        ]
        self._add_group(layout, "Values", value_actions)
        self._add_group(layout, "Maths", self._operation_actions(MATH_OPERATIONS))
        self._add_group(layout, "Mat4", self._operation_actions(MAT4_OPERATIONS))
        self._add_group(
            layout,
            "Quaternion",
            self._operation_actions(QUATERNION_OPERATIONS),
        )
        self._add_group(layout, "Result", [("Output", self.canvas.add_output_node)])
        layout.addStretch(1)

        example_button = QPushButton("Load Vec3 example")
        example_button.clicked.connect(
            lambda _checked=False: self.canvas.load_example()
        )
        layout.addWidget(example_button)
        clear_button = QPushButton("Clear graph")
        clear_button.clicked.connect(lambda _checked=False: self.canvas.clear_graph())
        layout.addWidget(clear_button)

    def _operation_actions(
        self,
        operations: tuple[Operation, ...],
    ) -> list[tuple[str, Callable[[], object]]]:
        """Return palette actions for an ordered set of operations."""
        return [
            (
                operation.value,
                lambda value=operation: self.canvas.add_operation_node(value),
            )
            for operation in operations
        ]

    def _add_group(
        self,
        parent_layout: QVBoxLayout,
        title: str,
        actions: list[tuple[str, Callable[[], object]]],
    ) -> None:
        """Add a titled group of palette buttons."""
        group = QGroupBox(title)
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(7, 10, 7, 7)
        group_layout.setSpacing(4)
        for label, callback in actions:
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, action=callback: action())
            group_layout.addWidget(button)
        parent_layout.addWidget(group)


class MathNodeWindow(QMainWindow):
    """Main window containing the palette and node graph canvas."""

    def __init__(self, load_example: bool = True) -> None:
        """Create the editor and optionally load the Vec3 example."""
        super().__init__()
        self.setWindowTitle("PyNGL Maths Node Editor")
        self.resize(1280, 760)
        self.canvas = MathNodeScene(self)
        self.view = MathNodeView(self.canvas, self)
        self.view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.palette = NodePalette(self.canvas, self)
        self.palette.setMinimumHeight(self.palette.sizeHint().height())
        self.palette_scroll = QScrollArea(self)
        self.palette_scroll.setWidget(self.palette)
        self.palette_scroll.setWidgetResizable(True)
        self.palette_scroll.setFixedWidth(240)
        self.palette_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        central_widget = QWidget()
        central_layout = QHBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.palette_scroll)
        central_layout.addWidget(self.view, 1)
        self.setCentralWidget(central_widget)
        self.statusBar().showMessage(
            "Edit values and wire nodes together; outputs update immediately"
        )
        if load_example:
            self.canvas.load_example()
            self.view.centerOn(0.0, 0.0)


def main() -> int:
    """Run the maths node editor application."""
    application = QApplication.instance()
    if application is None:
        application = QApplication(sys.argv)
    application.setApplicationName("PyNGL Maths Node Editor")
    window = MathNodeWindow(load_example=True)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
