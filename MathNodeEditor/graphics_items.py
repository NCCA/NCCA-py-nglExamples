"""Graphics items for the PyNGL maths node editor canvas."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsProxyWidget,
    QGraphicsSceneHoverEvent,
    QGraphicsTextItem,
    QGridLayout,
    QStyle,
    QStyleOptionGraphicsItem,
    QWidget,
)

from .math_graph import (
    OPERATION_ARITY,
    OPERATION_INPUT_NAMES,
    TYPE_SHAPES,
    MathType,
    Operation,
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


def _bezier_path(start: QPointF, end: QPointF) -> QPainterPath:
    """Return a cubic Bezier wire path between two scene points."""
    control_distance = max(70.0, abs(end.x() - start.x()) * 0.5)
    path = QPainterPath(start)
    path.cubicTo(
        QPointF(start.x() + control_distance, start.y()),
        QPointF(end.x() - control_distance, end.y()),
        end,
    )
    return path


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
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.update_path()

    def update_path(self) -> None:
        """Rebuild the Bezier path after either node moves."""
        self.setPath(
            _bezier_path(self.source.scene_centre(), self.target.scene_centre())
        )

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
