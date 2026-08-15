"""Graphics items for the PyNGL maths node editor canvas."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from math_graph import (
    GENERATOR_OUTPUT_TYPE,
    MESH_VIEWER_INPUT_NAMES,
    OPERATION_ARITY,
    OPERATION_INPUT_NAMES,
    OPERATION_PARAMETER_TYPES,
    TYPE_SHAPES,
    MathType,
    Operation,
)
from mesh_view import (
    SHADING_MODES,
    SHADING_SOLID,
    MeshPopupWindow,
    MeshPreviewWidget,
    MeshRenderState,
)
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
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsProxyWidget,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
    QGraphicsTextItem,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QStyle,
    QStyleOptionGraphicsItem,
    QWidget,
)

if TYPE_CHECKING:
    from math_graph import MeshViewerInputs

NODE_HEADER_HEIGHT = 32.0
PORT_RADIUS = 6.0
NUMERIC_DECIMALS = 7
NUMERIC_EDITOR_WIDTH = 92
NUMERIC_LIMIT = 1.0e30

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

MESH_ARRAY_LABELS: tuple[str, ...] = ("Vertices", "Faces", "UVs", "Normals")
MESH_ARRAY_COLOURS: dict[str, QColor] = {
    "Vertices": QColor("#7fe0c0"),
    "Faces": QColor("#e0c37f"),
    "UVs": QColor("#7fb8e0"),
    "Normals": QColor("#c47fe0"),
}


def node_title_font() -> QFont:
    """Return the font shared by node titles and width calculations."""
    title_font = QFont()
    title_font.setPointSize(10)
    title_font.setBold(True)
    return title_font


def numeric_spin_box(value: float) -> QDoubleSpinBox:
    """Build a numeric editor without throwing away useful float precision."""
    spin_box = QDoubleSpinBox()
    spin_box.setRange(-NUMERIC_LIMIT, NUMERIC_LIMIT)
    spin_box.setDecimals(NUMERIC_DECIMALS)
    spin_box.setSingleStep(0.1)
    spin_box.setValue(value)
    spin_box.setFixedWidth(NUMERIC_EDITOR_WIDTH)
    return spin_box


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


GENERATOR_DEFAULTS: dict[Operation, tuple[tuple[float, ...], ...]] = {
    Operation.LOOK_AT: ((0.0, 2.0, 8.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    Operation.PERSPECTIVE: ((45.0,), (1.778,), (0.1,), (100.0,)),
    Operation.ORTHO: ((-10.0,), (10.0,), (-10.0,), (10.0,), (0.1,), (100.0,)),
    Operation.FRUSTUM: ((-1.0,), (1.0,), (-1.0,), (1.0,), (0.1,), (100.0,)),
    Operation.MAT4_TRANSLATE: ((0.0,), (0.0,), (0.0,)),
    Operation.MAT4_SCALE: ((1.0,), (1.0,), (1.0,)),
    Operation.MAT4_ROTATE_X: ((0.0,),),
    Operation.MAT4_ROTATE_Y: ((0.0,),),
    Operation.MAT4_ROTATE_Z: ((0.0,),),
    Operation.TRANSFORM: ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    Operation.QUATERNION_FROM_AXIS_ANGLE: ((0.0, 1.0, 0.0), (0.0,)),
}


def default_generator_parameters(operation: Operation) -> tuple[tuple[float, ...], ...]:
    """Return teaching-friendly default parameters for a new generator node."""
    return GENERATOR_DEFAULTS[operation]


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
        source_node_id: str | None = None,
    ) -> None:
        """Create an input or output socket on a node.

        ``source_node_id`` lets an output socket report a different graph
        node id than the one its owning item was created with, which is how
        a single visual node (e.g. Obj Loader) can expose several outputs
        that are really separate graph nodes under the hood.
        """
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
        self.source_node_id = source_node_id
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

    @property
    def graph_node_id(self) -> str:
        """Return the graph node id this socket's values come from or go to."""
        return (
            self.source_node_id
            if self.source_node_id is not None
            else self.node.node_id
        )

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
        self.output_ports: list[PortItem] = []
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
            port = PortItem(self, None, output_colour)
            port.setPos(width, NODE_HEADER_HEIGHT + 28.0)
            self.output_ports.append(port)

    @property
    def output_port(self) -> PortItem | None:
        """Return the node's single output socket, if it has exactly one."""
        return self.output_ports[0] if self.output_ports else None

    def owned_node_ids(self) -> tuple[str, ...]:
        """Return every graph node id this visual item represents.

        Most nodes are a 1:1 wrapper around a single graph node. A few
        (e.g. Obj Loader) expose several outputs backed by separate graph
        nodes and override this to report all of them, so deletion cleans
        up the whole group.
        """
        return (self.node_id,)

    def set_name(self, name: str) -> None:
        """Set the name shown in the node header."""
        clean_name = name.strip()
        if not clean_name or clean_name == self.title:
            return
        self.title = clean_name
        title_width = QFontMetrics(node_title_font()).horizontalAdvance(self.title) + 24
        self._set_width(max(self.width, float(title_width)))
        self.update()
        scene = self.scene()
        if scene is not None:
            scene.mark_modified()

    def _set_width(self, width: float) -> None:
        """Resize the body and keep output sockets and wires aligned."""
        if width == self.width:
            return
        self.prepareGeometryChange()
        self.width = width
        for port in self.output_ports:
            port.setX(width)
            for connection in port.connections:
                connection.update_path()
        self.update()

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Ask for a new name when the node header is double-clicked."""
        header = QRectF(0.0, 0.0, self.width, NODE_HEADER_HEIGHT)
        if header.contains(event.pos()):
            scene = self.scene()
            parent = scene.views()[0] if scene is not None and scene.views() else None
            name, accepted = QInputDialog.getText(
                parent,
                "Rename Node",
                "Name:",
                text=self.title,
            )
            if accepted:
                self.set_name(name)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

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
        """Keep attached wires aligned when the node moves, and flag the scene dirty."""
        if change is QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            ports = [*self.input_ports, *self.output_ports]
            for port in ports:
                for connection in port.connections:
                    connection.update_path()
            scene = self.scene()
            if scene is not None:
                scene.mark_modified()
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
        width = max(190.0, columns * (NUMERIC_EDITOR_WIDTH + 8.0) + 24.0)
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
            spin_box = numeric_spin_box(component)
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


class GeneratorNodeItem(BaseNodeItem):
    """An operation node whose named parameters are typed in, not wired.

    Used for operations such as Look At, Perspective and Transform, whose
    PyNGL parameters are plain Float/Vec3 numbers rather than other
    computed values (see GeneratorNode in math_graph).
    """

    ROW_HEIGHT = 30.0

    def __init__(
        self,
        node_id: str,
        operation: Operation,
        parameters: tuple[tuple[float, ...], ...],
        on_change: Callable[[int, tuple[float, ...]], None],
    ) -> None:
        """Create one labelled spin-box row per named parameter."""
        parameter_names = OPERATION_INPUT_NAMES[operation]
        parameter_types = OPERATION_PARAMETER_TYPES[operation]
        label_font = QFont()
        label_font.setPointSize(9)
        label_width = max(
            QFontMetrics(label_font).horizontalAdvance(name) for name in parameter_names
        )
        max_columns = max(TYPE_SHAPES[math_type][1] for math_type in parameter_types)
        width = max(
            200.0,
            label_width + 16.0 + max_columns * (NUMERIC_EDITOR_WIDTH + 8.0) + 24.0,
        )
        height = NODE_HEADER_HEIGHT + len(parameter_names) * self.ROW_HEIGHT + 20.0
        output_type = GENERATOR_OUTPUT_TYPE[operation]
        super().__init__(
            node_id, operation.value, width, height, (), TYPE_COLOURS[output_type]
        )
        self.operation = operation
        self.parameter_names = parameter_names
        self.on_change = on_change
        self.spin_box_rows: list[list[QDoubleSpinBox]] = []

        editor = QWidget()
        editor.setStyleSheet(
            "QWidget { background: transparent; color: #c2cad7; }"
            "QDoubleSpinBox { background: #151a24; color: #edf1f7; border: 1px solid #46536a; border-radius: 3px; padding: 2px; }"
        )
        layout = QGridLayout(editor)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(4)
        for row_index, (name, components) in enumerate(
            zip(parameter_names, parameters)
        ):
            label = QLabel(name)
            layout.addWidget(label, row_index, 0)
            row_boxes: list[QDoubleSpinBox] = []
            for column_index, component in enumerate(components):
                spin_box = numeric_spin_box(component)
                spin_box.valueChanged.connect(
                    lambda _value, row=row_index: self._row_changed(row)
                )
                layout.addWidget(spin_box, row_index, column_index + 1)
                row_boxes.append(spin_box)
            self.spin_box_rows.append(row_boxes)

        self.proxy = QGraphicsProxyWidget(self)
        self.proxy.setWidget(editor)
        self.proxy.setPos(12.0, NODE_HEADER_HEIGHT + 9.0)
        self.proxy.setZValue(2.0)
        if self.output_port is not None:
            self.output_port.setPos(width, NODE_HEADER_HEIGHT + 20.0)

    def _row_changed(self, row_index: int) -> None:
        """Send one parameter's edited components back to the graph model."""
        components = tuple(box.value() for box in self.spin_box_rows[row_index])
        self.on_change(row_index, components)


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
        title_width = QFontMetrics(node_title_font()).horizontalAdvance(self.title) + 24
        required_width = max(260.0, float(longest_line + 48), float(title_width))
        self._set_width(required_width)
        self.result_text.setTextWidth(self.width - 35.0)
        required_height = max(132.0, 82.0 + self.result_text.boundingRect().height())
        if required_height != self.height:
            self.prepareGeometryChange()
            self.height = required_height
            self.update()

    def value_text(self) -> str:
        """Return the plain result string for tests and accessibility."""
        return self.result_text.toPlainText()


class ObjLoaderNodeItem(BaseNodeItem):
    """Loads a triangulated Obj file and exposes Vertices/Faces/UVs/Normals.

    Each output socket is backed by its own graph node id (see
    ``PortItem.source_node_id``), since one visual box here really wraps
    four separate literal values.
    """

    WIDTH = 220.0

    def __init__(
        self,
        node_id: str,
        array_node_ids: tuple[str, str, str, str],
        on_load_clicked: Callable[["ObjLoaderNodeItem"], None],
    ) -> None:
        """Create the load button and the four labelled output sockets."""
        height = NODE_HEADER_HEIGHT + len(MESH_ARRAY_LABELS) * 26.0 + 76.0
        super().__init__(node_id, "Obj Loader", self.WIDTH, height, (), None)
        self.array_node_ids = array_node_ids
        self.on_load_clicked = on_load_clicked

        for index, label in enumerate(MESH_ARRAY_LABELS):
            port = PortItem(
                self,
                None,
                MESH_ARRAY_COLOURS[label],
                source_node_id=array_node_ids[index],
            )
            port.setPos(self.WIDTH, NODE_HEADER_HEIGHT + 16.0 + index * 26.0)
            port.setToolTip(label)
            self.output_ports.append(port)

        button = QPushButton("Load Obj...")
        button.clicked.connect(lambda _checked=False: self.on_load_clicked(self))
        self.button_proxy = QGraphicsProxyWidget(self)
        self.button_proxy.setWidget(button)
        button_y = NODE_HEADER_HEIGHT + len(MESH_ARRAY_LABELS) * 26.0 + 8.0
        self.button_proxy.setPos(12.0, button_y)
        self.button_proxy.setZValue(2.0)

        self.status_text_item = QGraphicsTextItem("No file loaded", self)
        self.status_text_item.setDefaultTextColor(QColor("#aeb9c9"))
        self.status_text_item.setFont(QFont("Monaco", 8))
        self.status_text_item.setTextWidth(self.WIDTH - 24.0)
        self.status_text_item.setPos(12.0, button_y + 30.0)

    def owned_node_ids(self) -> tuple[str, ...]:
        """Return the four array node ids this loader owns."""
        return self.array_node_ids

    def set_status(self, text: str, is_error: bool = False) -> None:
        """Show the loaded filename and counts, or a load error."""
        colour = QColor("#ff8a8a") if is_error else QColor("#aeb9c9")
        self.status_text_item.setDefaultTextColor(colour)
        self.status_text_item.setPlainText(text)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        """Paint the node body plus the four output socket labels."""
        super().paint(painter, option, widget)
        label_font = QFont()
        label_font.setPointSize(9)
        painter.setFont(label_font)
        painter.setPen(QPen(QColor("#c2cad7")))
        for index, label in enumerate(MESH_ARRAY_LABELS):
            y_position = NODE_HEADER_HEIGHT + 8.0 + index * 26.0
            painter.drawText(
                QRectF(0.0, y_position, self.width - 16.0, 18.0),
                Qt.AlignmentFlag.AlignRight,
                label,
            )


class MeshViewerNodeItem(BaseNodeItem):
    """Merges Vertices/Faces/UVs/Normals/Colour and renders the mesh live.

    Renders in two places at once: a small embedded preview in the node
    body, and an optional pop-out window with the full arcball camera. Both
    read from the same :class:`~.mesh_view.MeshRenderState`.
    """

    WIDTH = 260.0
    PREVIEW_HEIGHT = 170.0

    def __init__(
        self,
        node_id: str,
        on_config_changed: Callable[[], None],
        shading_mode: str = SHADING_SOLID,
        wireframe: bool = False,
    ) -> None:
        """Create the input sockets, shading controls and embedded preview."""
        controls_top = NODE_HEADER_HEIGHT + 28.0 + len(MESH_VIEWER_INPUT_NAMES) * 30.0
        preview_top = controls_top + 48.0
        height = preview_top + self.PREVIEW_HEIGHT + 44.0
        super().__init__(
            node_id, "Mesh Viewer", self.WIDTH, height, MESH_VIEWER_INPUT_NAMES, None
        )
        self.on_config_changed = on_config_changed
        self.render_state = MeshRenderState()
        self.render_state.shading_mode = shading_mode
        self.render_state.wireframe = wireframe
        self._popup: MeshPopupWindow | None = None

        self.shading_combo = QComboBox()
        self.shading_combo.addItems(SHADING_MODES)
        self.shading_combo.setCurrentText(shading_mode)
        self.wireframe_check = QCheckBox("Wireframe")
        self.wireframe_check.setChecked(wireframe)

        controls = QWidget()
        controls.setStyleSheet(
            "QWidget { background: transparent; color: #e8edf5; }"
            "QComboBox { background: #151a24; border: 1px solid #46536a;"
            " border-radius: 3px; padding: 2px; }"
        )
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)
        controls_layout.addWidget(self.shading_combo, 1)
        controls_layout.addWidget(self.wireframe_check)
        self.controls_proxy = QGraphicsProxyWidget(self)
        self.controls_proxy.setWidget(controls)
        self.controls_proxy.setPos(12.0, controls_top)
        self.controls_proxy.setZValue(2.0)

        self.status_text_item = QGraphicsTextItem("", self)
        self.status_text_item.setDefaultTextColor(QColor("#ff8a8a"))
        self.status_text_item.setFont(QFont("Monaco", 8))
        self.status_text_item.setTextWidth(self.WIDTH - 24.0)
        self.status_text_item.setPos(12.0, controls_top + 26.0)

        self.preview = MeshPreviewWidget(self.render_state)
        self.preview.setFixedSize(int(self.WIDTH - 24.0), int(self.PREVIEW_HEIGHT))
        self.preview_proxy = QGraphicsProxyWidget(self)
        self.preview_proxy.setWidget(self.preview)
        self.preview_proxy.setPos(12.0, preview_top)
        self.preview_proxy.setZValue(2.0)

        pop_out_button = QPushButton("Pop Out")
        pop_out_button.clicked.connect(lambda _checked=False: self._pop_out())
        self.pop_out_proxy = QGraphicsProxyWidget(self)
        self.pop_out_proxy.setWidget(pop_out_button)
        self.pop_out_proxy.setPos(12.0, preview_top + self.PREVIEW_HEIGHT + 8.0)
        self.pop_out_proxy.setZValue(2.0)

        self.shading_combo.currentTextChanged.connect(self._shading_changed)
        self.wireframe_check.toggled.connect(self._wireframe_toggled)

    def _shading_changed(self, mode: str) -> None:
        """Switch shading mode and refresh the existing mesh."""
        self.render_state.set_shading_mode(mode)
        self._refresh_views()
        self.on_config_changed()

    def _wireframe_toggled(self, checked: bool) -> None:
        """Toggle wireframe and refresh the existing mesh."""
        self.render_state.set_wireframe(checked)
        self._refresh_views()
        self.on_config_changed()

    def _pop_out(self) -> None:
        """Open (or focus) the standalone arcball-camera viewer window."""
        if self._popup is None:
            self._popup = MeshPopupWindow(self.render_state, "Mesh Viewer")
        if not self._popup.isVisible():
            self._popup.show()
        else:
            self._popup.requestActivate()

    def close_popup(self) -> None:
        """Close and release this node's standalone viewer, if it has one."""
        if self._popup is None:
            return
        self._popup.close()
        self._popup = None

    def _refresh_views(self) -> None:
        """Repaint the embedded preview and the popup window, if it's open."""
        self.preview.update()
        if self._popup is not None:
            self._popup.update()

    def set_mesh_inputs(self, mesh_inputs: "MeshViewerInputs") -> None:
        """Clear any error status and render the given merged mesh."""
        self.status_text_item.setPlainText("")
        self.render_state.set_mesh(mesh_inputs)
        self._refresh_views()

    def show_error(self, message: str) -> None:
        """Clear the rendered mesh and show a graph error message."""
        self.status_text_item.setDefaultTextColor(QColor("#ff8a8a"))
        self.status_text_item.setPlainText(message)
        self.render_state.set_mesh(None)
        self._refresh_views()
