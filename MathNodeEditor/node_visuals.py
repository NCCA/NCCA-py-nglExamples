"""Icons and colours shared by the node canvas and creation controls."""

from __future__ import annotations

from dataclasses import dataclass

from math_graph import MathType, Operation
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap


@dataclass(frozen=True)
class NodeVisualStyle:
    """The symbol and header colour used to identify a node.

    Attributes
    ----------
        icon_symbol : str
            short symbol painted inside the node icon
        header_colour : QColor
            colour used for the node header and catalogue icon
    """

    icon_symbol: str
    header_colour: QColor


VALUE_NODE_STYLES: dict[MathType, NodeVisualStyle] = {
    MathType.FLOAT: NodeVisualStyle("ƒ", QColor("#596579")),
    MathType.VEC2: NodeVisualStyle("v₂", QColor("#287f7c")),
    MathType.VEC3: NodeVisualStyle("v₃", QColor("#39834f")),
    MathType.VEC4: NodeVisualStyle("v₄", QColor("#326da3")),
    MathType.MAT2: NodeVisualStyle("m₂", QColor("#92712e")),
    MathType.MAT3: NodeVisualStyle("m₃", QColor("#9b593a")),
    MathType.MAT4: NodeVisualStyle("m₄", QColor("#98445f")),
    MathType.QUATERNION: NodeVisualStyle("q", QColor("#7052a0")),
}

MATHS_HEADER_COLOUR = QColor("#466a9f")
MAT4_HEADER_COLOUR = QColor("#936737")
QUATERNION_HEADER_COLOUR = QColor("#7052a0")
MESH_HEADER_COLOUR = QColor("#307b72")
OUTPUT_HEADER_COLOUR = QColor("#98515b")

OPERATION_NODE_STYLES: dict[Operation, NodeVisualStyle] = {
    Operation.ADD: NodeVisualStyle("+", MATHS_HEADER_COLOUR),
    Operation.SUBTRACT: NodeVisualStyle("−", MATHS_HEADER_COLOUR),
    Operation.MULTIPLY: NodeVisualStyle("×", MATHS_HEADER_COLOUR),
    Operation.MATRIX_MULTIPLY: NodeVisualStyle("@", MATHS_HEADER_COLOUR),
    Operation.DOT: NodeVisualStyle("·", MATHS_HEADER_COLOUR),
    Operation.CROSS: NodeVisualStyle("⨯", MATHS_HEADER_COLOUR),
    Operation.NORMALISE: NodeVisualStyle("→", MATHS_HEADER_COLOUR),
    Operation.TRANSPOSE: NodeVisualStyle("T", MATHS_HEADER_COLOUR),
    Operation.INVERSE: NodeVisualStyle("⁻¹", MATHS_HEADER_COLOUR),
    Operation.MAT4_TO_MAT3: NodeVisualStyle("m₃", MAT4_HEADER_COLOUR),
    Operation.LOOK_AT: NodeVisualStyle("◉", MAT4_HEADER_COLOUR),
    Operation.PERSPECTIVE: NodeVisualStyle("△", MAT4_HEADER_COLOUR),
    Operation.ORTHO: NodeVisualStyle("□", MAT4_HEADER_COLOUR),
    Operation.FRUSTUM: NodeVisualStyle("◇", MAT4_HEADER_COLOUR),
    Operation.MAT4_TRANSLATE: NodeVisualStyle("↗", MAT4_HEADER_COLOUR),
    Operation.MAT4_SCALE: NodeVisualStyle("↔", MAT4_HEADER_COLOUR),
    Operation.MAT4_ROTATE_X: NodeVisualStyle("↻x", MAT4_HEADER_COLOUR),
    Operation.MAT4_ROTATE_Y: NodeVisualStyle("↻y", MAT4_HEADER_COLOUR),
    Operation.MAT4_ROTATE_Z: NodeVisualStyle("↻z", MAT4_HEADER_COLOUR),
    Operation.TRANSFORM: NodeVisualStyle("TRS", MAT4_HEADER_COLOUR),
    Operation.QUATERNION_FROM_AXIS_ANGLE: NodeVisualStyle(
        "θ", QUATERNION_HEADER_COLOUR
    ),
    Operation.QUATERNION_PRODUCT: NodeVisualStyle("q×", QUATERNION_HEADER_COLOUR),
    Operation.QUATERNION_ROTATE_VECTOR: NodeVisualStyle("q↻", QUATERNION_HEADER_COLOUR),
    Operation.QUATERNION_TO_MAT4: NodeVisualStyle("m₄", QUATERNION_HEADER_COLOUR),
    Operation.MAT4_TO_QUATERNION: NodeVisualStyle("q", QUATERNION_HEADER_COLOUR),
    Operation.QUATERNION_SLERP: NodeVisualStyle("⌢", QUATERNION_HEADER_COLOUR),
    Operation.QUATERNION_CONJUGATE: NodeVisualStyle("q̅", QUATERNION_HEADER_COLOUR),
    Operation.QUATERNION_INVERSE: NodeVisualStyle("q⁻¹", QUATERNION_HEADER_COLOUR),
    Operation.TRANSFORM_VERTICES: NodeVisualStyle("◇↗", MESH_HEADER_COLOUR),
    Operation.TRANSFORM_NORMALS: NodeVisualStyle("⊥↗", MESH_HEADER_COLOUR),
}

OBJ_LOADER_STYLE = NodeVisualStyle("⇥", MESH_HEADER_COLOUR)
MESH_VIEWER_STYLE = NodeVisualStyle("▦", MESH_HEADER_COLOUR)
OUTPUT_NODE_STYLE = NodeVisualStyle("=", OUTPUT_HEADER_COLOUR)


def value_node_style(math_type: MathType) -> NodeVisualStyle:
    """Return the visual style for a maths value node."""
    return VALUE_NODE_STYLES[math_type]


def operation_node_style(operation: Operation) -> NodeVisualStyle:
    """Return the visual style for an operation or generator node."""
    return OPERATION_NODE_STYLES[operation]


def catalogue_node_style(label: str) -> NodeVisualStyle:
    """Return the style belonging to a node catalogue label."""
    for math_type, style in VALUE_NODE_STYLES.items():
        if label == math_type.value:
            return style
    for operation, style in OPERATION_NODE_STYLES.items():
        if label == operation.value:
            return style
    if label == "Obj Loader":
        return OBJ_LOADER_STYLE
    if label == "Mesh Viewer":
        return MESH_VIEWER_STYLE
    if label == "Output":
        return OUTPUT_NODE_STYLE
    raise KeyError(label)


def node_icon(style: NodeVisualStyle, size: int = 20) -> QIcon:
    """Paint a small scalable Qt icon for a node style."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(style.header_colour)
    painter.drawRoundedRect(QRectF(0.5, 0.5, size - 1.0, size - 1.0), 4.0, 4.0)
    font = QFont()
    font.setBold(True)
    font.setPixelSize(8 if len(style.icon_symbol) > 2 else 11)
    painter.setFont(font)
    painter.setPen(QColor("#ffffff"))
    painter.drawText(
        QRectF(1.0, 0.0, size - 2.0, size - 1.0),
        Qt.AlignmentFlag.AlignCenter,
        style.icon_symbol,
    )
    painter.end()
    return QIcon(pixmap)
