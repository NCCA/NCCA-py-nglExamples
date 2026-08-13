"""PySide6 node editor for experimenting with PyNGL maths types."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QWidget,
)

from .canvas import MathNodeScene, MathNodeView
from .graphics_items import (
    GENERIC_PORT_COLOUR,
    NODE_HEADER_HEIGHT,
    PORT_RADIUS,
    TYPE_COLOURS,
    BaseNodeItem,
    ConnectionItem,
    OperationNodeItem,
    OutputNodeItem,
    PortItem,
    ValueNodeItem,
    default_components,
    node_title_font,
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
from .palette import (
    MAT4_OPERATIONS,
    MATH_OPERATIONS,
    NODE_CATALOGUE,
    QUATERNION_OPERATIONS,
    CatalogueEntry,
    CatalogueSection,
    NodeCreationMenu,
    NodePalette,
)

__all__ = [
    "GENERIC_PORT_COLOUR",
    "MAT4_OPERATIONS",
    "MATH_OPERATIONS",
    "NODE_CATALOGUE",
    "NODE_HEADER_HEIGHT",
    "OPERATION_ARITY",
    "OPERATION_INPUT_NAMES",
    "PORT_RADIUS",
    "QUATERNION_OPERATIONS",
    "TYPE_COLOURS",
    "TYPE_SHAPES",
    "BaseNodeItem",
    "CatalogueEntry",
    "CatalogueSection",
    "ConnectionItem",
    "GraphError",
    "MathGraph",
    "MathNodeScene",
    "MathNodeView",
    "MathNodeWindow",
    "MathType",
    "NodeCreationMenu",
    "NodePalette",
    "Operation",
    "OperationNodeItem",
    "OutputNodeItem",
    "PortItem",
    "ValueNodeItem",
    "default_components",
    "format_value",
    "main",
    "node_title_font",
]


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
        self.palette = NodePalette(self.canvas, self.view, self)
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
            "Press Tab to add a node; edit values and wire nodes together"
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
