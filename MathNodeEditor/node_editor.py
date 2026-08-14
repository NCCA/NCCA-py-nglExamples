"""PySide6 node editor for experimenting with PyNGL maths types."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
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
    GeneratorNodeItem,
    MeshViewerNodeItem,
    ObjLoaderNodeItem,
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
    MESH_OPERATIONS,
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
    "MESH_OPERATIONS",
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
    "GeneratorNodeItem",
    "GraphError",
    "MathGraph",
    "MathNodeScene",
    "MathNodeView",
    "MathNodeWindow",
    "MathType",
    "MeshViewerNodeItem",
    "NodeCreationMenu",
    "NodePalette",
    "ObjLoaderNodeItem",
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

WINDOW_TITLE = "PyNGL Maths Node Editor"

_LOAD_ERRORS = (
    OSError,
    json.JSONDecodeError,
    GraphError,
    KeyError,
    TypeError,
    ValueError,
    IndexError,
)


def _default_settings() -> QSettings:
    """Return the QSettings store used when a window isn't given one explicitly."""
    return QSettings()


class MathNodeWindow(QMainWindow):
    """Main window containing the palette and node graph canvas."""

    def __init__(
        self,
        load_example: bool = True,
        settings: QSettings | None = None,
    ) -> None:
        """Create the editor, restore its settings, and load a starting graph."""
        super().__init__()
        self.settings = settings if settings is not None else _default_settings()
        self.current_file: Path | None = None
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
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
        self._build_file_menu()
        self.canvas.modifiedChanged.connect(lambda _modified: self._update_title())
        if load_example:
            self._load_startup_graph()
            self.view.centerOn(0.0, 0.0)
        self._update_title()

    def _build_file_menu(self) -> None:
        """Build the File menu's New/Open/Save/Save As/Quit actions."""
        file_menu = self.menuBar().addMenu("&File")

        self.action_new = QAction("&New", self)
        self.action_new.setShortcut(QKeySequence.StandardKey.New)
        self.action_new.triggered.connect(self._new_graph)
        file_menu.addAction(self.action_new)

        self.action_open = QAction("&Open...", self)
        self.action_open.setShortcut(QKeySequence.StandardKey.Open)
        self.action_open.triggered.connect(self._open_graph)
        file_menu.addAction(self.action_open)

        self.action_save = QAction("&Save", self)
        self.action_save.setShortcut(QKeySequence.StandardKey.Save)
        self.action_save.triggered.connect(self._save_graph)
        file_menu.addAction(self.action_save)

        self.action_save_as = QAction("Save &As...", self)
        self.action_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.action_save_as.triggered.connect(self._save_graph_as)
        file_menu.addAction(self.action_save_as)

        file_menu.addSeparator()

        self.action_quit = QAction("&Quit", self)
        self.action_quit.setShortcut(QKeySequence.StandardKey.Quit)
        self.action_quit.triggered.connect(self.close)
        file_menu.addAction(self.action_quit)

    def _update_title(self) -> None:
        """Show the current file name and an unsaved-changes marker in the title bar."""
        name = self.current_file.name if self.current_file else "Untitled"
        star = "*" if self.canvas.modified else ""
        self.setWindowTitle(f"{WINDOW_TITLE} — {name}{star}")

    def _confirm_discard_changes(self) -> bool:
        """Ask to save unsaved changes; return whether it's safe to proceed."""
        if not self.canvas.modified:
            return True
        response = QMessageBox.question(
            self,
            WINDOW_TITLE,
            "Save changes to the current graph before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if response == QMessageBox.StandardButton.Cancel:
            return False
        if response == QMessageBox.StandardButton.Discard:
            return True
        self._save_graph()
        return not self.canvas.modified

    def _load_startup_graph(self) -> None:
        """Reopen the last file used, falling back to the bundled example."""
        recent_file = self.settings.value("recentFile", "", type=str)
        if recent_file and Path(recent_file).is_file():
            if self._open_path(Path(recent_file)):
                return
        self.canvas.load_example()
        self.current_file = None

    def _new_graph(self) -> None:
        """Discard the current graph and start a blank one."""
        if not self._confirm_discard_changes():
            return
        self.canvas.clear_graph()
        self.current_file = None
        self._update_title()

    def _open_graph(self) -> None:
        """Prompt for a path and replace the current graph with its contents."""
        if not self._confirm_discard_changes():
            return
        path, _name_filter = QFileDialog.getOpenFileName(
            self, "Open Graph", "", "JSON Files (*.json)"
        )
        if not path:
            return
        self._open_path(Path(path))

    def _open_path(self, path: Path) -> bool:
        """Load a graph file, reporting failure instead of raising. Return success."""
        try:
            self.canvas.load_from_file(path)
        except _LOAD_ERRORS as error:
            QMessageBox.warning(self, "Open Graph", f"Could not open graph: {error}")
            return False
        self.current_file = path
        self.settings.setValue("recentFile", str(path))
        self._update_title()
        return True

    def _save_graph(self) -> None:
        """Save to the current file, or prompt for one if there isn't one yet."""
        if self.current_file is None:
            self._save_graph_as()
            return
        self._save_path(self.current_file)

    def _save_graph_as(self) -> None:
        """Prompt for a path and save the current graph to it."""
        path, _name_filter = QFileDialog.getSaveFileName(
            self, "Save Graph As", "", "JSON Files (*.json)"
        )
        if not path:
            return
        self._save_path(Path(path))

    def _save_path(self, path: Path) -> bool:
        """Write the current graph to a path, reporting failure. Return success."""
        try:
            self.canvas.save_to_file(path)
        except OSError as error:
            QMessageBox.warning(self, "Save Graph", f"Could not save graph: {error}")
            return False
        self.current_file = path
        self.settings.setValue("recentFile", str(path))
        self.canvas.modified = False
        self._update_title()
        return True

    def closeEvent(self, event: QCloseEvent) -> None:
        """Confirm discarding unsaved changes, then persist the window geometry."""
        if not self._confirm_discard_changes():
            event.ignore()
            return
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)


def main() -> int:
    """Run the maths node editor application."""
    application = QApplication.instance()
    if application is None:
        surface_format = QSurfaceFormat()
        surface_format.setSamples(4)
        surface_format.setMajorVersion(4)
        surface_format.setMinorVersion(1)
        surface_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        surface_format.setDepthBufferSize(24)
        QSurfaceFormat.setDefaultFormat(surface_format)
        # The Mesh Viewer node's embedded preview and its pop-out window are
        # two separate top-level GL surfaces; without this they don't share
        # a context and ShaderLib's program ids from one are invalid in the
        # other.
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
        application = QApplication(sys.argv)
    # QSettings derives its storage path from the organization/application
    # name, so this has to happen before the first QSettings() is created
    # (inside MathNodeWindow.__init__) or it falls back to an unnamed store.
    application.setOrganizationName("NCCA")
    application.setApplicationName("MathNodeEditor")
    window = MathNodeWindow(load_example=True)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
