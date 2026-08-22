"""PySide6 node editor for experimenting with PyNGL maths types."""

from __future__ import annotations

import contextlib
import io
import json
import traceback
from pathlib import Path

from canvas import MathNodeScene, MathNodeView
from graphics_items import (
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
from math_graph import (
    OPERATION_ARITY,
    OPERATION_INPUT_NAMES,
    TYPE_SHAPES,
    GraphError,
    MathGraph,
    MathType,
    Operation,
    format_value,
)
from node_visuals import (
    NodeVisualStyle,
    catalogue_node_style,
    node_icon,
    operation_node_style,
    value_node_style,
)
from palette import (
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
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from python_highlighter import PythonHighlighter

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
    "NodeVisualStyle",
    "ObjLoaderNodeItem",
    "Operation",
    "OperationNodeItem",
    "OutputNodeItem",
    "PortItem",
    "ValueNodeItem",
    "default_components",
    "catalogue_node_style",
    "format_value",
    "node_icon",
    "node_title_font",
    "operation_node_style",
    "value_node_style",
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
    return QSettings("NCCA", "MathNodeEditor")


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
        self.palette_scroll.setFixedWidth(260)
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
        self._build_code_dock()
        self.statusBar().showMessage(
            "Press Tab to add a node; edit values and wire nodes together"
        )
        self._build_file_menu()
        self.canvas.modifiedChanged.connect(lambda _modified: self._update_title())
        self.canvas.graphChanged.connect(self._update_code_view)
        if load_example:
            self._load_startup_graph()
            QTimer.singleShot(0, self.view.frame_all)
        self._update_title()
        self._update_code_view()

    def _build_code_dock(self) -> None:
        """Create the hidden dock which displays generated PyNGL Python."""
        self.code_dock = QDockWidget("Python Code", self)
        self.code_dock.setObjectName("pythonCodeDock")
        self.code_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.code_editor = QPlainTextEdit(self.code_dock)
        self.code_editor.setReadOnly(True)
        self.code_editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.code_editor.setFixedWidth(320)
        self.code_editor.setStyleSheet(
            "QPlainTextEdit {"
            " background: #111620; color: #f8f8f2; border: 0;"
            " selection-background-color: #44475a;"
            "}"
        )
        self.code_output = QPlainTextEdit(self.code_dock)
        self.code_output.setObjectName("pythonCodeOutput")
        self.code_output.setReadOnly(True)
        self.code_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.code_output.setFixedHeight(160)
        self.code_output.setStyleSheet(
            "QPlainTextEdit {"
            " background: #111620; color: #f8f8f2; border-top: 1px solid #44475a;"
            " selection-background-color: #44475a;"
            "}"
        )
        self.copy_code_button = QPushButton("Copy", self.code_dock)
        self.copy_code_button.clicked.connect(self._copy_code)
        self.save_code_button = QPushButton("Save…", self.code_dock)
        self.save_code_button.clicked.connect(self._save_code)
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(6, 6, 6, 6)
        button_layout.addStretch()
        button_layout.addWidget(self.copy_code_button)
        button_layout.addWidget(self.save_code_button)
        dock_content = QWidget(self.code_dock)
        dock_layout = QVBoxLayout(dock_content)
        dock_layout.setContentsMargins(0, 0, 0, 0)
        dock_layout.setSpacing(0)
        dock_layout.addWidget(self.code_editor, 1)
        dock_layout.addWidget(self.code_output)
        dock_layout.addLayout(button_layout)
        self.code_dock.setWidget(dock_content)
        self.code_highlighter = PythonHighlighter(self.code_editor.document())
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.code_dock)

    def _copy_code(self) -> None:
        """Copy the complete generated Python script to the clipboard."""
        QApplication.clipboard().setText(self.code_editor.toPlainText())

    def _save_code(self) -> None:
        """Save the complete generated Python script to a chosen file."""
        path, _name_filter = QFileDialog.getSaveFileName(
            self, "Save Python Code", "", "Python Files (*.py)"
        )
        if not path:
            return
        try:
            Path(path).write_text(self.code_editor.toPlainText(), encoding="utf-8")
        except OSError as error:
            QMessageBox.warning(
                self, "Save Python Code", f"Could not save code: {error}"
            )

    def _update_code_view(self) -> None:
        """Regenerate the Python preview for every regular Output node."""
        output_ids = [
            node_id
            for node_id, node in self.canvas.nodes.items()
            if isinstance(node, OutputNodeItem)
        ]
        code = self.canvas.graph.generate_python(output_ids)
        self.code_editor.setPlainText(code)
        self.code_output.setPlainText(self._run_generated_code(code))

    @staticmethod
    def _run_generated_code(code: str) -> str:
        """Run generated Python in this process and return its printed output."""
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                exec(code, {})
        except Exception:
            return traceback.format_exc()
        return output.getvalue()

    def _build_view_menu(self) -> None:
        """Build the View menu action controlling the generated-code dock."""
        view_menu = self.menuBar().addMenu("&View")
        self.action_code_view = QAction("&Code View", self, checkable=True)
        self.action_code_view.toggled.connect(self.code_dock.setVisible)
        self.code_dock.visibilityChanged.connect(self.action_code_view.setChecked)
        self.action_code_view.setChecked(True)
        view_menu.addAction(self.action_code_view)

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
        self._build_view_menu()

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
        snapshot = self.canvas.to_dict()
        was_modified = self.canvas.modified
        try:
            self.canvas.load_from_file(path)
        except _LOAD_ERRORS as error:
            # from_dict clears the graph before rebuilding it, so a schema
            # error partway through leaves a half-built graph rather than the
            # one the user actually had open. Restore it from the snapshot
            # taken above instead of leaving that wreckage sitting under a
            # clean (non-dirty) title bar, ready for Ctrl+S to overwrite the
            # good file on disk with it.
            self.canvas.from_dict(snapshot)
            if was_modified:
                self.canvas.mark_modified()
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
        self.canvas.mark_clean()
        self._update_title()
        return True

    def closeEvent(self, event: QCloseEvent) -> None:
        """Confirm discarding unsaved changes, then persist the window geometry."""
        if not self._confirm_discard_changes():
            event.ignore()
            return
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)
