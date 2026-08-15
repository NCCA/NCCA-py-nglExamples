"""Node-creation UI: the searchable Tab menu and the side palette."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from math_graph import GENERATOR_OPERATIONS, MathType, Operation
from node_visuals import catalogue_node_style, node_icon
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

if TYPE_CHECKING:
    from canvas import MathNodeScene, MathNodeView

MATH_OPERATIONS = (
    Operation.ADD,
    Operation.SUBTRACT,
    Operation.MULTIPLY,
    Operation.MATRIX_MULTIPLY,
    Operation.DOT,
    Operation.CROSS,
    Operation.NORMALISE,
    Operation.TRANSPOSE,
    Operation.INVERSE,
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
    Operation.MAT4_TO_MAT3,
    Operation.TRANSFORM,
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
MESH_OPERATIONS = (
    Operation.TRANSFORM_VERTICES,
    Operation.TRANSFORM_NORMALS,
)

CatalogueEntry = tuple[str, Callable[["MathNodeScene", "QPointF | None"], object]]
CatalogueSection = tuple[str, list[CatalogueEntry]]


def _value_catalogue_entries() -> list[CatalogueEntry]:
    """Return one catalogue entry per creatable value type."""
    return [
        (
            math_type.value,
            lambda canvas, position, value=math_type: canvas.add_value_node(
                value, position
            ),
        )
        for math_type in MathType
    ]


def _operation_catalogue_entries(
    operations: tuple[Operation, ...],
) -> list[CatalogueEntry]:
    """Return one catalogue entry per operation in the given group.

    Operations in GENERATOR_OPERATIONS (Look At, Perspective, Transform,
    ...) route through add_generator_node, since they take typed-in
    Float/Vec3 parameters rather than wired inputs; everything else keeps
    add_operation_node's wired sockets.
    """
    entries: list[CatalogueEntry] = []
    for operation in operations:
        if operation in GENERATOR_OPERATIONS:
            entries.append(
                (
                    operation.value,
                    lambda canvas, position, value=operation: canvas.add_generator_node(
                        value, position
                    ),
                )
            )
        else:
            entries.append(
                (
                    operation.value,
                    lambda canvas, position, value=operation: canvas.add_operation_node(
                        value, position
                    ),
                )
            )
    return entries


NODE_CATALOGUE: tuple[CatalogueSection, ...] = (
    ("Values", _value_catalogue_entries()),
    ("Maths", _operation_catalogue_entries(MATH_OPERATIONS)),
    ("Mat4", _operation_catalogue_entries(MAT4_OPERATIONS)),
    ("Quaternion", _operation_catalogue_entries(QUATERNION_OPERATIONS)),
    (
        "Mesh",
        [
            (
                "Obj Loader",
                lambda canvas, position: canvas.add_obj_loader_node(position),
            ),
            *_operation_catalogue_entries(MESH_OPERATIONS),
            (
                "Mesh Viewer",
                lambda canvas, position: canvas.add_mesh_viewer_node(position),
            ),
        ],
    ),
    (
        "Result",
        [("Output", lambda canvas, position: canvas.add_output_node(position))],
    ),
)


class NodeCreationMenu(QMenu):
    """Searchable menu which creates nodes at a chosen scene position."""

    def __init__(
        self,
        canvas: MathNodeScene,
        parent: QWidget | None = None,
    ) -> None:
        """Build a node menu from the same catalogue as the side palette."""
        super().__init__(parent)
        self.canvas = canvas
        self.scene_position = QPointF()
        self.creation_actions: list[QAction] = []
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search nodes...")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._filter_actions)
        self.search_edit.returnPressed.connect(self._create_first_visible_node)
        search_action = QWidgetAction(self)
        search_action.setDefaultWidget(self.search_edit)
        self.addAction(search_action)

        for title, entries in NODE_CATALOGUE:
            self._add_section(title, entries)

    def _add_section(self, title: str, entries: list[CatalogueEntry]) -> None:
        """Add a labelled section to the creation menu."""
        self.addSection(title)
        for label, factory in entries:
            action = self.addAction(label)
            action.setIcon(node_icon(catalogue_node_style(label)))
            action.triggered.connect(
                lambda _checked=False, create_node=factory: create_node(
                    self.canvas, self.scene_position
                )
            )
            self.creation_actions.append(action)

    def _filter_actions(self, search_text: str) -> None:
        """Show node actions whose labels contain the entered words."""
        words = search_text.casefold().split()
        for action in self.creation_actions:
            label = action.text().casefold()
            action.setVisible(all(word in label for word in words))

    def _create_first_visible_node(self) -> None:
        """Create the first filtered node when Return is pressed."""
        for action in self.creation_actions:
            if action.isVisible():
                action.trigger()
                self.close()
                return

    def open_at(self, scene_position: QPointF, global_position: QPoint) -> None:
        """Show the menu and create its selected node at the scene position."""
        self.scene_position = QPointF(scene_position)
        self.search_edit.clear()
        self.popup(global_position)
        self.search_edit.setFocus(Qt.FocusReason.PopupFocusReason)


class NodePalette(QWidget):
    """Buttons used to add value, operation and output nodes."""

    def __init__(
        self,
        canvas: MathNodeScene,
        view: MathNodeView,
        parent: QWidget | None = None,
    ) -> None:
        """Build the node palette for a canvas and its view."""
        super().__init__(parent)
        self.canvas = canvas
        self.view = view
        self.setFixedWidth(240)
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
            "Click to add nodes, or press Tab over the canvas and search. Drag from an output socket to an input socket to connect them. Press H to frame all nodes."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #aeb9c9;")
        layout.addWidget(help_label)

        for title, entries in NODE_CATALOGUE:
            self._add_group(layout, title, entries)
        layout.addStretch(1)

        frame_all_button = QPushButton("Frame All")
        frame_all_button.clicked.connect(lambda _checked=False: self.view.frame_all())
        layout.addWidget(frame_all_button)

    def _add_group(
        self,
        parent_layout: QVBoxLayout,
        title: str,
        entries: list[CatalogueEntry],
    ) -> None:
        """Add a titled group of palette buttons."""
        group = QGroupBox(title)
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(7, 10, 7, 7)
        group_layout.setSpacing(4)
        for label, factory in entries:
            button = QPushButton(label)
            button.setIcon(node_icon(catalogue_node_style(label)))
            button.clicked.connect(
                lambda _checked=False, create_node=factory: create_node(
                    self.canvas, None
                )
            )
            group_layout.addWidget(button)
        parent_layout.addWidget(group)
