#!/usr/bin/env -S uv run --script
"""
A PySide6 GUI application to browse and run py-ngl examples.

This script scans a directory for executable Python files, treating each as a demo.
It presents a list of these demos in a GUI. When a demo is selected, it displays
any associated README.md file and a preview image (PNG). A "Run Demo" button
allows the user to execute the selected demo script in a separate process.
"""

import os
import shlex
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import ncca.ngl
from PySide6.QtCore import QFile, Qt
from PySide6.QtGui import QIntValidator, QKeyEvent, QPixmap
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

# Default smoketest run time in milliseconds, matching the default used by
# the demos' own `--smoketest` argparse option.
DEFAULT_SMOKETEST_MS = 200


@dataclass
class Demo:
    """A data class to hold information about a single demo."""

    button_name: str
    root_path: str
    app_full_path: str


class DemoRunner(QMainWindow):
    """
    The main window for the demo runner application.

    This class discovers executable demos, builds the UI, and handles
    user interaction to select and run demos.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"PyNGL Version : {ncca.ngl.__version__} Demos")

        # These will be populated by methods called below
        self.executables: list[Demo] = []
        self.active_demo: Demo | None = None
        self.button_group: QButtonGroup

        self._find_executables()
        self.load_ui()

        # Create a scroll area for the demo list to handle many demos
        self.demo_scroll_area = QScrollArea()
        scroll_area = self.demo_scroll_area
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # A search box sits above the scroll area to filter demos by name.
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search demos...")
        self.search_box.textChanged.connect(self.filter_demos)

        search_container = QWidget()
        search_layout = QVBoxLayout(search_container)
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.addWidget(self.search_box)
        search_layout.addWidget(scroll_area)

        # The .ui file has a placeholder QWidget called 'demo_list'.
        # We replace this placeholder with our search box + scroll area.
        parent_layout = self.demo_list.parentWidget().layout()
        parent_layout.replaceWidget(self.demo_list, search_container)
        # The original demo_list widget is now managed by the scroll area
        scroll_area.setWidget(self.demo_list)

        # Now, populate the demo_list's layout with buttons for each found executable
        layout = self.demo_list.layout()
        self.button_group = QButtonGroup(self)
        # Ensure only one demo button can be checked at a time
        self.button_group.buttonToggled.connect(self.on_demo_toggled)

        for demo in self.executables:
            button = QPushButton(demo.button_name)
            button.setObjectName(demo.button_name)
            button.setCheckable(True)
            button.setFlat(True)  # Flat style for a cleaner list look
            button.setStyleSheet("QPushButton { text-align: left; padding: 5px; }")
            # Keep keyboard focus on the main window so its keyPressEvent
            # (arrow-key navigation) isn't hijacked by native button focus
            # traversal, which stops our custom handler from firing again.
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.button_group.addButton(button)
            layout.addWidget(button)

        # Select the first demo in the list by default
        if self.button_group.buttons():
            self.button_group.buttons()[0].setChecked(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def _find_executables(self) -> None:
        """
        Recursively finds all executable Python scripts to be treated as demos.

        It walks the current working directory, ignoring common temporary/VCS
        directories. Any file ending in .py that is marked as executable is
        added to the list of demos.
        """
        root = Path.cwd()
        # Define directories and file stems to ignore during the search
        exclude_dirs = {
            ".venv",
            ".git",
            ".worktrees",
            "__pycache__",
            ".ruff_cache",
            ".mypy_cache",
            ".pytest_cache",
            ".ropeproject",
        }
        # Exclude this script itself from the demo list
        exclude_stems = {Path(__file__).stem}

        def walk(current_root: Path) -> Iterator[Path]:
            """A generator that recursively walks a directory yielding executable python files."""
            for p in current_root.iterdir():
                if p.is_dir():
                    # If the directory is in our exclude list, skip it
                    if p.name in exclude_dirs:
                        continue
                    # Otherwise, recurse into it
                    yield from walk(p)
                elif p.suffix == ".py":
                    # Skip this script itself
                    if p.stem in exclude_stems:
                        continue
                    # Check if the file has execute permissions
                    if os.access(p, os.X_OK):
                        yield p

        scripts = [p for p in walk(root) if p.stem != "RunDemos"]

        # Folders with more than one executable script would otherwise produce
        # several buttons with the same label (the folder name); disambiguate
        # those with the script filename appended.
        counts: dict[str, int] = {}
        for p in scripts:
            counts[p.parent.name] = counts.get(p.parent.name, 0) + 1

        self.executables = []
        for p in scripts:
            folder_name = p.parent.name
            if counts[folder_name] > 1:
                button_name = f"{folder_name} — {p.name}"
            else:
                button_name = folder_name
            demo = Demo(
                button_name=button_name,
                root_path=str(p.parent),
                app_full_path=str(p),
            )
            self.executables.append(demo)

    def load_ui(self) -> None:
        """
        Load the UI from the 'DemoUI.ui' file.

        This method uses QUiLoader to load the UI definition and sets up
        the central widget. It also dynamically assigns named widgets from the
        .ui file as attributes of `self` for easy access.
        """
        loader = QUiLoader()
        ui_file = QFile("DemoUI.ui")
        ui_file.open(QFile.OpenModeFlag.ReadOnly)
        # Load the UI. `self` is passed as the parent.
        loaded_ui = loader.load(ui_file, self)
        self.setCentralWidget(loaded_ui)

        # Dynamically find all QWidgets with an objectName in the loaded UI
        # and assign them as attributes of `self` (e.g., self.run_demo).
        for child in loaded_ui.findChildren(QWidget):
            name = child.objectName()
            if name:
                setattr(self, name, child)
        ui_file.close()
        # Connect the 'Run Demo' button's clicked signal to its handler
        self.run_demo.clicked.connect(self.on_demo_clicked)
        # Only positive integer millisecond values are accepted
        self.smoketest_time.setValidator(QIntValidator(1, 600_000, self))
        self.smoketest_time.setText(str(DEFAULT_SMOKETEST_MS))
        self.smoketest_all.clicked.connect(self.on_smoketest_all_clicked)

    def on_demo_clicked(self) -> None:
        """
        Handles the 'Run Demo' button click.

        This runs the currently active demo's main script in a new process.
        """
        if self.active_demo:
            # Run the selected demo script.
            # The script is run in its own directory to ensure correct relative path handling.

            subprocess.run(
                [self.active_demo.app_full_path],
                shell=True,
                cwd=self.active_demo.root_path,
            )

    def on_smoketest_all_clicked(self) -> None:
        """
        Runs every discovered demo with `--smoketest <ms>` in turn.

        Each demo is expected to exit itself after the given number of
        milliseconds. Progress and a final pass/fail summary are shown in
        the description pane; failures include demos that exit non-zero or
        that fail to exit within a grace period after the smoketest time.
        """
        try:
            smoketest_ms = int(self.smoketest_time.text())
        except ValueError:
            smoketest_ms = DEFAULT_SMOKETEST_MS

        self.smoketest_all.setEnabled(False)
        total = len(self.executables)
        failures: list[str] = []
        try:
            for index, demo in enumerate(self.executables, start=1):
                self.demo_text.setPlainText(
                    f"Smoketesting {index}/{total}: {demo.button_name}..."
                )
                QApplication.processEvents()
                command = (
                    f"{shlex.quote(demo.app_full_path)} --smoketest {smoketest_ms}"
                )
                try:
                    result = subprocess.run(
                        command,
                        shell=True,
                        cwd=demo.root_path,
                        timeout=(smoketest_ms / 1000.0) + 30.0,
                    )
                    if result.returncode != 0:
                        failures.append(demo.button_name)
                except subprocess.TimeoutExpired:
                    failures.append(f"{demo.button_name} (timed out)")
        finally:
            self.smoketest_all.setEnabled(True)

        if failures:
            summary = (
                f"Smoketest FAILED for {len(failures)}/{total} demo(s):\n"
                + "\n".join(failures)
            )
        else:
            summary = f"Smoketest OK for all {total} demos."
        self.demo_text.setPlainText(summary)

    def filter_demos(self, text: str) -> None:
        """
        Shows only the demo buttons whose name contains the search text
        (case-insensitive), hiding the rest.

        Parameters
        ----------
            text : str
                The current contents of the search box.
        """
        needle = text.lower()
        for button in self.button_group.buttons():
            button.setVisible(needle in button.objectName().lower())

    def _load_image(self, path: str) -> None:
        """
        Loads and displays the first PNG image found in the demo's directory.

        Parameters
        ----------
            path : str
                The root directory path of the demo.
        """
        image_path = Path(path)
        # Clear the label first
        self.image_label.clear()
        # Search for any .png file in the demo's root directory
        for file in image_path.glob("*.png"):
            pixmap = QPixmap(str(file))
            if not pixmap.isNull():
                # If a valid image is found, display it and stop searching
                self.image_label.setPixmap(pixmap)
                break

    def _load_readme(self, path: str) -> None:
        """
        Loads and displays the README.md file from the demo's directory.

        Parameters
        ----------
            path : str
                The root directory path of the demo.
        """
        try:
            readme_path = Path(path) / "README.md"
            # The QTextBrowser widget can render Markdown
            self.demo_text.setMarkdown(readme_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            # If no README is found, clear the text display
            self.demo_text.clear()
            print(f"README.md not found in {path}")

    def on_demo_toggled(self, button: QPushButton, checked: bool) -> None:
        """
        Slot for when a demo button is toggled.

        If a button is checked, it finds the corresponding Demo object,
        sets it as the active demo, and loads its image and README.

        Parameters
        ----------
            button : QPushButton
                The button that was toggled.
            checked : bool
                True if the button was checked, False if unchecked.
        """
        if checked:
            button_name = button.objectName()
            # Find the demo that corresponds to the clicked button
            for demo in self.executables:
                if demo.button_name == button_name:
                    self.active_demo = demo
                    # Update the UI with the new demo's content
                    self._load_image(self.active_demo.root_path)
                    self._load_readme(self.active_demo.root_path)
                    break

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Handle key press events.

        - Up/Down arrows move the demo selection.
        - Return/Enter runs the selected demo.
        - Escape closes the application.

        Parameters
        ----------
            event : QKeyEvent
                The key event.
        """
        key = event.key()
        buttons = [b for b in self.button_group.buttons() if b.isVisible()]
        if not buttons:
            return

        # Find the currently checked button
        current_index = next((i for i, b in enumerate(buttons) if b.isChecked()), 0)
        if key == Qt.Key.Key_Up:
            # Move selection up
            new_index = (current_index - 1) % len(buttons)
            buttons[new_index].setChecked(True)
            self.demo_scroll_area.ensureWidgetVisible(buttons[new_index])
        elif key == Qt.Key.Key_Down:
            # Move selection down
            new_index = (current_index + 1) % len(buttons)
            buttons[new_index].setChecked(True)
            self.demo_scroll_area.ensureWidgetVisible(buttons[new_index])
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            # Run the currently selected demo
            self.on_demo_clicked()
        elif key == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    # Standard application setup
    app = QApplication(sys.argv)
    demo_runner = DemoRunner()
    demo_runner.resize(800, 600)
    demo_runner.show()
    sys.exit(app.exec())
