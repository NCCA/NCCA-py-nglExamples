#!/usr/bin/env -S uv run --script
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QFile, Qt
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget


@dataclass
class Demo:
    button_name: str
    root_path: str
    app_full_path: str


class DemoRunner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyNGL Demos")
        self._find_executables()
        self.load_ui()
        # now add a new button for each executable found
        layout = self.demo_list.layout()
        for demo in self.executables:
            button = QPushButton(demo.button_name)
            button.setObjectName(demo.button_name)
            button.clicked.connect(self.on_button_clicked)
            layout.addWidget(button)

    def _find_executables(self):
        root = Path.cwd()
        exclude_dirs = {".venv", ".git", "__pycache__"}
        exclude_stems = {"__name__"}

        def walk(root: Path):
            for p in root.iterdir():
                if p.is_dir():
                    if p.name in exclude_dirs:
                        continue
                    yield from walk(p)
                elif p.suffix == ".py":
                    if (
                        p.stem in exclude_stems
                        or p.name == f"{next(iter(exclude_stems))}.py"
                    ):
                        continue
                    if os.access(p, os.X_OK):
                        yield p

        self.executables = []
        for p in walk(root):
            demo = Demo(
                button_name=p.parent.name,
                root_path=str(p.parent),
                app_full_path=str(p),
            )
            self.executables.append(demo)
            print(demo)

    def load_ui(self) -> None:
        """Load the UI from a .ui file and set up the connections."""
        loader = QUiLoader()
        ui_file = QFile("DemoUI.ui")
        ui_file.open(QFile.ReadOnly)
        # Load the UI into `self` as the parent
        loaded_ui = loader.load(ui_file, self)
        self.setCentralWidget(loaded_ui)
        # add all children with object names to `self`
        for child in loaded_ui.findChildren(QWidget):
            name = child.objectName()
            if name:
                setattr(self, name, child)
                print(name)
        ui_file.close()

    def on_button_clicked(self):
        sender = self.sender()
        if not sender:
            return
        button_name = sender.objectName()
        for demo in self.executables:
            if demo.button_name == button_name:
                print(f"Button clicked: {demo.button_name}, Path: {demo.app_full_path}")
                break

    # @Slot(bool)
    # def start_button_toggled(self, state: bool):
    #     self.run_sim ^= True
    #     if state:
    #         self.start_button.setText("Stop")
    #     else:
    #         self.start_button.setText("Start")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            ...


if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = DemoRunner()
    demo.show()
    sys.exit(app.exec())
