#!/usr/bin/env -S uv run --script
"""Bake IBL maps from an HDRI and save them for reuse. See README.md.

Load an equirectangular ``.exr`` or ``.hdr`` panorama, preview it, bake the
split-sum image-based-lighting maps on the GPU, eyeball the results as
thumbnails, then save the whole set to a single ``.npz`` a demo can load
without baking anything itself.
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
from bake_ibl import bake_maps
from hdri_input import load_equirect_hdr
from ibl_maps import prefilter_key, save_maps
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

HERE = Path(__file__).resolve().parent
DEFAULT_HDRI = HERE / "images" / "historic_cloister_passage_1k.exr"


def _tonemap_to_qimage(rgb: np.ndarray) -> QImage:
    """Reinhard tonemap + gamma an (H,W,3+) float array to an 8-bit QImage."""
    rgb = np.asarray(rgb[..., :3], dtype=np.float32)
    mapped = rgb / (rgb + 1.0)
    srgb = np.clip(mapped, 0.0, 1.0) ** (1.0 / 2.2)
    buf = np.ascontiguousarray((srgb * 255).astype(np.uint8))
    h, w, _ = buf.shape
    return QImage(buf.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class HDRIBakerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HDRI IBL Baker")
        self.image: np.ndarray | None = None
        self.maps: dict | None = None

        toolbar = QToolBar()
        self.addToolBar(toolbar)
        self.open_btn = QPushButton("Open HDRI…")
        self.bake_btn = QPushButton("Bake")
        self.save_btn = QPushButton("Save .npz…")
        self.bake_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.on_open)
        self.bake_btn.clicked.connect(self.on_bake)
        self.save_btn.clicked.connect(self.on_save)
        for b in (self.open_btn, self.bake_btn, self.save_btn):
            toolbar.addWidget(b)

        self.preview = QLabel("Open an HDRI to begin")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 320)

        self.thumbs = [QLabel() for _ in range(3)]
        thumb_row = QHBoxLayout()
        for label, caption in zip(self.thumbs, ("irradiance", "prefilter", "brdf")):
            col = QVBoxLayout()
            label.setFixedSize(128, 128)
            label.setAlignment(Qt.AlignCenter)
            col.addWidget(label)
            cap = QLabel(caption)
            cap.setAlignment(Qt.AlignCenter)
            col.addWidget(cap)
            thumb_row.addLayout(col)

        layout = QVBoxLayout()
        layout.addWidget(self.preview, 1)
        layout.addLayout(thumb_row)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def _show_error(self, title: str, err: Exception) -> None:
        traceback.print_exc()
        QMessageBox.critical(self, title, str(err))

    def on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open HDRI", str(DEFAULT_HDRI.parent), "HDRI (*.exr *.hdr)"
        )
        if not path:
            return
        try:
            self.image = load_equirect_hdr(path)
        except Exception as err:  # noqa: BLE001 - surfaced to the user
            self._show_error("Could not load HDRI", err)
            return
        self._source = path
        pix = QPixmap.fromImage(_tonemap_to_qimage(self.image))
        self.preview.setPixmap(
            pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.bake_btn.setEnabled(True)
        self.save_btn.setEnabled(False)

    def on_bake(self) -> None:
        if self.image is None:
            return
        try:
            self.maps = bake_maps(self.image, source=Path(self._source).name)
        except Exception as err:  # noqa: BLE001
            self._show_error("Bake failed", err)
            return
        previews = (
            self.maps["irradiance"][0],
            self.maps[prefilter_key(2)][0],
            # BRDF LUT is 2-channel; pad a zero blue so it tonemaps as RGB
            np.dstack(
                [
                    self.maps["brdf_lut"],
                    np.zeros(self.maps["brdf_lut"].shape[:2], np.float16),
                ]
            ),
        )
        for label, data in zip(self.thumbs, previews):
            img = _tonemap_to_qimage(np.asarray(data, np.float32))
            label.setPixmap(QPixmap.fromImage(img).scaled(128, 128, Qt.KeepAspectRatio))
        self.save_btn.setEnabled(True)

    def on_save(self) -> None:
        if self.maps is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save maps", str(HERE / "ibl_maps.npz"), "NumPy (*.npz)"
        )
        if not path:
            return
        try:
            save_maps(self.maps, path)
        except Exception as err:  # noqa: BLE001
            self._show_error("Save failed", err)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest", nargs="?", const=200, default=None, type=int, metavar="MS"
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = HDRIBakerWindow()
    win.resize(760, 620)
    win.show()
    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
