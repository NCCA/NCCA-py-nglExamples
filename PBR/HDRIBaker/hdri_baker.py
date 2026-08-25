#!/usr/bin/env -S uv run --script
"""Bake IBL maps from an HDRI and save them for reuse. See README.md.

Load an equirectangular ``.exr`` or ``.hdr`` panorama, preview it, bake the
split-sum image-based-lighting maps on the GPU, eyeball the results as
thumbnails, then save the whole set to a single ``.npz`` a demo can load
without baking anything itself.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from bake_ibl import bake_maps
from bake_settings import BakeSettings, prefilter_key
from hdri_input import load_equirect_hdr
from ibl_maps import save_maps
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
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


def _power_of_two_combo(choices: list[int], current: int) -> QComboBox:
    """A combo of power-of-two sizes -- a free-text spin box would only invite
    values the bake has to reject."""
    box = QComboBox()
    for value in choices:
        box.addItem(str(value), value)
    box.setCurrentIndex(choices.index(current))
    return box


class SettingsPanel(QGroupBox):
    """The bake's knobs. Sizes trade file size and detail; sample counts trade
    bake time against noise."""

    def __init__(self) -> None:
        super().__init__("Bake settings")
        d = BakeSettings()
        self.env = _power_of_two_combo([128, 256, 512, 1024, 2048], d.env_size)
        self.irradiance = _power_of_two_combo([8, 16, 32, 64], d.irradiance_size)
        self.prefilter = _power_of_two_combo([32, 64, 128, 256], d.prefilter_size)
        self.lut = _power_of_two_combo([64, 128, 256, 512], d.lut_size)

        self.mips = QSpinBox()
        self.mips.setRange(2, 8)  # 1 mip cannot span a roughness range
        self.mips.setValue(d.prefilter_mips)

        self.prefilter_samples = QSpinBox()
        self.prefilter_samples.setRange(1, 8192)
        self.prefilter_samples.setValue(d.prefilter_samples)

        self.brdf_samples = QSpinBox()
        self.brdf_samples.setRange(1, 8192)
        self.brdf_samples.setValue(d.brdf_samples)

        self.sample_delta = QDoubleSpinBox()
        self.sample_delta.setRange(0.005, 1.0)
        self.sample_delta.setSingleStep(0.005)
        self.sample_delta.setDecimals(3)
        self.sample_delta.setValue(d.irradiance_sample_delta)

        form = QFormLayout()
        form.addRow("Environment cube", self.env)
        form.addRow("Irradiance cube", self.irradiance)
        form.addRow("Prefilter cube", self.prefilter)
        form.addRow("Prefilter mips", self.mips)
        form.addRow("BRDF LUT", self.lut)
        form.addRow("Prefilter samples", self.prefilter_samples)
        form.addRow("BRDF samples", self.brdf_samples)
        form.addRow("Irradiance sample delta", self.sample_delta)
        self.setLayout(form)

    def settings(self) -> BakeSettings:
        return BakeSettings(
            env_size=self.env.currentData(),
            irradiance_size=self.irradiance.currentData(),
            prefilter_size=self.prefilter.currentData(),
            prefilter_mips=self.mips.value(),
            lut_size=self.lut.currentData(),
            prefilter_samples=self.prefilter_samples.value(),
            brdf_samples=self.brdf_samples.value(),
            irradiance_sample_delta=self.sample_delta.value(),
        )


class HDRIBakerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HDRI IBL Baker")
        self.image: np.ndarray | None = None
        self.maps: dict | None = None
        self._source: str | None = None

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

        self.settings_panel = SettingsPanel()
        self.timing = QLabel("")
        self.timing.setAlignment(Qt.AlignCenter)

        side = QVBoxLayout()
        side.addWidget(self.settings_panel)
        side.addWidget(self.timing)
        side.addStretch(1)

        top = QHBoxLayout()
        top.addWidget(self.preview, 1)
        top.addLayout(side)

        layout = QVBoxLayout()
        layout.addLayout(top, 1)
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
        settings = self.settings_panel.settings()
        try:
            settings.validate()
        except ValueError as err:
            self._show_error("Invalid bake settings", err)
            return

        # A big bake blocks the event loop for a while; at least let the
        # button look pressed and the cursor say so.
        self.bake_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        start = time.perf_counter()
        try:
            self.maps = bake_maps(self.image, settings, source=Path(self._source).name)
        except Exception as err:  # noqa: BLE001
            self._show_error("Bake failed", err)
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.bake_btn.setEnabled(True)
        self.timing.setText(f"baked in {time.perf_counter() - start:.2f}s")
        thumb_mip = min(2, settings.prefilter_mips - 1)
        previews = (
            self.maps["irradiance"][0],
            self.maps[prefilter_key(thumb_mip)][0],
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
            label.setPixmap(
                QPixmap.fromImage(img).scaled(
                    128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
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
    win.resize(1040, 640)
    win.show()
    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
