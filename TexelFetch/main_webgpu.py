#!/usr/bin/env -S uv run --script
"""TexelFetch (WebGPU): the same 200x200 animated height grid as main.py,
reinterpreted for a backend with no Texture Buffer Object and no
texelFetch. The height values live in a read-only storage buffer instead
-- `var<storage, read> array<f32>` in TexelFetchShader.wgsl -- indexed by
`@builtin(vertex_index)` in the vertex shader, which is WebGPU's native
equivalent of "per-vertex data read straight out of a buffer by its
index". XZ positions still sit in an ordinary vertex buffer.

`_build_xz`/`_build_y` are duplicated verbatim from main.py rather than
imported: they're a few lines each, and this repo has no precedent for
sharing helpers that small between an OpenGL demo and its WebGPU sibling.
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from ncca.ngl import Mat4, PerspMode, Vec3, logger, look_at, perspective
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

# 200 samples per axis in [-10, 10) -- 40,000 points total, matching
# main.py and the C++ source's m_gridSize.
_GRID_DIM = 10.0
_GRID_STEP = 0.1
_GRID_N = int(round(2 * _GRID_DIM / _GRID_STEP))  # 200
_COORDS = (np.arange(_GRID_N, dtype=np.float32) * _GRID_STEP) - _GRID_DIM


def _build_xz() -> np.ndarray:
    """Build the (x, z) grid positions in the same order as the C++'s outer-z/inner-x loop.

    Returns
    -------
        np.ndarray
            (_GRID_N * _GRID_N, 2) array of (x, z) pairs, Z varying slowest.
    """
    z, x = np.meshgrid(_COORDS, _COORDS, indexing="ij")
    return np.stack([x.ravel(), z.ravel()], axis=1).astype(np.float32)


def _build_y(offset: float | None) -> np.ndarray:
    """Build the height (Y) values for the storage buffer, in the same order as `_build_xz`.

    Parameters
    ----------
        offset : float | None
            Animation offset. None reproduces the source's `buildTextureBuffer()`,
            which uses `sin(x)` alone before the first tick -- everything after
            that first frame goes through `updateTextureBuffer()`'s
            `sin(x + offset) + cos(x - offset)` instead. That asymmetry is in
            the original C++, not a bug here.

    Returns
    -------
        np.ndarray
            (_GRID_N * _GRID_N,) array of heights.
    """
    z, x = np.meshgrid(_COORDS, _COORDS, indexing="ij")
    if offset is None:
        return np.sin(x).ravel().astype(np.float32)
    return (np.sin(x + offset) + np.cos(x - offset)).ravel().astype(np.float32)


class WebGPUScene(WebGPUWidget):
    """Renders the grid as a point-list, reading height from a storage buffer per vertex."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TexelFetch (WebGPU)")
        self.msaa_sample_count = 4

        # --- camera / mouse state (hand-copied from Blending/BlendingWebGPU.py) ---
        self.model_position = Vec3()
        self.rotate = False
        self.translate = False
        self.spin_x_face = 0
        self.spin_y_face = 0
        self.original_x_rotation = 0
        self.original_y_rotation = 0
        self.original_x_pos = 0
        self.original_y_pos = 0
        self.INCREMENT = 0.01
        self.ZOOM = 0.1

        self.offset = 0.0
        self.vertex_count = _GRID_N * _GRID_N

        self.view = look_at(Vec3(0, 1, 4), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.project = perspective(
            45.0, self.width() / self.height(), 0.01, 150.0, PerspMode.WebGPU
        )

        self.device = get_default_device()
        self._create_pipeline()
        self._create_scene()
        self._create_render_buffer()

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._on_tick)
        self.animation_timer.start(20)

        self.update()

    # ------------------------------------------------------------------
    # pipeline
    # ------------------------------------------------------------------
    def _create_pipeline(self) -> None:
        shader_src = (Path(__file__).parent / "TexelFetchShader.wgsl").read_text()
        shader_module = self.device.create_shader_module(code=shader_src)

        # Binding 0 is the MVP uniform, binding 1 is the height storage
        # buffer -- both read only in the vertex stage, so that's the only
        # visibility either needs.
        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
            ]
        )
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout]
        )

        self.pipeline = self.device.create_render_pipeline(
            label="texel_fetch_pipeline",
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [
                    {
                        "array_stride": 2 * 4,
                        "attributes": [
                            {
                                "format": wgpu.VertexFormat.float32x2,
                                "offset": 0,
                                "shader_location": 0,
                            }
                        ],
                    }
                ],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.point_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": self.msaa_sample_count},
        )

    # ------------------------------------------------------------------
    # scene
    # ------------------------------------------------------------------
    def _create_scene(self) -> None:
        xz = _build_xz()
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=xz.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )

        # Height data lives in a storage buffer rather than a second vertex
        # attribute, so the vertex shader can index it by vertex_index --
        # the WebGPU equivalent of main.py's TBO. Created once here, then
        # re-uploaded in place every tick (see _on_tick) rather than
        # recreated, matching ShadedGrid/main_webgpu.py's per-frame-update
        # convention.
        y = _build_y(None)
        self.y_buffer = self.device.create_buffer_with_data(
            data=y.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        # A single uniform buffer for MVP -- one draw call per frame, so
        # the per-draw uniform buffer pool other demos use for aliasing
        # safety doesn't apply here.
        self.uniform_buffer = self.device.create_buffer(
            size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

        self.bind_group = self.device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.uniform_buffer,
                        "offset": 0,
                        "size": self.uniform_buffer.size,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self.y_buffer,
                        "offset": 0,
                        "size": self.y_buffer.size,
                    },
                },
            ],
        )

    def _on_tick(self) -> None:
        """Recompute the height data and re-upload it to the storage buffer."""
        y = _build_y(self.offset)
        self.device.queue.write_buffer(self.y_buffer, 0, y.tobytes())
        self.offset += 0.01
        self.update()

    def scene_global_tx(self) -> Mat4:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        tx = rot_y @ rot_x
        tx[3, 0] = self.model_position.x
        tx[3, 1] = self.model_position.y
        tx[3, 2] = self.model_position.z
        return tx

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def paintWebGPU(self) -> None:
        mvp = self.project @ self.view @ self.scene_global_tx()
        self.device.queue.write_buffer(
            self.uniform_buffer, 0, mvp.to_numpy().astype(np.float32).tobytes()
        )

        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.multisample_texture_view,
                    "resolve_target": self.colour_buffer_texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.4, 0.4, 0.4, 1.0),
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_buffer_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(self.vertex_count)
        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.project = perspective(
            45.0, width / max(height, 1), 0.05, 150.0, PerspMode.WebGPU
        )
        self.update()

    # ------------------------------------------------------------------
    # input (hand-copied from Blending/BlendingWebGPU.py -- no shared
    # mixin exists for QWidget-based WebGPU demos)
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)
        self.update()
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        position = event.position()
        if event.button() == Qt.LeftButton:
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.rotate = True
        elif event.button() == Qt.RightButton:
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.translate = True

    def mouseMoveEvent(self, event) -> None:
        if self.rotate and event.buttons() == Qt.LeftButton:
            position = event.position()
            diff_x = position.x() - self.original_x_rotation
            diff_y = position.y() - self.original_y_rotation
            self.spin_x_face += int(0.5 * diff_y)
            self.spin_y_face += int(0.5 * diff_x)
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.update()
        elif self.translate and event.buttons() == Qt.RightButton:
            position = event.position()
            diff_x = int(position.x() - self.original_x_pos)
            diff_y = int(position.y() - self.original_y_pos)
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.model_position.x += self.INCREMENT * diff_x
            self.model_position.y -= self.INCREMENT * diff_y
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.rotate = False
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        self.model_position.z += self.ZOOM * (delta / 120.0)
        self.update()

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        # Stop the animation timer before the base class tears down the
        # wgpu surface/device -- otherwise a queued timer tick can fire a
        # GPU call (queue.write_buffer / paintWebGPU) after teardown and
        # crash. Same fix as main.py's closeEvent, ported here from the
        # start rather than waiting for it to bite (see
        # Collisions/SphereSphere/main_webgpu.py for the same pattern).
        self.animation_timer.stop()
        super().closeEvent(event)


class DebugApplication(QApplication):
    """QApplication that re-raises exceptions from Qt event handlers instead of swallowing them."""

    def __init__(self, argv):
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoketest",
        nargs="?",
        const=200,
        default=None,
        type=int,
        metavar="MS",
        help="run for MS milliseconds (default 200), print SMOKETEST OK and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run with DebugApplication (tracebacks from Qt event handlers)",
    )
    args = parser.parse_args()

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    window = WebGPUScene()
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
