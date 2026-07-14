#!/usr/bin/env -S uv run --active --script
import argparse
import sys
import traceback
from typing import Set

import numpy as np
import wgpu
from Emitter import Emitter
from ncca.ngl import FirstPersonCamera, Vec3
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication
from WebGPUWidget import WebGPUWidget


class WebGPUScene(WebGPUWidget):
    """
    A concrete implementation of AbstractWebGPUWidget for a WebGPU scene.

    This class implements the abstract methods to provide functionality for initializing,
    painting, and resizing the WebGPU context.
    """

    def __init__(self, width=1024, height=720):
        super().__init__(width, height)
        self.camera = FirstPersonCamera(
            Vec3(0.0, 1.0, 5.0),
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            45.0,
        )
        self.camera.set_projection(45.0, self.width() / self.height(), 0.5, 2000.0)
        self.animate = True
        self.key_pressed: Set[int] = set()
        self.spinXFace: int = 0
        self.spinYFace: int = 0
        self.rotate: bool = False
        self.translate: bool = False
        self.origX: int = 0
        self.origY: int = 0
        self.INCREMENT: float = 0.01
        self.ZOOM: float = 0.5
        self.circle_square = 1

    def initializeWebGPU(self) -> None:
        """
        Initialize the WebGPU context.

        This method sets up the WebGPU context for the scene.
        """
        super().initializeWebGPU()
        self.emitter = Emitter(
            num_particles=50000,
            max_alive=50000,
            num_per_frame=500,
        )
        self._init_billboard()
        self._create_render_pipeline()
        self.startTimer(10)

    def _create_render_pipeline(self) -> None:
        """
        Create a render pipeline.
        """
        with open("particle_shader.wgsl", "r") as f:
            shader_code = f.read()
            shader_module = self.device.create_shader_module(code=shader_code)

        # Vertex buffer layouts
        vertex_buffers = [
            # Quad vertices layout (2D)
            {
                "array_stride": 2 * 4,  # vec2<f32>
                "step_mode": wgpu.VertexStepMode.vertex,
                "attributes": [
                    {
                        "format": wgpu.VertexFormat.float32x2,
                        "offset": 0,
                        "shader_location": 0,
                    }
                ],
            },
            # Instance attributes layout (vec3 pos + float size)
            {
                "array_stride": 4 * 8,  # vec4<f32>
                "step_mode": wgpu.VertexStepMode.instance,
                "attributes": [
                    {
                        "format": wgpu.VertexFormat.float32x4,
                        "offset": 0,
                        "shader_location": 1,
                    },
                    {
                        "format": wgpu.VertexFormat.float32x4,
                        "offset": 12,
                        "shader_location": 2,
                    },
                ],
            },
        ]

        _default_depth_stencil = {
            "format": wgpu.TextureFormat.depth24plus,
            "depth_write_enabled": True,
            "depth_compare": wgpu.CompareFunction.less,
            "view": self.depth_buffer_view,
        }

        self.pipeline = self.device.create_render_pipeline(
            label="particle_pipeline",
            layout="auto",
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": vertex_buffers,
            },
            fragment={
                "module": shader_module,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
        )
        # Create a uniform buffer
        self.uniform_data = np.zeros(
            (),
            dtype=[
                ("MVP", "float32", (4, 4)),
                ("eye", "float32", (3)),
                ("circle_square", "int32", (1)),
            ],
        )

        self.uniform_buffer = self.device.create_buffer_with_data(
            data=self.uniform_data.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="line_pipeline_uniform_buffer",
        )

        bind_group_layout = self.pipeline.get_bind_group_layout(0)
        # Create the bind group
        self.bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,  # Matches @binding(0) in the shader
                    "resource": {"buffer": self.uniform_buffer},
                }
            ],
        )

    def _init_buffers(self):
        self.vertices = self.emitter.get_numpy()
        self.vertex_buffer = self.device.create_buffer(
            size=self.vertices.nbytes,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )

        # Create a copy buffer to update the vertex buffer
        self.vertex_buffer.copy_buffer = self.device.create_buffer(
            size=self.vertices.nbytes,
            usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
        )

    def _init_billboard(self):
        # Quad vertices (for billboards)
        quad_vertices = np.array(
            [
                [-0.5, -0.5],
                [0.5, -0.5],
                [-0.5, 0.5],
                [-0.5, 0.5],
                [0.5, -0.5],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        )

        # Create vertex buffers
        self.billboard_buffer = self.device.create_buffer_with_data(
            data=quad_vertices, usage=wgpu.BufferUsage.VERTEX
        )

    def paintWebGPU(self) -> None:
        """
        Paint the WebGPU content.

        This method renders the WebGPU content for the scene.
        """
        self.render_text(10, 20, "Particle System", size=20, colour=Qt.white)
        texture = self.device.create_texture(
            size=self.texture_size,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        texture_view = texture.create_view()

        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": texture_view,
                    "resolve_target": None,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.3, 0.3, 0.3, 1.0),
                },
            ],
            depth_stencil_attachment={
                "view": self.depth_buffer_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )
        particles = self.emitter.get_numpy()

        if len(particles) == 0:
            return
        verts = self.device.create_buffer_with_data(
            data=particles.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )

        self.update_uniform_buffers()
        render_pass.set_viewport(0, 0, self.texture_size[0], self.texture_size[1], 0, 1)
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
        # render_pass.set_vertex_buffer(0, verts)
        render_pass.set_vertex_buffer(0, self.billboard_buffer)
        render_pass.set_vertex_buffer(1, verts)

        render_pass.draw(6, len(particles) // 12)
        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self.update_colour_buffer(texture)

    def resizeWebGPU(self, width: int, height: int) -> None:
        """
        Resize the WebGPU context.

        This method handles resizing of the WebGPU context for the scene.

        Args:
            width (int): The new width of the widget.
            height (int): The new height of the widget.
        """
        super().resizeWebGPU(width, height)
        self.camera.set_projection(45.0, width / height, 0.5, 2000.0)

    def timerEvent(self, event):
        if self.animate:
            self.emitter.update()

        x = 0.0
        y = 0.0
        UPDATE = 0.1
        for k in self.key_pressed:
            if k == Qt.Key.Key_Left:
                y -= UPDATE
            elif k == Qt.Key.Key_Right:
                y += UPDATE
            elif k == Qt.Key.Key_Up:
                x += UPDATE
            elif k == Qt.Key.Key_Down:
                x -= UPDATE
        self.camera.move(x, y, 0.1)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """
        Handle the mouse press event.

        Args:
            event (QMouseEvent): The mouse press event.
        """
        pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            self.origX = pos.x()
            self.origY = pos.y()
            self.rotate = True
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """
        Handle the mouse move event.

        Args:
            event (QMouseEvent): The mouse move event.
        """
        if event.buttons() == Qt.MouseButton.LeftButton:
            pos = event.position()
            diffx = int(pos.x() - self.origX)
            diffy = int(pos.y() - self.origY)
            self.spinXFace += 0.5 * diffy
            self.spinYFace += 0.5 * diffx
            self.origX = pos.x()
            self.origY = pos.y()
            self.camera.process_mouse_movement(diffx, diffy)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """
        Handle the mouse release event.

        Args:
            event (QMouseEvent): The mouse release event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self.rotate = False

        elif event.button() == Qt.MouseButton.RightButton:
            self.translate = False
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """
        Handle the mouse wheel event.

        Args:
            event (QWheelEvent): The mouse wheel event.
        """
        _ = event.pixelDelta()

        self.update()

    def keyReleaseEvent(self, event: QEvent) -> None:
        """
        Handle the key release event.

        Args:
            event (QEvent): The key release event.
        """
        key = event.key()
        self.key_pressed.remove(key)
        self.update()

    def keyPressEvent(self, event):
        key = event.key()
        self.key_pressed.add(key)

        if key == Qt.Key_Space:
            self.animate = not self.animate
        if key == Qt.Key_U:
            self.emitter.update()
            self.emitter.debug()
        if key == Qt.Key.Key_Escape:
            self.close()
        if key == Qt.Key.Key_C:
            self.circle_square ^= 1

    def update_uniform_buffers(self) -> None:
        """
        update the uniform buffers for the line pipeline.
        """
        mvp_matrix = self.camera.get_vp()
        self.uniform_data["MVP"] = mvp_matrix.to_numpy()
        self.uniform_data["eye"] = np.array(
            [self.camera.eye.x, self.camera.eye.y, self.camera.eye.z], dtype=np.float32
        )
        self.uniform_data["circle_square"] = self.circle_square
        self.device.queue.write_buffer(
            buffer=self.uniform_buffer,
            buffer_offset=0,
            data=self.uniform_data.tobytes(),
        )


class DebugApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
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

    if args.debug:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    win = WebGPUScene()
    win.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())
