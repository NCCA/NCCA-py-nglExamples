#!/usr/bin/env -S uv run --active --script
import argparse
import sys
import traceback

import numpy as np
import wgpu
import wgpu.utils
from ncca.ngl import PerspMode, ortho
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import QElapsedTimer, Qt, QTimer, QTimerEvent
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

SIM_WIDTH = 200
SIM_HEIGHT = 200


class WebGPUScene(WebGPUWidget):
    """
    A concrete implementation of WebGPUWidget for a WebGPU scene.

    This class implements the abstract methods to provide functionality for initializing,
    painting, and resizing the WebGPU context.
    """

    def __init__(self, num_points=10000):
        super().__init__()
        self.window_width: int = 1024  # Window width
        self.window_height: int = 720  # Window height
        self.setWindowTitle("WebGPU 2D Pan and Zoom")
        self.device = None
        self.pipeline = None
        self.position_buffer = None
        self.colour_buffer = None
        self.num_points = num_points
        self.msaa_sample_count = 4
        self.ratio = self.devicePixelRatio()
        self.animate = True
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = 1.0
        self.is_panning = False
        self.last_mouse_pos = None
        self.point_size = 0.6
        self.wind = np.array([0.0, 0.0], dtype=np.float32)
        self.timer = QElapsedTimer()
        self.dt = 0.0
        #  pan (world-space center) so we can zoom around mouse and pan the view.
        self.pan = np.array([0.0, 0.0], dtype=np.float32)
        # Track last mouse position (QPointF) for right-drag panning.
        self._last_mouse_pos = None

        self.gen_points(self.num_points)
        self._initialize_web_gpu()
        self.update()

    def gen_points(self, num_points: int) -> None:
        """
        Generates random 2D points with associated positions, directions, and colours.

        This function initializes three numpy arrays:
        - self.positions: A 2D array storing the (x, y) coordinates of each point.
        - self.directions: A 2D array storing the (dx, dy) velocity vector for each point.
        - self.colours: A 2D array storing the (r, g, b) colour of each point.

        Args:
            num_points: The number of points to generate.

        """
        # generate positions in 2D space the size of the simulation with 0,0 the center
        self.positions = np.zeros((num_points, 2), dtype=np.float32)
        self.positions[:, 0] = np.random.uniform(
            -SIM_WIDTH / 2, SIM_WIDTH / 2, num_points
        )
        self.positions[:, 1] = np.random.uniform(
            -SIM_HEIGHT / 2, SIM_HEIGHT / 2, num_points
        )

        # generate directions in 2D space with random velocities
        self.directions = np.zeros((num_points, 2), dtype=np.float32)
        self.directions[:, 0] = np.random.uniform(-1, 1, num_points)
        self.directions[:, 1] = np.random.uniform(-1, 1, num_points)

        # Normalize directions to unit length, so they only represent direction
        self.directions /= np.linalg.norm(self.directions, axis=1, keepdims=True)

        min_speed = 0.5
        max_speed = 2.0
        # Create a (num_points, 1) array of random speeds
        speeds = np.random.uniform(min_speed, max_speed, (num_points, 1))
        # Multiply the unit direction vectors by the speeds to get final velocity vectors
        self.directions *= speeds

        self.colours = np.random.random((num_points, 3)).astype(np.float32)

    def _initialize_web_gpu(self) -> None:
        """
        Initialize the WebGPU context.

        This method sets up the WebGPU context for the scene.
        """
        print("initializeWebGPU")
        try:
            self.device = get_default_device()
            self._init_buffers()
            self._create_render_buffer()
            self._create_render_pipeline()
            self.startTimer(16)
            self.timer.start()
        except Exception as e:
            print(f"Failed to initialize WebGPU: {e}")

    def _init_buffers(self):
        # Create a buffer for the positions.
        self.position_buffer = self.device.create_buffer_with_data(
            data=self.positions.tobytes(),
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )
        # # Create a copy buffer to update the vertex buffer
        self.position_buffer.copy_buffer = self.device.create_buffer(
            size=self.positions.nbytes,
            usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
        )
        # Create a buffer for the colours.
        self.colour_buffer = self.device.create_buffer_with_data(
            data=self.colours.tobytes(),
            usage=wgpu.BufferUsage.VERTEX,
        )

    def _create_render_buffer(self):
        # This is the texture that the multisampled texture will be resolved to
        colour_buffer_texture = self.device.create_texture(
            size=self.texture_size,
            sample_count=1,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        self.colour_buffer_texture = colour_buffer_texture
        self.colour_buffer_texture_view = self.colour_buffer_texture.create_view()

        # This is the multisampled texture that will be rendered to
        self.multisample_texture = self.device.create_texture(
            size=self.texture_size,
            sample_count=self.msaa_sample_count,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        self.multisample_texture_view = self.multisample_texture.create_view()

        # Now create a depth buffer
        depth_texture = self.device.create_texture(
            size=self.texture_size,
            format=wgpu.TextureFormat.depth24plus,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            sample_count=self.msaa_sample_count,
        )
        self.depth_buffer_view = depth_texture.create_view()

        # Calculate aligned buffer size for texture copy
        buffer_size = self._calculate_aligned_buffer_size()
        self.readback_buffer = self.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

    def _create_render_pipeline(self) -> None:
        """
        Create a render pipeline.
        """
        with open("PointShader.wgsl", "r") as f:
            shader_code = f.read()
            shader_module = self.device.create_shader_module(code=shader_code)

        self.pipeline = self.device.create_render_pipeline(
            label="particle_pipeline",
            layout="auto",
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 2 * 4,  # vec3 position
                        "step_mode": "instance",
                        "attributes": [
                            {"format": "float32x2", "offset": 0, "shader_location": 0},
                        ],
                    },
                    {
                        "array_stride": 3 * 4,  # vec3 colour
                        "step_mode": "instance",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 1},
                        ],
                    },
                ],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_strip},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={
                "count": self.msaa_sample_count,
            },
        )

        # Create a uniform buffer
        self.uniform_data = np.zeros(
            (),
            dtype=[
                ("projection_matrix", "float32", (4, 4)),
                ("size", "float32"),
                ("padding", np.uint32, 3),  # 3 * 4 = 12 bytes padding
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

    def resizeWebGPU(self, width, height) -> None:
        """
        Called whenever the window is resized.
        It's crucial to update the viewport and projection matrix here.

        Args:
            width: The new width of the window.
            height: The new height of the window.
        """
        self.window_width = int(width * self.ratio)
        self.window_height = int(height * self.ratio)

        self.update()

    def paintWebGPU(self) -> None:
        """
        Paint the WebGPU content.

        This method renders the WebGPU content for the scene.
        """
        self.render_text(
            10,
            25,
            f"Wind Value use Arrow Keys to Change, Space Reset [{self.wind[0]:.02f}, {self.wind[1]:.02f}]",
            size=20,
            colour=Qt.yellow,
        )
        try:
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
            self.update_uniform_buffers()
            render_pass.set_viewport(
                0, 0, self.texture_size[0], self.texture_size[1], 0, 1
            )
            render_pass.set_pipeline(self.pipeline)
            render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
            render_pass.set_vertex_buffer(0, self.position_buffer)
            render_pass.set_vertex_buffer(1, self.colour_buffer)
            render_pass.draw(4, self.num_points)
            render_pass.end()
            self.device.queue.submit([command_encoder.finish()])
            self._update_colour_buffer()
        except Exception as e:
            print(f"Failed to paint WebGPU content: {e}")

    def update_uniform_buffers(self) -> None:
        """
        update the uniform buffers for the line pipeline.
        """
        # Use pan when creating the orthographic projection so we can translate the view.
        half_w = SIM_WIDTH / 2 * self.zoom
        half_h = SIM_HEIGHT / 2 * self.zoom
        proj = ortho(
            self.pan[0] - half_w,
            self.pan[0] + half_w,
            self.pan[1] - half_h,
            self.pan[1] + half_h,
            0,
            1,
            PerspMode.WebGPU,
        )

        self.uniform_data["projection_matrix"] = proj.to_numpy()
        self.uniform_data["size"] = self.point_size
        self.device.queue.write_buffer(
            buffer=self.uniform_buffer,
            buffer_offset=0,
            data=self.uniform_data.tobytes(),
        )
        self.device.queue.write_buffer(
            buffer=self.position_buffer,
            buffer_offset=0,
            data=self.positions.tobytes(),
        )

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Handles keyboard press events.

        Args:
            event: The QKeyEvent object containing information about the key press.
        """
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()  # Exit the application
        elif key == Qt.Key_A:
            self.animate = not self.animate
        elif key == Qt.Key_Space:
            self.wind[0] = 0
            self.wind[1] = 0
            self.zoom = 1.0
            # Reset pan as well when space is pressed
            self.pan[:] = 0.0
        elif key == Qt.Key_Up:
            self.wind[1] += 0.1
        elif key == Qt.Key_Down:
            self.wind[1] -= 0.1
        elif key == Qt.Key_Left:
            self.wind[0] -= 0.1
        elif key == Qt.Key_Right:
            self.wind[0] += 0.1
        # Trigger a redraw to apply changes
        self.update()
        # Call the base class implementation for any unhandled events
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """
        Handles mouse movement events for camera control.

        Args:
            event: The QMouseEvent object containing the new mouse position.
        """
        # Rotate the scene if the left mouse button is pressed
        if event.buttons() == Qt.LeftButton:
            self.update()
        # Translate (pan) the scene if the right mouse button is pressed
        elif event.buttons() == Qt.RightButton:
            # perform panning: compute pixel delta and convert to world delta
            if self._last_mouse_pos is not None:
                cur = event.position()
                # pixel delta (consider device pixel ratio)
                dx = (cur.x() - self._last_mouse_pos.x()) * self.ratio
                dy = (cur.y() - self._last_mouse_pos.y()) * self.ratio

                view_w = SIM_WIDTH * self.zoom
                view_h = SIM_HEIGHT * self.zoom

                # convert pixel delta to world units: moving mouse right should pan view left (so subtract)
                self.pan[0] -= dx / max(1, self.window_width) * view_w
                # for y: pixel y increases downwards; moving mouse down should pan view up, so add
                self.pan[1] += dy / max(1, self.window_height) * view_h

                self._last_mouse_pos = cur
                self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """
        Handles mouse button press events to initiate rotation or translation.

        Args:
            event: The QMouseEvent object.
        """
        # store the mouse position for drag operations
        try:
            self._last_mouse_pos = event.position()
        except Exception:
            # fallback in case old PySide6 returns different type
            self._last_mouse_pos = event.pos()
        # Left button initiates rotation

    def wheelEvent(self, event: QWheelEvent) -> None:
        """
        Handles mouse wheel events for zooming.

        Zoom is performed around the mouse cursor position. The algorithm:
        1. Convert the mouse pixel position to world coordinates with the current zoom/pan.
        2. Update the zoom factor.
        3. Convert the same pixel position to world coordinates with the new zoom.
        4. Adjust pan by the difference so the same world point remains under the cursor.
        """
        # angleDelta().y() is the vertical wheel movement (positive means up / zoom in).
        delta = event.angleDelta().y()

        # read mouse position in widget coordinates
        try:
            pos = event.position()
            mouse_x = pos.x()
            mouse_y = pos.y()
        except Exception:
            p = event.pos()
            mouse_x = p.x()
            mouse_y = p.y()

        # convert to framebuffer pixels using device pixel ratio
        pixel_x = mouse_x * self.ratio
        pixel_y = mouse_y * self.ratio

        # current view extents
        half_w = SIM_WIDTH / 2 * self.zoom
        half_h = SIM_HEIGHT / 2 * self.zoom
        left = self.pan[0] - half_w
        right = self.pan[0] + half_w
        bottom = self.pan[1] - half_h
        top = self.pan[1] + half_h

        # map pixel to world (pixel origin is top-left)
        world_x_before = left + (pixel_x / max(1, self.window_width)) * (right - left)
        # pixel y=0 -> world = top, pixel y increases downward, so subtract fraction from top
        world_y_before = top - (pixel_y / max(1, self.window_height)) * (top - bottom)

        # update zoom factor: use a sensible scaling from wheel delta
        # Typical angleDelta is 120 per notch; scale such that small increments are smooth
        scale_factor = 1.0 + (delta / 1200.0)  # tweakable
        new_zoom = (
            self.zoom / scale_factor
        )  # dividing so wheel up (positive delta) zooms in

        # clamp
        new_zoom = max(0.05, min(10.0, new_zoom))

        # compute new view extents with new zoom
        new_half_w = SIM_WIDTH / 2 * new_zoom
        new_half_h = SIM_HEIGHT / 2 * new_zoom
        new_left = self.pan[0] - new_half_w
        new_right = self.pan[0] + new_half_w
        new_bottom = self.pan[1] - new_half_h
        new_top = self.pan[1] + new_half_h

        # compute world coordinate at the same pixel after zoom (but before adjusting pan)
        world_x_after = new_left + (pixel_x / max(1, self.window_width)) * (
            new_right - new_left
        )
        world_y_after = new_top - (pixel_y / max(1, self.window_height)) * (
            new_top - new_bottom
        )

        # adjust pan so that the world point under the cursor remains the same
        # new_pan = pan + (world_before - world_after)
        shift = np.array(
            [world_x_before - world_x_after, world_y_before - world_y_after],
            dtype=np.float32,
        )
        self.pan += shift

        # finally set the zoom
        self.zoom = new_zoom

        self.update()

    def initialize_buffer(self) -> None:
        """
        Initialize the numpy buffer for rendering .

        """
        print("initialize numpy buffer")
        width = int(self.width() * self.ratio)
        height = int(self.height() * self.ratio)
        self.frame_buffer = np.zeros([height, width, 4], dtype=np.uint8)

    def timerEvent(self, event: QTimerEvent) -> None:
        """
        This event is called at a regular interval (set by startTimer).
        It's used to update the animation of the scene.

        Here, it updates the positions of the points and makes them bounce
        off the edges of the simulation area.

        Args:
            event: The QTimerEvent object, not used in this method but required by the API.


        """
        if not self.animate:
            return
        # Add the wind factor to the particle's own direction to get the final velocity
        velocities = self.directions + self.wind
        # Update positions using the final velocity
        self.positions += velocities

        # Define the boundaries of the simulation area
        x_min = -SIM_WIDTH / 2
        x_max = SIM_WIDTH / 2
        y_min = -SIM_HEIGHT / 2
        y_max = SIM_HEIGHT / 2

        # Create boolean masks to find points that are out of bounds
        hit_left = self.positions[:, 0] < x_min
        hit_right = self.positions[:, 0] > x_max
        hit_bottom = self.positions[:, 1] < y_min
        hit_top = self.positions[:, 1] > y_max

        # Reflect the particle's intrinsic direction for points that have hit a wall.
        self.directions[hit_left | hit_right, 0] *= -1
        self.directions[hit_bottom | hit_top, 1] *= -1

        # Clamp positions to the boundaries to prevent particles from getting stuck.
        # If a particle hit the left wall, its x position is set to the left boundary.
        self.positions[hit_left, 0] = x_min
        # If a particle hit the right wall, its x position is set to the right boundary.
        self.positions[hit_right, 0] = x_max
        # If a particle hit the bottom wall, its y position is set to the bottom boundary.
        self.positions[hit_bottom, 1] = y_min
        # If a particle hit the top wall, its y position is set to the top boundary.
        self.positions[hit_top, 1] = y_max
        self.update()


class DebugApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


def main():
    """
    Main function to run the application.
    Parses command line arguments and initializes the WebGPUScene.
    """
    parser = argparse.ArgumentParser(description="A WebGPU points demo")
    parser.add_argument(
        "-p",
        "--points",
        type=int,
        default=1000,
        help="The number of points to generate.",
    )
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

    win = WebGPUScene(num_points=args.points)
    win.resize(1024, 720)
    win.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
