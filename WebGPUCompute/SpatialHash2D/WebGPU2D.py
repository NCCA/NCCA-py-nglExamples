#!/usr/bin/env -S uv run --active --script
import argparse
import sys

import numpy as np
import wgpu
from ncca.ngl import PerspMode, ortho
from PySide6.QtCore import QElapsedTimer, Qt, QTimerEvent
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication
from WebGPUWidget import WebGPUWidget
from wgpu.utils import get_default_device

SIM_WIDTH = 500
SIM_HEIGHT = 500
GRID_CELL_SIZE = 50.0  # Size of each grid cell
PARTICLE_RADIUS = 0.6  # Collision radius for particles


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
        self.setWindowTitle("WebGPU 2D Pan and Zoom with Collisions")
        self.device = None
        self.pipeline = None
        self.compute_pipeline = None
        self.line_pipeline = None
        self.grid_buffer = None
        self.line_bind_group = None
        self.particle_buffer = None
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
        self.show_grid = True
        self.point_size = 0.6
        self.wind = np.array([0.0, 0.0], dtype=np.float32)
        self.timer = QElapsedTimer()
        self.dt = 0.0
        self.last_time = 0.0
        #  pan (world-space center) so we can zoom around mouse and pan the view.
        self.pan = np.array([0.0, 0.0], dtype=np.float32)
        # Track last mouse position (QPointF) for right-drag panning.
        self._last_mouse_pos = None

        self.gen_points(self.num_points)
        self._initialize_web_gpu()
        self.update()

    def _gen_grid_lines(self) -> None:
        """
        Generates vertex data for drawing the grid lines.
        """
        grid_lines = []
        # Vertical lines
        for i in range(self.grid_width + 1):
            x = -SIM_WIDTH / 2 + i * GRID_CELL_SIZE
            grid_lines.append([x, -SIM_HEIGHT / 2])
            grid_lines.append([x, SIM_HEIGHT / 2])
        # Horizontal lines
        for i in range(self.grid_height + 1):
            y = -SIM_HEIGHT / 2 + i * GRID_CELL_SIZE
            grid_lines.append([-SIM_WIDTH / 2, y])
            grid_lines.append([SIM_WIDTH / 2, y])
        self.grid_lines = np.array(grid_lines, dtype=np.float32)

    def gen_points(self, num_points: int) -> None:
        """
        Generates random 2D points with associated positions, directions, and colours.

        This function initializes particle data in a structured format suitable for compute shaders.

        Args:
            num_points: The number of points to generate.
        """
        # Create structured array matching the Particle struct in the compute shader
        self.particle_data = np.zeros(
            num_points,
            dtype=[
                ("pos", "float32", 2),
                ("vel", "float32", 2),
            ],
        )

        # Generate positions in 2D space the size of the simulation with 0,0 the center
        self.particle_data["pos"][:, 0] = np.random.uniform(
            -SIM_WIDTH / 2, SIM_WIDTH / 2, num_points
        )
        self.particle_data["pos"][:, 1] = np.random.uniform(
            -SIM_HEIGHT / 2, SIM_HEIGHT / 2, num_points
        )

        # Generate directions in 2D space with random velocities
        # Use angles to ensure uniform distribution and avoid zero-length vectors
        angles = np.random.uniform(0, 2 * np.pi, num_points)
        directions = np.zeros((num_points, 2), dtype=np.float32)
        directions[:, 0] = np.cos(angles)
        directions[:, 1] = np.sin(angles)

        min_speed = 5.0
        max_speed = 15.0
        speeds = np.random.uniform(min_speed, max_speed, (num_points, 1))
        directions *= speeds

        self.particle_data["vel"] = directions

        # Generate colors separately
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
            self._create_compute_pipeline()
            self._create_render_buffer()
            self._create_render_pipeline()
            self._create_line_render_pipeline()
            self.startTimer(16)
            self.timer.start()
            self.last_time = self.timer.elapsed() / 1000.0
        except Exception as e:
            print(f"Failed to initialize WebGPU: {e}")

    def _create_line_render_pipeline(self) -> None:
        """
        Create a render pipeline for drawing lines.
        """
        with open("LineShader.wgsl", "r") as f:
            shader_code = f.read()
            shader_module = self.device.create_shader_module(code=shader_code)

        self.line_pipeline = self.device.create_render_pipeline(
            label="line_pipeline",
            layout="auto",
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 2 * 4,  # vec2 pos
                        "step_mode": "vertex",
                        "attributes": [
                            {
                                "format": "float32x2",
                                "offset": 0,
                                "shader_location": 0,
                            },
                        ],
                    },
                ],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.line_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={
                "count": self.msaa_sample_count,
            },
        )
        bind_group_layout = self.line_pipeline.get_bind_group_layout(0)
        # Create the bind group
        self.line_bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,  # Matches @binding(0) in the shader
                    "resource": {"buffer": self.uniform_buffer},
                }
            ],
        )

    def _init_buffers(self):
        # Create a storage buffer for particles (used by compute shader and rendering)
        self.particle_buffer = self.device.create_buffer_with_data(
            data=self.particle_data.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.VERTEX,
        )

        # Create a buffer for the colours
        self.colour_buffer = self.device.create_buffer_with_data(
            data=self.colours.tobytes(),
            usage=wgpu.BufferUsage.VERTEX,
        )

        # Calculate grid dimensions
        self.grid_width = int(np.ceil(SIM_WIDTH / GRID_CELL_SIZE))
        self.grid_height = int(np.ceil(SIM_HEIGHT / GRID_CELL_SIZE))
        self.total_cells = self.grid_width * self.grid_height
        self._gen_grid_lines()
        # Create a buffer for the grid lines
        self.grid_buffer = self.device.create_buffer_with_data(
            data=self.grid_lines.tobytes(),
            usage=wgpu.BufferUsage.VERTEX,
        )

        print(f"Grid: {self.grid_width}x{self.grid_height} = {self.total_cells} cells")

        # Create uniform buffer for simulation parameters
        self.sim_params = np.zeros(
            (),
            dtype=[
                ("dt", "float32"),
                ("width", "float32"),
                ("height", "float32"),
                ("wind_x", "float32"),
                ("wind_y", "float32"),
                ("grid_width", "uint32"),
                ("grid_height", "uint32"),
                ("cell_size", "float32"),
                ("particle_radius", "float32"),
                ("padding", "uint32", 3),
            ],
        )
        self.sim_params["width"] = SIM_WIDTH
        self.sim_params["height"] = SIM_HEIGHT
        self.sim_params["wind_x"] = 0.0
        self.sim_params["wind_y"] = 0.0
        self.sim_params["grid_width"] = self.grid_width
        self.sim_params["grid_height"] = self.grid_height
        self.sim_params["cell_size"] = GRID_CELL_SIZE
        self.sim_params["particle_radius"] = PARTICLE_RADIUS

        self.sim_params_buffer = self.device.create_buffer_with_data(
            data=self.sim_params.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # Create grid buffers for spatial hashing
        self.grid_indices_buffer = self.device.create_buffer(
            size=self.num_points * 4,  # u32 per particle
            usage=wgpu.BufferUsage.STORAGE,
        )

        self.grid_offsets_buffer = self.device.create_buffer(
            size=self.total_cells * 4,  # u32 per cell
            usage=wgpu.BufferUsage.STORAGE,
        )

        self.cell_particle_count_buffer = self.device.create_buffer(
            size=self.total_cells * 4,  # u32 per cell
            usage=wgpu.BufferUsage.STORAGE,
        )

    def _create_compute_pipeline(self) -> None:
        """
        Create the compute pipeline for particle simulation with collisions.
        """
        with open("CollisionCompute.wgsl", "r") as f:
            compute_shader_code = f.read()
            compute_shader_module = self.device.create_shader_module(
                code=compute_shader_code
            )

        # Create a shared bind group layout that all pipelines will use
        bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.storage},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.storage},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.storage},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.storage},
                },
            ]
        )

        # Create pipeline layout
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )

        # Create multiple compute pipelines for different phases using the shared layout
        self.clear_grid_pipeline = self.device.create_compute_pipeline(
            label="clear_grid",
            layout=pipeline_layout,
            compute={
                "module": compute_shader_module,
                "entry_point": "clear_grid",
            },
        )

        self.count_particles_pipeline = self.device.create_compute_pipeline(
            label="count_particles",
            layout=pipeline_layout,
            compute={
                "module": compute_shader_module,
                "entry_point": "count_particles",
            },
        )

        self.build_offsets_pipeline = self.device.create_compute_pipeline(
            label="build_offsets",
            layout=pipeline_layout,
            compute={
                "module": compute_shader_module,
                "entry_point": "build_offsets",
            },
        )

        self.fill_grid_pipeline = self.device.create_compute_pipeline(
            label="fill_grid",
            layout=pipeline_layout,
            compute={
                "module": compute_shader_module,
                "entry_point": "fill_grid",
            },
        )

        self.detect_collisions_pipeline = self.device.create_compute_pipeline(
            label="detect_collisions",
            layout=pipeline_layout,
            compute={
                "module": compute_shader_module,
                "entry_point": "detect_collisions",
            },
        )

        self.update_physics_pipeline = self.device.create_compute_pipeline(
            label="update_physics",
            layout=pipeline_layout,
            compute={
                "module": compute_shader_module,
                "entry_point": "update_physics",
            },
        )

        # Create a shared bind group with all resources
        self.compute_bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.particle_buffer,
                        "offset": 0,
                        "size": self.particle_data.nbytes,
                    },
                },
                {
                    "binding": 1,
                    "resource": {"buffer": self.sim_params_buffer},
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": self.grid_indices_buffer,
                        "offset": 0,
                        "size": self.num_points * 4,
                    },
                },
                {
                    "binding": 3,
                    "resource": {
                        "buffer": self.grid_offsets_buffer,
                        "offset": 0,
                        "size": self.total_cells * 4,
                    },
                },
                {
                    "binding": 4,
                    "resource": {
                        "buffer": self.cell_particle_count_buffer,
                        "offset": 0,
                        "size": self.total_cells * 4,
                    },
                },
            ],
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
                        "array_stride": 4
                        * 4,  # Full particle: vec2 pos + vec2 vel = 16 bytes
                        "step_mode": "instance",
                        "attributes": [
                            {
                                "format": "float32x2",
                                "offset": 0,
                                "shader_location": 0,
                            },  # Only read pos
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
        current_time = self.timer.elapsed() / 1000.0
        self.dt = current_time - self.last_time
        self.last_time = current_time
        self.render_text(
            10,
            25,
            f" Points {self.num_points}  Wind [{self.wind[0]:.02f}, {self.wind[1]:.02f}] dt: {self.dt:.4f} FPS: {1.0 / self.dt if self.dt > 0 else 0:.2f}",
            size=20,
            colour=Qt.yellow,
        )
        try:
            command_encoder = self.device.create_command_encoder()

            # Run compute shader passes if animating
            if self.animate:
                self.update_simulation_params()

                # Calculate workgroup counts
                particle_workgroups = (self.num_points + 63) // 64
                cell_workgroups = (self.total_cells + 63) // 64

                # Phase 1: Clear grid
                compute_pass = command_encoder.begin_compute_pass()
                compute_pass.set_pipeline(self.clear_grid_pipeline)
                compute_pass.set_bind_group(0, self.compute_bind_group, [], 0, 999999)
                compute_pass.dispatch_workgroups(cell_workgroups, 1, 1)
                compute_pass.end()

                # Phase 2: Count particles per cell
                compute_pass = command_encoder.begin_compute_pass()
                compute_pass.set_pipeline(self.count_particles_pipeline)
                compute_pass.set_bind_group(0, self.compute_bind_group, [], 0, 999999)
                compute_pass.dispatch_workgroups(particle_workgroups, 1, 1)
                compute_pass.end()

                # Phase 3: Build cell offsets (prefix sum)
                compute_pass = command_encoder.begin_compute_pass()
                compute_pass.set_pipeline(self.build_offsets_pipeline)
                compute_pass.set_bind_group(0, self.compute_bind_group, [], 0, 999999)
                compute_pass.dispatch_workgroups(1, 1, 1)
                compute_pass.end()

                # Phase 4: Fill grid with particle indices
                compute_pass = command_encoder.begin_compute_pass()
                compute_pass.set_pipeline(self.fill_grid_pipeline)
                compute_pass.set_bind_group(0, self.compute_bind_group, [], 0, 999999)
                compute_pass.dispatch_workgroups(particle_workgroups, 1, 1)
                compute_pass.end()

                # Phase 5: Detect and resolve collisions
                compute_pass = command_encoder.begin_compute_pass()
                compute_pass.set_pipeline(self.detect_collisions_pipeline)
                compute_pass.set_bind_group(0, self.compute_bind_group, [], 0, 999999)
                compute_pass.dispatch_workgroups(particle_workgroups, 1, 1)
                compute_pass.end()

                # Phase 6: Update physics (movement and boundary collision)
                compute_pass = command_encoder.begin_compute_pass()
                compute_pass.set_pipeline(self.update_physics_pipeline)
                compute_pass.set_bind_group(0, self.compute_bind_group, [], 0, 999999)
                compute_pass.dispatch_workgroups(particle_workgroups, 1, 1)
                compute_pass.end()

            # Render pass
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
            render_pass.set_vertex_buffer(0, self.particle_buffer)
            render_pass.set_vertex_buffer(1, self.colour_buffer)
            render_pass.draw(4, self.num_points)
            if self.show_grid:
                # Draw the grid
                render_pass.set_pipeline(self.line_pipeline)
                render_pass.set_bind_group(0, self.line_bind_group, [], 0, 999999)
                render_pass.set_vertex_buffer(0, self.grid_buffer)
                render_pass.draw(len(self.grid_lines), 1)

            render_pass.end()
            self.device.queue.submit([command_encoder.finish()])
            self._update_colour_buffer()
        except Exception as e:
            print(f"Failed to paint WebGPU content: {e}")

    def update_simulation_params(self) -> None:
        """
        Update the simulation parameters buffer with current wind values.
        """
        self.sim_params["dt"] = self.dt
        self.sim_params["wind_x"] = self.wind[0]
        self.sim_params["wind_y"] = self.wind[1]
        self.device.queue.write_buffer(
            buffer=self.sim_params_buffer,
            buffer_offset=0,
            data=self.sim_params.tobytes(),
        )

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
        elif key == Qt.Key_G:
            self.show_grid = not self.show_grid
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
        Initialize the numpy buffer for rendering.
        """
        print("initialize numpy buffer")
        width = int(self.width() * self.ratio)
        height = int(self.height() * self.ratio)
        self.frame_buffer = np.zeros([height, width, 4], dtype=np.uint8)

    def timerEvent(self, event: QTimerEvent) -> None:
        """
        This event is called at a regular interval (set by startTimer).
        Now that we're using compute shaders, this just triggers a redraw.

        Args:
            event: The QTimerEvent object, not used in this method but required by the API.
        """
        if self.animate:
            self.update()


def main():
    """
    Main function to run the application.
    Parses command line arguments and initializes the WebGPUScene.
    """
    parser = argparse.ArgumentParser(
        description="A WebGPU points demo with compute shader"
    )
    parser.add_argument(
        "-p",
        "--points",
        type=int,
        default=1000,
        help="The number of points to generate.",
    )
    args = parser.parse_args()
    app = QApplication(sys.argv)
    win = WebGPUScene(num_points=args.points)
    win.resize(1024, 720)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
