#!/usr/bin/env -S uv run --active --script
import argparse
import sys
import traceback

import numpy as np
import wgpu
from ncca.ngl import PerspMode, Vec2, logger, ortho
from ncca.ngl.webgpu import PipelineFactory, PipelineType, WebGPUWidget
from PySide6.QtCore import QElapsedTimer, Qt, QTimerEvent
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

SIM_WIDTH = 800
SIM_HEIGHT = 800
GRID_CELL_SIZE = 50.0  # Size of each grid cell
PARTICLE_RADIUS = 1.0  # Collision radius for particles


class WebGPUScene(WebGPUWidget):
    """
    A concrete implementation of WebGPUWidget for a WebGPU scene.

    This class implements the abstract methods to provide functionality for initializing,
    painting, and resizing the WebGPU context.
    """

    def __init__(self, num_points=10000, distribution: str = "random"):
        super().__init__()
        self.window_width: int = 1024  # Window width
        self.window_height: int = 720  # Window height
        self.setWindowTitle("WebGPU Compute Collisions with Spatial Hashing")
        self.device = None
        self.compute_pipeline = None
        self.line_pipeline = None
        self.grid_buffer = None
        self.line_bind_group = None
        self.particle_buffer = None
        self.colour_buffer = None
        self.num_points = num_points
        self.msaa_sample_count = 4
        self.ratio = self.devicePixelRatio()
        self.animate = False
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = 1.0
        self.is_panning = False
        self.last_mouse_pos = None
        self.show_grid = True
        self.show_numbers = True
        self.point_size = 1.0
        self.wind = np.array([0.0, 0.0], dtype=np.float32)
        self.timer = QElapsedTimer()
        self.dt = 0.0
        self.last_time = 0.0
        #  pan (world-space center) so we can zoom around mouse and pan the view.
        self.pan = np.array([0.0, 0.0], dtype=np.float32)
        # Track last mouse position (QPointF) for right-drag panning.
        self._last_mouse_pos = None

        self.gen_points(self.num_points, distribution)
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

    def gen_points(self, num_points: int, distribution: str = "random") -> None:
        """
        Generates 2D points with associated positions, directions, and colours.

        This function initializes particle data in a structured format suitable for compute shaders.

        Args:
            num_points: The number of points to generate.
            distribution: The distribution of the points ('random' or 'equispaced').
        """
        # Create structured array matching the Particle struct in the compute shader
        self.particle_data = np.zeros(
            num_points,
            dtype=[
                ("pos", "float32", 2),
                ("vel", "float32", 2),
            ],
        )

        if distribution == "random":
            # Generate positions in 2D space the size of the simulation with 0,0 the center
            self.particle_data["pos"][:, 0] = np.random.uniform(
                -SIM_WIDTH / 2, SIM_WIDTH / 2, num_points
            )
            self.particle_data["pos"][:, 1] = np.random.uniform(
                -SIM_HEIGHT / 2, SIM_HEIGHT / 2, num_points
            )
        elif distribution == "equispaced":
            # calculate number of points per row/col
            num_per_side = int(np.ceil(np.sqrt(num_points)))
            # create a grid of points
            x = np.linspace(-SIM_WIDTH / 2, SIM_WIDTH / 2, num_per_side)
            y = np.linspace(-SIM_HEIGHT / 2, SIM_HEIGHT / 2, num_per_side)
            xv, yv = np.meshgrid(x, y)
            # flatten and take the first num_points
            positions = np.stack([xv.flatten(), yv.flatten()], axis=-1)
            self.particle_data["pos"] = positions[:num_points].astype(np.float32)

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
            print(f"WebGPU device created: {self.device}")
            self._init_buffers()
            self._create_compute_pipeline()
            # stride must be 16 as pos and dir in buffer are both Vec2

            self.point_pipeline = PipelineFactory.create_pipeline(
                self.device,
                PipelineType.MULTI_COLOURED_POINTS,
                data_type="Vec2",
                stride=16,
            )
            self.line_pipeline = PipelineFactory.create_pipeline(
                self.device,
                PipelineType.SINGLE_COLOUR_LINES,
                data_type="Vec2",
                topology=wgpu.PrimitiveTopology.line_list,
            )
            self.startTimer(16)
            self.timer.start()
            self.last_time = self.timer.elapsed() / 1000.0
            print("WebGPU initialization complete")
        except Exception as e:
            print(f"Failed to initialize WebGPU: {e}")
            import traceback

            traceback.print_exc()

    def _init_buffers(self):
        # Create a storage buffer for particles (used by compute shader and rendering)
        self.particle_buffer = self.device.create_buffer_with_data(
            data=self.particle_data.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.VERTEX,
        )

        # Pad colours for 16-byte alignment
        padded_colours = np.zeros((self.num_points, 4), dtype=np.float32)
        padded_colours[:, :3] = self.colours

        # Create a buffer for the colours
        self.colour_buffer = self.device.create_buffer_with_data(
            data=padded_colours.tobytes(),
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

        # Create a staging buffer for reading back cell particle counts
        self.cell_particle_count_buffer = self.device.create_buffer(
            size=self.total_cells * 4,  # u32 per cell
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,  # Add COPY_SRC!
        )
        self.cell_count_staging_buffer = self.device.create_buffer(
            size=self.total_cells * 4,  # u32 per cell
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )

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
            label="Simulation Parameters Buffer",
        )

        # Create grid buffers for spatial hashing
        self.grid_indices_buffer = self.device.create_buffer(
            size=self.num_points * 4,  # u32 per particle
            usage=wgpu.BufferUsage.STORAGE,
            label="Grid Indices Buffer",
        )

        self.grid_offsets_buffer = self.device.create_buffer(
            size=self.total_cells * 4,  # u32 per cell
            usage=wgpu.BufferUsage.STORAGE,
            label="Grid Offsets Buffer",
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

    def read_cell_particle_counts(self) -> np.ndarray:
        """
        Synchronous wrapper for reading cell particle counts.
        Uses wgpu synchronous API.

        Returns:
            numpy array of shape (grid_height, grid_width) containing particle counts per cell
        """
        # Create a command encoder for the copy operation
        command_encoder = self.device.create_command_encoder()

        # Copy from the GPU buffer to the staging buffer
        command_encoder.copy_buffer_to_buffer(
            source=self.cell_particle_count_buffer,
            source_offset=0,
            destination=self.cell_count_staging_buffer,
            destination_offset=0,
            size=self.total_cells * 4,  # size of uint32
        )
        # Submit the copy command and wait for completion
        self.device.queue.submit([command_encoder.finish()])

        self.cell_count_staging_buffer.map_sync(mode=wgpu.MapMode.READ)
        # Read the data
        data = self.cell_count_staging_buffer.read_mapped()
        # Convert to numpy array
        counts = np.frombuffer(data, dtype=np.uint32).copy()

        # Unmap the buffer
        self.cell_count_staging_buffer.unmap()

        # Reshape to 2D grid
        counts_2d = counts.reshape(self.grid_height, self.grid_width)
        return counts_2d

    def sim_to_qt_transformed(self, x: float, y: float) -> tuple:
        """Transform simulation coordinates to screen coordinates accounting for zoom and pan."""
        pixel_w = self.width()
        pixel_h = self.height()

        # Apply the same transformation as the WebGPU projection matrix
        # First transform by zoom and pan, then map to screen coordinates
        transformed_x = (x - self.pan[0]) / self.zoom
        transformed_y = (y - self.pan[1]) / self.zoom

        # Map from transformed simulation range to screen coordinates
        qt_x = (transformed_x + SIM_WIDTH / 2.0) * (pixel_w / SIM_WIDTH)
        qt_y = (SIM_HEIGHT / 2.0 - transformed_y) * (pixel_h / SIM_HEIGHT)

        return qt_x, qt_y

    def calculate_font_size(self, base_size: int = 10) -> int:
        """Calculate appropriate font size based on zoom level."""
        # Scale font size inversely with square root of zoom for perceptual consistency
        scaled_size = base_size / np.sqrt(self.zoom)
        # Clamp to reasonable bounds
        return max(6, min(20, int(scaled_size)))

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

    def _compute_pass(self):
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
            self.device.queue.submit([command_encoder.finish()])
        except Exception as e:
            print(f"Failed to paint WebGPU content: {e}")

    def _read_back_pass(self):
        # After all compute passes are complete, read back the data
        cell_counts = self.read_cell_particle_counts()

        # Print statistics
        total_particles = np.sum(cell_counts)
        max_in_cell = np.max(cell_counts)
        avg_per_cell = np.mean(cell_counts)
        non_empty_cells = np.count_nonzero(cell_counts)
        # print("Frame Stats:")
        # print(f"  Total particles counted: {total_particles}")
        # print(f"  Max particles in a cell: {max_in_cell}")
        # print(f"  Avg particles per cell: {avg_per_cell:.2f}")
        # print(f"  Non-empty cells: {non_empty_cells}/{self.total_cells}")
        # print("-" * 50)

        def sim_to_qt(x, y):
            # # widget size in device pixels (account for HiDPI)
            # if hasattr(self, "devicePixelRatioF"):
            #     dpr = self.devicePixelRatioF() or 1.0
            # else:
            #     dpr = 1.0
            pixel_w = self.width()
            pixel_h = self.height()

            # map from simulation range (-SIM_WIDTH/2..SIM_WIDTH/2) to (0..pixel_w)
            qt_x = (x + SIM_WIDTH / 2.0) * (pixel_w / SIM_WIDTH)
            # flip Y: simulation y=+SIM_HEIGHT/2 -> top of widget (small y in Qt)
            qt_y = (SIM_HEIGHT / 2.0 - y) * (pixel_h / SIM_HEIGHT)

            return qt_x, qt_y

        def sim_to_qt_transformed(x, y):
            """Transform simulation coordinates to screen coordinates accounting for zoom and pan."""
            pixel_w = self.width()
            pixel_h = self.height()

            # Apply the same transformation as the WebGPU projection matrix
            # First transform by zoom and pan, then map to screen coordinates
            transformed_x = (x - self.pan[0]) / self.zoom
            transformed_y = (y - self.pan[1]) / self.zoom

            # Map from transformed simulation range to screen coordinates
            qt_x = (transformed_x + SIM_WIDTH / 2.0) * (pixel_w / SIM_WIDTH)
            qt_y = (SIM_HEIGHT / 2.0 - transformed_y) * (pixel_h / SIM_HEIGHT)

            return qt_x, qt_y

        def calculate_font_size(base_size=10):
            """Calculate appropriate font size based on zoom level."""
            # Scale font size inversely with square root of zoom for perceptual consistency
            scaled_size = base_size / np.sqrt(self.zoom)
            # Clamp to reasonable bounds
            return max(6, min(20, scaled_size))

        # Calculate base font size once
        font_size = calculate_font_size()

        # Calculate cell size in screen pixels to determine if text should be rendered
        cell_x_start, cell_y_start = sim_to_qt_transformed(
            -SIM_WIDTH / 2, -SIM_HEIGHT / 2
        )
        cell_x_end, cell_y_end = sim_to_qt_transformed(
            -SIM_WIDTH / 2 + GRID_CELL_SIZE, -SIM_HEIGHT / 2 + GRID_CELL_SIZE
        )
        cell_pixel_width = abs(cell_x_end - cell_x_start)
        cell_pixel_height = abs(cell_y_end - cell_y_start)

        # Only render text if cells are large enough to be readable
        if cell_pixel_width > 15 and cell_pixel_height > 15:
            for row in range(self.grid_height):
                for col in range(self.grid_width):
                    x = -SIM_WIDTH / 2 + col * GRID_CELL_SIZE
                    y = -SIM_HEIGHT / 2 + row * GRID_CELL_SIZE
                    # Use transformed coordinates to account for zoom and pan
                    qt_x, qt_y = sim_to_qt_transformed(x, y)

                    # Only render if text is visible on screen
                    if 0 <= qt_x <= self.width() and 0 <= qt_y <= self.height():
                        # Convert to integers for Qt text rendering
                        self.render_text(
                            int(qt_x),
                            int(qt_y),
                            f"{cell_counts[row, col]}",
                            size=font_size,
                            colour=Qt.yellow,
                        )

    def _render_pass(self):
        try:
            # Create a new command encoder for the render pass
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
            self.point_pipeline.set_data(
                positions=self.particle_buffer, colours=self.colour_buffer
            )
            self.point_pipeline.render(
                render_pass=render_pass
            )  # num_points=self.num_points)

            if self.show_grid:
                self.line_pipeline.set_data(positions=self.grid_buffer)
                self.line_pipeline.render(render_pass=render_pass)

            render_pass.end()
            self.device.queue.submit([command_encoder.finish()])
            self._update_colour_buffer()
        except Exception as e:
            print(f"Failed to paint WebGPU content: {e}")

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
        self._compute_pass()
        if self.show_numbers:
            self._read_back_pass()
        self._render_pass()

    def update_simulation_params(self) -> None:
        """
        Update the simulation parameters buffer with current wind values.
        """
        self.sim_params["dt"] = self.dt
        self.sim_params["wind_x"] = self.wind[0]
        self.sim_params["wind_y"] = self.wind[1]
        self.sim_params["width"] = SIM_WIDTH
        self.sim_params["height"] = SIM_HEIGHT
        self.sim_params["grid_width"] = self.grid_width
        self.sim_params["grid_height"] = self.grid_height
        self.sim_params["cell_size"] = GRID_CELL_SIZE
        self.sim_params["particle_radius"] = PARTICLE_RADIUS
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
        self.point_pipeline.update_uniforms(
            mvp=proj.to_numpy(), point_size=self.point_size
        )
        self.line_pipeline.update_uniforms(mvp=proj.to_numpy())

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Handles keyboard press events.

        Args:
            event: The QKeyEvent object containing information about the key press.
        """
        key = event.key()
        handled = False
        if key == Qt.Key.Key_Escape:
            self.close()  # Exit the application
            handled = True
        elif key == Qt.Key.Key_A:
            self.animate = not self.animate
            # Update the GUI checkbox to match
            if hasattr(self.parent(), "control_panel"):
                self.parent().control_panel.animate_checkbox.setChecked(self.animate)
            handled = True
        elif key == Qt.Key.Key_G:
            self.show_grid = not self.show_grid
            handled = True
        elif key == Qt.Key.Key_N:
            self.show_numbers = not self.show_numbers
            handled = True
        elif key == Qt.Key.Key_Space:
            self.wind[0] = 0
            self.wind[1] = 0
            self.zoom = 1.0
            # Reset pan as well when space is pressed
            self.pan[:] = 0.0
            handled = True
        elif key == Qt.Key.Key_Up:
            self.wind[1] += 0.1
            handled = True
        elif key == Qt.Key.Key_Down:
            self.wind[1] -= 0.1
            handled = True
        elif key == Qt.Key.Key_Left:
            self.wind[0] -= 0.1
            handled = True
        elif key == Qt.Key.Key_Right:
            self.wind[0] += 0.1
            handled = True

        if handled:
            # Trigger a redraw to apply changes
            self.update()
        else:
            # Pass unhandled events to parent window
            if self.parent():
                self.parent().keyPressEvent(event)
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


class DebugApplication(QApplication):
    """
    A custom QApplication subclass for improved debugging.

    By default, Qt's event loop can suppress exceptions that occur within event handlers
    (like paintGL or mouseMoveEvent), making it very difficult to debug as the application
    may simply crash or freeze without any error message. This class overrides the `notify`
    method to catch these exceptions, print a full traceback to the console, and then
    re-raise the exception to halt the program, making the error immediately visible.
    """

    def __init__(self, argv):
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver, event):
        """
        Overrides the central event handler to catch and report exceptions.
        """
        try:
            # Attempt to process the event as usual
            return super().notify(receiver, event)
        except Exception:
            # If an exception occurs, print the full traceback
            traceback.print_exc()
            # Re-raise the exception to stop the application
            raise


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
    dist_group = parser.add_mutually_exclusive_group()
    dist_group.add_argument(
        "-r",
        "--random",
        action="store_const",
        dest="distribution",
        const="random",
        help="Randomly distribute points (default).",
    )
    dist_group.add_argument(
        "-e",
        "--equispaced",
        action="store_const",
        dest="distribution",
        const="equispaced",
        help="Equispaced point distribution.",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Run in full debug mode"
    )
    parser.set_defaults(distribution="random")
    args = parser.parse_args()

    if args.debug:
        print("Running in debug mode")
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    win = WebGPUScene(num_points=args.points, distribution=args.distribution)
    win.resize(1024, 720)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
