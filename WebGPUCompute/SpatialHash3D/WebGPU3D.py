#!/usr/bin/env -S uv run --active --script
import argparse
import sys
import traceback

import numpy as np
import wgpu
from ncca.ngl import Mat4, PerspMode, PrimData, Vec3, look_at, perspective
from ncca.ngl.webgpu import PipelineFactory, PipelineType, WebGPUWidget
from PySide6.QtCore import QElapsedTimer, Qt, QTimerEvent
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

SIM_WIDTH = 800
SIM_HEIGHT = 800
SIM_DEPTH = 800
GRID_CELL_SIZE = 50.0
PARTICLE_RADIUS = 1.0
SPHERE_PRECISION = 10


class WebGPUScene3D(WebGPUWidget):
    def __init__(self, num_points=10000, distribution: str = "random"):
        super().__init__()
        self.window_width: int = 1024
        self.window_height: int = 720
        self.setWindowTitle("WebGPU Compute 3D Collisions with Spatial Hashing")
        self.device = None
        self.compute_pipeline = None
        self.line_pipeline = None
        self.grid_buffer = None
        self.particle_buffer = None
        self.colour_buffer = None
        self.num_points = num_points
        self.msaa_sample_count = 4
        self.ratio = self.devicePixelRatio()
        self.animate = True  # False
        self.point_size = 6.0
        self.wind = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.timer = QElapsedTimer()
        self.dt = 0.0
        self.last_time = 0.0

        # 3D camera controls
        self.camera_distance = 500.0  # Much closer for debugging
        self.rotation_x = 30.0
        self.rotation_y = 45.0
        self.is_rotating = False
        self.last_mouse_pos = None
        self.project = perspective(
            fov=45.0, aspect=self.ratio, near=0.1, far=22000.0, mode=PerspMode.WebGPU
        )
        self.show_grid = True
        self.show_numbers = False

        self.gen_points(self.num_points, distribution)
        self._initialize_web_gpu()
        self.update()

    def _gen_grid_lines(self) -> None:
        grid_lines = []

        # Grid lines parallel to X axis
        for z in range(self.grid_depth + 1):
            for y in range(self.grid_height + 1):
                z_pos = -SIM_DEPTH / 2 + z * GRID_CELL_SIZE
                y_pos = -SIM_HEIGHT / 2 + y * GRID_CELL_SIZE
                grid_lines.append([-SIM_WIDTH / 2, y_pos, z_pos])
                grid_lines.append([SIM_WIDTH / 2, y_pos, z_pos])

        # Grid lines parallel to Y axis
        for z in range(self.grid_depth + 1):
            for x in range(self.grid_width + 1):
                z_pos = -SIM_DEPTH / 2 + z * GRID_CELL_SIZE
                x_pos = -SIM_WIDTH / 2 + x * GRID_CELL_SIZE
                grid_lines.append([x_pos, -SIM_HEIGHT / 2, z_pos])
                grid_lines.append([x_pos, SIM_HEIGHT / 2, z_pos])

        # Grid lines parallel to Z axis
        for y in range(self.grid_height + 1):
            for x in range(self.grid_width + 1):
                y_pos = -SIM_HEIGHT / 2 + y * GRID_CELL_SIZE
                x_pos = -SIM_WIDTH / 2 + x * GRID_CELL_SIZE
                grid_lines.append([x_pos, y_pos, -SIM_DEPTH / 2])
                grid_lines.append([x_pos, y_pos, SIM_DEPTH / 2])

        self.grid_lines = np.array(grid_lines, dtype=np.float32)

    def gen_points(self, num_points: int, distribution: str = "random") -> None:
        self.particle_data = np.zeros(
            num_points,
            dtype=[
                ("pos", "float32", 3),
                ("vel", "float32", 3),
            ],
        )

        if distribution == "random":
            # For debugging: place points at origin and simple positions
            if num_points <= 3:
                self.particle_data["pos"][0] = [0.0, 0.0, 0.0]  # Center
                if num_points > 1:
                    self.particle_data["pos"][1] = [100.0, 0.0, 0.0]  # Right
                if num_points > 2:
                    self.particle_data["pos"][2] = [0.0, 100.0, 0.0]  # Up
            else:
                self.particle_data["pos"][:, 0] = np.random.uniform(
                    -SIM_WIDTH / 2, SIM_WIDTH / 2, num_points
                )
                self.particle_data["pos"][:, 1] = np.random.uniform(
                    -SIM_HEIGHT / 2, SIM_HEIGHT / 2, num_points
                )
                self.particle_data["pos"][:, 2] = np.random.uniform(
                    -SIM_DEPTH / 2, SIM_DEPTH / 2, num_points
                )
        elif distribution == "equispaced":
            num_per_dim = int(np.ceil(num_points ** (1 / 3)))
            x = np.linspace(-SIM_WIDTH / 2, SIM_WIDTH / 2, num_per_dim)
            y = np.linspace(-SIM_HEIGHT / 2, SIM_HEIGHT / 2, num_per_dim)
            z = np.linspace(-SIM_DEPTH / 2, SIM_DEPTH / 2, num_per_dim)
            xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
            positions = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=-1)
            self.particle_data["pos"] = positions[:num_points].astype(np.float32)

        # Generate random velocities
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        phi = np.random.uniform(0, np.pi, num_points)
        min_speed = 5.0
        max_speed = 15.0
        speeds = np.random.uniform(min_speed, max_speed, num_points)

        directions = np.zeros((num_points, 3), dtype=np.float32)
        directions[:, 0] = speeds * np.sin(phi) * np.cos(theta)
        directions[:, 1] = speeds * np.sin(phi) * np.sin(theta)
        directions[:, 2] = speeds * np.cos(phi)

        self.particle_data["vel"] = directions
        self.colours = np.random.random((num_points, 3)).astype(np.float32)

    def _initialize_web_gpu(self) -> None:
        print("initializeWebGPU 3D")
        try:
            self.device = get_default_device()
            print(f"WebGPU device created: {self.device}")
            self._init_buffers()
            self._create_compute_pipeline()

            self.sphere_pipeline = PipelineFactory.create_pipeline(
                self.device,
                PipelineType.MULTI_COLOURED_INSTANCED_GEOMETRY,
            )
            self.sphere_pipeline.set_data(
                positions=self.particle_buffer,
                colours=self.colour_buffer,
                geometry_data=self.sphere_geometry_buffer,
            )
            print(f"DEBUG: Sphere pipeline created: {self.sphere_pipeline}")

            self.line_pipeline = PipelineFactory.create_pipeline(
                self.device,
                PipelineType.SINGLE_COLOUR_LINES,
                data_type="Vec3",
                topology=wgpu.PrimitiveTopology.line_list,
            )

            self.startTimer(16)
            self.timer.start()
            self.last_time = self.timer.elapsed() / 1000.0
            print("WebGPU 3D initialization complete")
        except Exception as e:
            print(f"Failed to initialize WebGPU 3D: {e}")
            import traceback

            traceback.print_exc()

    def _init_buffers(self):
        # Create full particle data buffer for compute shader
        self.compute_particle_buffer = self.device.create_buffer_with_data(
            data=self.particle_data.tobytes(),
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )

        # Extract just position data for rendering (first 12 bytes of each particle)
        position_data = self.particle_data["pos"].astype(np.float32)
        self.particle_buffer = self.device.create_buffer_with_data(
            data=position_data.tobytes(),
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.VERTEX
            | wgpu.BufferUsage.COPY_DST,
        )
        print(
            f"DEBUG: Position data shape: {position_data.shape}, min: {position_data.min()}, max: {position_data.max()}"
        )
        print(
            f"DEBUG: Position extents - X: [{position_data[:, 0].min():.2f}, {position_data[:, 0].max():.2f}]"
        )
        print(
            f"DEBUG: Position extents - Y: [{position_data[:, 1].min():.2f}, {position_data[:, 1].max():.2f}]"
        )
        print(
            f"DEBUG: Position extents - Z: [{position_data[:, 2].min():.2f}, {position_data[:, 2].max():.2f}]"
        )
        print(
            f"DEBUG: Simulation bounds: width={SIM_WIDTH}, height={SIM_HEIGHT}, depth={SIM_DEPTH}"
        )
        print(f"DEBUG: Grid cell size: {GRID_CELL_SIZE}")
        print(f"DEBUG: Particle radius: {PARTICLE_RADIUS}")

        # Instanced sphere geometry pipeline expects per-instance colour as a
        # tightly packed Vec3 (no padding), unlike the old points pipeline.
        self.colour_buffer = self.device.create_buffer_with_data(
            data=self.colours.astype(np.float32).tobytes(),
            usage=wgpu.BufferUsage.VERTEX,
        )

        # Sphere geometry shared by every particle instance, sized to PARTICLE_RADIUS
        sphere_data = PrimData.sphere(
            radius=PARTICLE_RADIUS, precision=SPHERE_PRECISION
        )
        self.sphere_geometry_buffer = self.device.create_buffer_with_data(
            data=sphere_data.tobytes(),
            usage=wgpu.BufferUsage.VERTEX,
        )

        self.grid_width = int(np.ceil(SIM_WIDTH / GRID_CELL_SIZE))
        self.grid_height = int(np.ceil(SIM_HEIGHT / GRID_CELL_SIZE))
        self.grid_depth = int(np.ceil(SIM_DEPTH / GRID_CELL_SIZE))
        self.total_cells = self.grid_width * self.grid_height * self.grid_depth
        self._gen_grid_lines()

        self.grid_buffer = self.device.create_buffer_with_data(
            data=self.grid_lines.tobytes(),
            usage=wgpu.BufferUsage.VERTEX,
        )
        print(
            f"Grid: {self.grid_width}x{self.grid_height}x{self.grid_depth} = {self.total_cells} cells"
        )

        self.cell_particle_count_buffer = self.device.create_buffer(
            size=self.total_cells * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        self.cell_count_staging_buffer = self.device.create_buffer(
            size=self.total_cells * 4,
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )

        self.sim_params = np.zeros(
            (),
            dtype=[
                ("dt", "float32"),
                ("width", "float32"),
                ("height", "float32"),
                ("depth", "float32"),
                ("wind_x", "float32"),
                ("wind_y", "float32"),
                ("wind_z", "float32"),
                ("grid_width", "uint32"),
                ("grid_height", "uint32"),
                ("grid_depth", "uint32"),
                ("cell_size", "float32"),
                ("particle_radius", "float32"),
                ("padding", "uint32", 3),
            ],
        )
        self.sim_params["width"] = SIM_WIDTH
        self.sim_params["height"] = SIM_HEIGHT
        self.sim_params["depth"] = SIM_DEPTH
        self.sim_params["wind_x"] = 0.0
        self.sim_params["wind_y"] = 0.0
        self.sim_params["wind_z"] = 0.0
        self.sim_params["grid_width"] = self.grid_width
        self.sim_params["grid_height"] = self.grid_height
        self.sim_params["grid_depth"] = self.grid_depth
        self.sim_params["cell_size"] = GRID_CELL_SIZE
        self.sim_params["particle_radius"] = PARTICLE_RADIUS

        self.sim_params_buffer = self.device.create_buffer_with_data(
            data=self.sim_params.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="Simulation Parameters Buffer",
        )

        self.grid_indices_buffer = self.device.create_buffer(
            size=self.num_points * 4,
            usage=wgpu.BufferUsage.STORAGE,
            label="Grid Indices Buffer",
        )

        self.grid_offsets_buffer = self.device.create_buffer(
            size=self.total_cells * 4,
            usage=wgpu.BufferUsage.STORAGE,
            label="Grid Offsets Buffer",
        )

    def _init_textures(self):
        """Initialize WebGPU textures and views for rendering."""
        if self.device is None:
            print("Cannot initialize textures - device not available")
            return

        self.texture_size = (self.window_width, self.window_height, 1)

        # Create depth texture
        self.depth_texture = self.device.create_texture(
            size=self.texture_size,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.depth24plus,
            mip_level_count=1,
            sample_count=1,
        )
        self.depth_buffer_view = self.depth_texture.create_view()

        # Create multisample color texture
        self.multisample_texture = self.device.create_texture(
            size=self.texture_size,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.bgra8unorm,
            mip_level_count=1,
            sample_count=self.msaa_sample_count,
        )
        self.multisample_texture_view = self.multisample_texture.create_view()

        # Create color buffer texture
        self.colour_buffer_texture = self.device.create_texture(
            size=self.texture_size,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.TEXTURE_BINDING,
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.bgra8unorm,
            mip_level_count=1,
            sample_count=1,
        )
        self.depth_buffer_view = self.depth_texture.create_view()

        # Create multisample color texture
        self.multisample_texture = self.device.create_texture(
            size=self.texture_size,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.bgra8unorm,
            mip_level_count=1,
            sample_count=self.msaa_sample_count,
        )
        self.multisample_texture_view = self.multisample_texture.create_view()

        # Create color buffer texture
        self.colour_buffer_texture = self.device.create_texture(
            size=self.texture_size,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.TEXTURE_BINDING,
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.bgra8unorm,
            mip_level_count=1,
            sample_count=1,
        )
        self.colour_buffer_texture_view = self.colour_buffer_texture.create_view()

        print(f"Textures initialized: {self.texture_size}")

    def _create_compute_pipeline(self) -> None:
        with open("CollisionCompute3D.wgsl", "r") as f:
            compute_shader_code = f.read()
            compute_shader_module = self.device.create_shader_module(
                code=compute_shader_code
            )

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

        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )

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

        self.compute_bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.compute_particle_buffer,
                        "offset": 0,
                        "size": self.compute_particle_buffer.size,
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

    def resizeWebGPU(self, w, h) -> None:
        """
        Called whenever the window is resized.
        It's crucial to update the viewport and projection matrix here.

        Args:
            w: The new width of the window.
            h: The new height of the window.
        """

        # Update texture size to match window dimensions
        self.texture_size = (w, h)

        # Update projection matrix
        self.project = perspective(
            45.0, w / h if h > 0 else 1, 0.1, 10000.0, PerspMode.WebGPU
        )

        self.update()

    def _compute_pass(self):
        try:
            command_encoder = self.device.create_command_encoder()

            if self.animate:
                print("DEBUG: Running compute pass")
                self.update_simulation_params()

                particle_workgroups = (self.num_points + 63) // 64
                cell_workgroups = (self.total_cells + 63) // 64
                print(
                    f"DEBUG: Workgroups - particles: {particle_workgroups}, cells: {cell_workgroups}"
                )

                print("DEBUG: Clear grid")
                compute_pass = command_encoder.begin_compute_pass()
                compute_pass.set_pipeline(self.clear_grid_pipeline)
                compute_pass.set_bind_group(0, self.compute_bind_group, [], 0, 999999)
                compute_pass.dispatch_workgroups(cell_workgroups, 1, 1)
                compute_pass.end()

                print("DEBUG: Count particles")
                compute_pass = command_encoder.begin_compute_pass()
                compute_pass.set_pipeline(self.count_particles_pipeline)
                compute_pass.set_bind_group(0, self.compute_bind_group, [], 0, 999999)
                compute_pass.dispatch_workgroups(particle_workgroups, 1, 1)
                compute_pass.end()

                print("DEBUG: Build offsets")
                compute_pass = command_encoder.begin_compute_pass()
                compute_pass.set_pipeline(self.build_offsets_pipeline)
                compute_pass.set_bind_group(0, self.compute_bind_group, [], 0, 999999)
                compute_pass.dispatch_workgroups(1, 1, 1)
                compute_pass.end()

                print("DEBUG: Fill grid")
                compute_pass = command_encoder.begin_compute_pass()
                compute_pass.set_pipeline(self.fill_grid_pipeline)
                compute_pass.set_bind_group(0, self.compute_bind_group, [], 0, 999999)
                compute_pass.dispatch_workgroups(particle_workgroups, 1, 1)
                compute_pass.end()

                print("DEBUG: Detect collisions")
                compute_pass = command_encoder.begin_compute_pass()
                compute_pass.set_pipeline(self.detect_collisions_pipeline)
                compute_pass.set_bind_group(0, self.compute_bind_group, [], 0, 999999)
                compute_pass.dispatch_workgroups(particle_workgroups, 1, 1)
                compute_pass.end()

                print("DEBUG: Update physics")
                compute_pass = command_encoder.begin_compute_pass()
                compute_pass.set_pipeline(self.update_physics_pipeline)
                compute_pass.set_bind_group(0, self.compute_bind_group, [], 0, 999999)
                compute_pass.dispatch_workgroups(particle_workgroups, 1, 1)
                compute_pass.end()

                print("DEBUG: Compute pass completed")
            self.device.queue.submit([command_encoder.finish()])
        except Exception as e:
            print(f"Failed to run compute pass: {e}")
            import traceback

            traceback.print_exc()

    def _render_pass(self):
        try:
            # Check if texture views are initialized
            if (
                not hasattr(self, "multisample_texture_view")
                or not hasattr(self, "colour_buffer_texture_view")
                or not hasattr(self, "depth_buffer_view")
            ):
                print("Texture views not initialized - creating them now")
                self._init_textures()

            command_encoder = self.device.create_command_encoder()
            render_pass = command_encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": self.multisample_texture_view,
                        "resolve_target": self.colour_buffer_texture_view,
                        "load_op": wgpu.LoadOp.clear,
                        "store_op": wgpu.StoreOp.store,
                        "clear_value": (0.4, 0.4, 0.4, 1.0),  #
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

            if self.show_grid:
                self.line_pipeline.set_data(positions=self.grid_buffer)
                self.line_pipeline.render(render_pass=render_pass)
            # particle_buffer is mutated in place by the compute pass, and set_data
            # was already bound once in _initialize_web_gpu; re-calling it here with
            # the same GPUBuffer objects would destroy them (pipeline.set_data
            # destroys any previously bound buffer before rebinding).
            self.sphere_pipeline.render(
                render_pass=render_pass, num_instances=self.num_points
            )

            render_pass.end()
            # Update position buffer for rendering from compute results
            if self.animate:
                command_encoder2 = self.device.create_command_encoder()
                command_encoder2.copy_buffer_to_buffer(
                    self.compute_particle_buffer,
                    0,
                    self.particle_buffer,
                    0,
                    self.particle_buffer.size,
                )
                self.device.queue.submit([command_encoder2.finish()])

            self.device.queue.submit([command_encoder.finish()])
        except Exception as e:
            print(f"Failed to paint WebGPU content: {e}")
            import traceback

            traceback.print_exc()

    def paintWebGPU(self) -> None:
        print("Painting WebGPU content")
        current_time = self.timer.elapsed() / 1000.0
        self.dt = current_time - self.last_time
        self.last_time = current_time

        self.render_text(
            10,
            25,
            f" Spheres {self.num_points}  Wind [{self.wind[0]:.02f}, {self.wind[1]:.02f}, {self.wind[2]:.02f}] dt: {self.dt:.4f} FPS: {1.0 / self.dt if self.dt > 0 else 0:.2f}",
            size=20,
            colour=Qt.yellow,
        )
        self.render_text(
            10,
            50,
            f" Camera: Distance={self.camera_distance:.0f} RotX={self.rotation_x:.0f}° RotY={self.rotation_y:.0f}°",
            size=16,
            colour=Qt.cyan,
        )

        self._compute_pass()
        self._render_pass()

    def update_simulation_params(self) -> None:
        self.sim_params["dt"] = self.dt
        self.sim_params["wind_x"] = self.wind[0]
        self.sim_params["wind_y"] = self.wind[1]
        self.sim_params["wind_z"] = self.wind[2]
        self.sim_params["width"] = SIM_WIDTH
        self.sim_params["height"] = SIM_HEIGHT
        self.sim_params["depth"] = SIM_DEPTH
        self.sim_params["grid_width"] = self.grid_width
        self.sim_params["grid_height"] = self.grid_height
        self.sim_params["grid_depth"] = self.grid_depth
        self.sim_params["cell_size"] = GRID_CELL_SIZE
        self.sim_params["particle_radius"] = PARTICLE_RADIUS
        self.device.queue.write_buffer(
            buffer=self.sim_params_buffer,
            buffer_offset=0,
            data=self.sim_params.tobytes(),
        )

    def update_uniform_buffers(self) -> None:
        # Calculate camera position based on distance and rotation
        eye_x = (
            self.camera_distance
            * np.cos(np.radians(self.rotation_y))
            * np.cos(np.radians(self.rotation_x))
        )
        eye_y = self.camera_distance * np.sin(np.radians(self.rotation_x))
        eye_z = (
            self.camera_distance
            * np.sin(np.radians(self.rotation_y))
            * np.cos(np.radians(self.rotation_x))
        )

        self.view = look_at(
            Vec3(eye_x, eye_y, eye_z),
            Vec3(0, 0, 0),
            Vec3(0, 1, 0),
        )

        # Spheres are real 3D geometry (not billboarded sprites), so use the
        # full camera view/projection directly.
        self.mvp_matrix = (self.project @ self.view).to_numpy().astype(np.float32)
        self.view_matrix = self.view.to_numpy().astype(np.float32)

        # point_size doubles as a sphere scale multiplier (6.0 == neutral, no rescale)
        sphere_scale = self.point_size / 6.0
        instance_transform = (
            Mat4.scale(sphere_scale, sphere_scale, sphere_scale)
            .to_numpy()
            .astype(np.float32)
        )
        self.sphere_pipeline.update_uniforms(
            mvp=self.mvp_matrix,
            view_matrix=self.view_matrix,
            instance_transform=instance_transform,
        )
        self.line_pipeline.update_uniforms(mvp=self.mvp_matrix)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        handled = False

        if key == Qt.Key.Key_Escape:
            self.close()
            handled = True
        elif key == Qt.Key.Key_A:
            self.animate = not self.animate
            handled = True
        elif key == Qt.Key.Key_G:
            self.show_grid = not self.show_grid
            handled = True
        elif key == Qt.Key.Key_Space:
            self.wind[0] = 0.0
            self.wind[1] = 0.0
            self.wind[2] = 0.0
            self.camera_distance = 1500.0
            self.rotation_x = 30.0
            self.rotation_y = 45.0
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
        elif key == Qt.Key.Key_PageUp:
            self.wind[2] += 0.1
            handled = True
        elif key == Qt.Key.Key_PageDown:
            self.wind[2] -= 0.1
            handled = True

        if handled:
            # Don't call update() here - Qt will handle repaint automatically
            pass
        else:
            if self.parent():
                self.parent().keyPressEvent(event)
            super().keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.is_rotating = True
            try:
                self.last_mouse_pos = event.position()
            except Exception:
                self.last_mouse_pos = event.pos()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.is_rotating = False

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.is_rotating and self.last_mouse_pos is not None:
            try:
                cur = event.position()
                dx = cur.x() - self.last_mouse_pos.x()
                dy = cur.y() - self.last_mouse_pos.y()
            except Exception:
                cur = event.pos()
                dx = cur.x() - self.last_mouse_pos.x()
                dy = cur.y() - self.last_mouse_pos.y()

            self.rotation_y += dx * 0.5
            self.rotation_x += dy * 0.5
            self.rotation_x = np.clip(self.rotation_x, -89, 89)

            self.last_mouse_pos = cur
            # Only update if really needed to avoid Qt painter conflicts
            if abs(dx) > 1 or abs(dy) > 1:
                self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        scale_factor = 1.0 + (delta / 1200.0)
        new_distance = self.camera_distance / scale_factor
        old_distance = self.camera_distance
        self.camera_distance = np.clip(new_distance, 100, 5000)
        # Only update if distance changed significantly
        if abs(new_distance - old_distance) > 10:
            self.update()

    def timerEvent(self, event: QTimerEvent) -> None:
        print("Update")
        self.update()


class DebugApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        print("Running in full debug mode")

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(
        description="A WebGPU 3D points demo with compute shader and spatial hashing"
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

    win = WebGPUScene3D(num_points=args.points, distribution=args.distribution)
    win.resize(1024, 720)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
