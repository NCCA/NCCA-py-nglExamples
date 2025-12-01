import numpy as np
import wgpu
from ncca.ngl import (
    Mat4,
    PerspMode,
    PrimData,
    Prims,
    Transform,
    Vec3,
    look_at,
    perspective,
)
from PySide6.QtCore import Qt, Slot
from WebGPUWidget import WebGPUWidget
from wgpu.utils import get_default_device


class WebGPUScene(WebGPUWidget):
    """
    A concrete implementation of WebGPUWidget for a teapot scene.
    This class handles the WebGPU rendering pipeline, user input, and scene management.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("WebGPU Teapot PBR")
        self.device = None

        # Scene and camera setup
        self.model_transform = Transform()
        self.eye = Vec3(0.0, 2.0, 4.0)
        self.view = look_at(self.eye, Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.project = perspective(
            45.0, self.width() / self.height(), 0.1, 100.0, PerspMode.WebGPU
        )

        # --- Window and UI Attributes ---
        self.window_width: int = 1024  # Window width
        self.window_height: int = 720  # Window height

        # --- Mouse Control Attributes for Camera Manipulation ---
        self.rotate: bool = False  # Flag to check if the scene is being rotated
        self.translate: bool = (
            False  # Flag to check if the scene is being translated (panned)
        )
        self.spin_x_face: int = 0  # Accumulated rotation around the X-axis
        self.spin_y_face: int = 0  # Accumulated rotation around the Y-axis
        self.original_x_rotation: int = (
            0  # Initial X position of the mouse when a rotation starts
        )
        self.original_y_rotation: int = (
            0  # Initial Y position of the mouse when a rotation starts
        )
        self.original_x_pos: int = (
            0  # Initial X position of the mouse when a translation starts
        )
        self.original_y_pos: int = (
            0  # Initial Y position of the mouse when a translation starts
        )
        self.INCREMENT: float = 0.01  # Sensitivity for translation
        self.ZOOM: float = 0.1  # Sensitivity for zooming
        self.model_position = Vec3(0, 0, 0)
        # WebGPU resources
        self.pipeline = None
        self.vertex_buffer = None
        self.teapot_size = 0
        self.transform_buffer = None
        self.material_buffer = None
        self.light_buffer = None
        self.view_buffer = None
        self.bind_group_0 = None
        self.bind_group_1 = None

        # Uniform data
        self._material = {
            "albedo": Vec3(0.950, 0.71, 0.29),
            "metallic": 1.02,
            "roughness": 0.38,
            "ao": 0.2,
        }
        self._light_colour = Vec3(400.0, 400.0, 400.0)
        self._light_position = Vec3(0.0, 2.0, 2.0)

        self._initialize_web_gpu()
        self.update()

    @Slot(float, float, float)
    def set_model_rotation(self, x: float, y: float, z: float):
        self.model_transform.set_rotation(x, y, z)
        self.update()

    @Slot(float, float, float)
    def set_model_position(self, x: float, y: float, z: float):
        self.model_transform.set_position(x, y, z)
        self.update()

    @Slot(float, float, float)
    def set_model_scale(self, x: float, y: float, z: float):
        self.model_transform.set_scale(x, y, z)
        self.update()

    @Slot(float, float, float)
    def set_colour(self, r: float, g: float, b: float):
        self._material["albedo"].set(r, g, b)
        self._update_material_buffer()
        self.update()

    @Slot(float)
    def set_metallic(self, m: float):
        self._material["metallic"] = m
        self._update_material_buffer()
        self.update()

    @Slot(float)
    def set_roughness(self, r: float):
        self._material["roughness"] = r
        self._update_material_buffer()
        self.update()

    @Slot(float)
    def set_ao(self, ao: float):
        self._material["ao"] = ao
        self._update_material_buffer()
        self.update()

    def set_light(self, position: Vec3, colour: Vec3):
        self._light_position = position
        if colour is not None:
            self._light_colour = colour
        self._update_light_buffer()
        self.update()

    def _initialize_web_gpu(self):
        try:
            self.device = get_default_device()
            self._create_render_buffer()
            self._create_render_pipeline()
            self.start_update_timer(16)  # ~60 FPS
        except Exception as e:
            print(f"Failed to initialize WebGPU: {e}")
            exit(1)

    def _create_render_pipeline(self):
        with open("PBRShader.wgsl", "r") as f:
            shader_code = f.read()
        shader_module = self.device.create_shader_module(code=shader_code)

        self.pipeline = self.device.create_render_pipeline(
            label="teapot_pipeline",
            layout="auto",
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 8 * 4,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x3", "offset": 12, "shader_location": 1},
                            {"format": "float32x2", "offset": 24, "shader_location": 2},
                        ],
                    }
                ],
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
            multisample={
                "count": self.msaa_sample_count,
                "mask": 0xFFFFFFFF,
                "alpha_to_coverage_enabled": False,
            },
        )

        teapot = PrimData.primitive(Prims.TEAPOT.value)
        self.teapot_size = teapot.size // 8
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=teapot, usage=wgpu.BufferUsage.VERTEX
        )

        self._create_uniform_buffers()
        self._create_bind_groups()

    def _create_uniform_buffers(self):
        transform_dtype = np.dtype(
            [
                ("MVP", np.float32, (4, 4)),
                ("normal_matrix", np.float32, (4, 4)),
                ("M", np.float32, (4, 4)),
            ]
        )
        self.transform_uniforms = np.zeros((), dtype=transform_dtype)
        self.transform_buffer = self.device.create_buffer(
            size=self.transform_uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="transform_uniform_buffer",
        )

        material_dtype = np.dtype(
            {
                "names": ["albedo", "metallic", "roughness", "ao"],
                "formats": [(np.float32, 3), np.float32, np.float32, np.float32],
                "offsets": [0, 12, 16, 20],
                "itemsize": 32,
            }
        )
        self.material_uniforms = np.zeros((), dtype=material_dtype)
        self.material_buffer = self.device.create_buffer(
            size=self.material_uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="material_uniform_buffer",
        )
        self._update_material_buffer()

        light_dtype = np.dtype(
            {
                "names": ["lightPosition", "lightColour"],
                "formats": [(np.float32, 3), (np.float32, 3)],
                "offsets": [0, 16],
                "itemsize": 32,
            }
        )
        self.light_uniforms = np.zeros((), dtype=light_dtype)
        self.light_buffer = self.device.create_buffer(
            size=self.light_uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="light_uniform_buffer",
        )
        self._update_light_buffer()

        view_dtype = np.dtype(
            {
                "names": ["camPos", "exposure"],
                "formats": [(np.float32, 3), np.float32],
                "offsets": [0, 12],
                "itemsize": 16,
            }
        )
        self.view_uniforms = np.zeros((), dtype=view_dtype)
        self.view_uniforms["exposure"] = 2.2
        self.view_buffer = self.device.create_buffer(
            size=self.view_uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="view_uniform_buffer",
        )

    def _create_bind_groups(self):
        bind_group_layout_0 = self.pipeline.get_bind_group_layout(0)
        self.bind_group_0 = self.device.create_bind_group(
            layout=bind_group_layout_0,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.transform_buffer,
                        "offset": 0,
                        "size": self.transform_buffer.size,
                    },
                }
            ],
        )

        bind_group_layout_1 = self.pipeline.get_bind_group_layout(1)
        self.bind_group_1 = self.device.create_bind_group(
            layout=bind_group_layout_1,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.material_buffer,
                        "offset": 0,
                        "size": self.material_buffer.size,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self.light_buffer,
                        "offset": 0,
                        "size": self.light_buffer.size,
                    },
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": self.view_buffer,
                        "offset": 0,
                        "size": self.view_buffer.size,
                    },
                },
            ],
        )

    def _update_uniforms(self):
        # Apply rotation based on user input
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        # Update model position
        self.mouse_global_tx[3][0] = self.model_position.x
        self.mouse_global_tx[3][1] = self.model_position.y
        self.mouse_global_tx[3][2] = self.model_position.z
        # Update transform UBO
        model_view = (
            self.view @ self.model_transform.get_matrix() @ self.mouse_global_tx
        )
        self.transform_uniforms["M"] = model_view.to_numpy()
        self.transform_uniforms["MVP"] = (self.project @ model_view).to_numpy()
        normal_matrix = model_view.copy()
        normal_matrix.inverse().transpose()
        self.transform_uniforms["normal_matrix"] = normal_matrix.to_numpy()
        self.device.queue.write_buffer(
            self.transform_buffer, 0, self.transform_uniforms.tobytes()
        )

        # Update view UBO
        self.view_uniforms["camPos"] = self.eye.to_numpy()
        self.device.queue.write_buffer(
            self.view_buffer, 0, self.view_uniforms.tobytes()
        )

    def _update_material_buffer(self):
        self.material_uniforms["albedo"] = self._material["albedo"].to_numpy()
        self.material_uniforms["metallic"] = self._material["metallic"]
        self.material_uniforms["roughness"] = self._material["roughness"]
        self.material_uniforms["ao"] = self._material["ao"]
        self.device.queue.write_buffer(
            self.material_buffer, 0, self.material_uniforms.tobytes()
        )

    def _update_light_buffer(self):
        self.light_uniforms["lightPosition"] = self._light_position.to_numpy()
        self.light_uniforms["lightColour"] = self._light_colour.to_numpy()
        self.device.queue.write_buffer(
            self.light_buffer, 0, self.light_uniforms.tobytes()
        )

    def paintWebGPU(self):
        self._update_uniforms()
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
        width = int(self.width() * self.ratio)
        height = int(self.height() * self.ratio)
        render_pass.set_viewport(0, 0, width, height, 0, 1)
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group_0, [], 0, 999999)
        render_pass.set_bind_group(1, self.bind_group_1, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(self.teapot_size)
        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

    def resizeWebGPU(self, width, height):
        self.project = perspective(
            45.0, width / height if height > 0 else 1, 0.1, 100.0, PerspMode.WebGPU
        )
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_Space:
            # Reset camera rotation and position
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0, 0, 0)
            self.update()
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """
        Handles mouse movement events for camera control.

        Args:
            event: The QMouseEvent object containing the new mouse position.
        """
        # Rotate the scene if the left mouse button is pressed
        if self.rotate and event.buttons() == Qt.LeftButton:
            position = event.position()
            diff_x = position.x() - self.original_x_rotation
            diff_y = position.y() - self.original_y_rotation
            self.spin_x_face += int(0.5 * diff_y)
            self.spin_y_face += int(0.5 * diff_x)
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.update()
        # Translate (pan) the scene if the right mouse button is pressed
        elif self.translate and event.buttons() == Qt.RightButton:
            position = event.position()
            diff_x = int(position.x() - self.original_x_pos)
            diff_y = int(position.y() - self.original_y_pos)
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.model_position.x += self.INCREMENT * diff_x
            self.model_position.y -= self.INCREMENT * diff_y
            self.update()

    def mousePressEvent(self, event) -> None:
        """
        Handles mouse button press events to initiate rotation or translation.

        Args:
            event: The QMouseEvent object.
        """
        position = event.position()
        # Left button initiates rotation
        if event.button() == Qt.LeftButton:
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.rotate = True
        # Right button initiates translation
        elif event.button() == Qt.RightButton:
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.translate = True

    def mouseReleaseEvent(self, event) -> None:
        """
        Handles mouse button release events to stop rotation or translation.

        Args:
            event: The QMouseEvent object.
        """
        # Stop rotating when the left button is released
        if event.button() == Qt.LeftButton:
            self.rotate = False
        # Stop translating when the right button is released
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        """
        Handles mouse wheel events for zooming.

        Args:
            event: The QWheelEvent object.
        """
        num_pixels = event.angleDelta()
        # Zoom in or out by adjusting the Z position of the model
        if num_pixels.x() > 0:
            self.model_position.z += self.ZOOM
        elif num_pixels.x() < 0:
            self.model_position.z -= self.ZOOM
        self.update()
