#!/usr/bin/env -S uv run --script
"""WebGPU version of the image maze demo."""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import wgpu
from maze_scene import (
    ActorState,
    Direction,
    Maze,
    actor_forward,
    actor_world_position,
    move_actor,
    top_view,
)
from mesh_data import (
    build_coloured_mesh,
    build_wall_mesh,
    build_wireframe_wall_mesh,
    ground_mesh,
)
from ncca.ngl import (
    Mat4,
    PerspMode,
    PrimData,
    Prims,
    Vec3,
    logger,
    look_at,
    perspective,
)
from ncca.ngl.webgpu import WebGPUWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

MAPS_DIR = Path(__file__).parent / "maps"


class WebGPUScene(WebGPUWidget):
    def __init__(self, map_path: str | Path) -> None:
        super().__init__()
        self.setWindowTitle("Image Maze (WebGPU)")
        self.msaa_sample_count = 4

        self.maze = Maze.from_file(map_path)
        self.walls = self.maze.wall_cells()
        self.actor = ActorState(2, 2)
        self.active_camera = 1
        self.wireframe = False

        self.mouse_global_tx = Mat4()
        self.model_position = Vec3()
        self.rotate = False
        self.translate = False
        self.spin_x_face = 0
        self.spin_y_face = 0
        self.original_x_rotation = 0.0
        self.original_y_rotation = 0.0
        self.original_x_pos = 0.0
        self.original_y_pos = 0.0
        self.translation_sensitivity = 0.01
        self.zoom_sensitivity = 0.1

        self.top_view = top_view()
        self.actor_view = Mat4()
        self._update_actor_camera()
        self.project = perspective(
            45.0,
            self.width() / max(self.height(), 1),
            0.5,
            50.0,
            PerspMode.WebGPU,
        )

        self.device = get_default_device()
        self._create_pipelines()
        self._create_scene()
        self._create_render_buffer()

    def _create_pipelines(self) -> None:
        shader_source = (Path(__file__).parent / "ImageMazeShader.wgsl").read_text()
        shader = self.device.create_shader_module(code=shader_source)
        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )
        layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout]
        )

        def make_pipeline(topology):
            return self.device.create_render_pipeline(
                layout=layout,
                vertex={
                    "module": shader,
                    "entry_point": "vertex_main",
                    "buffers": [
                        {
                            "array_stride": 7 * 4,
                            "step_mode": "vertex",
                            "attributes": [
                                {
                                    "format": "float32x3",
                                    "offset": 0,
                                    "shader_location": 0,
                                },
                                {
                                    "format": "float32x4",
                                    "offset": 12,
                                    "shader_location": 1,
                                },
                            ],
                        }
                    ],
                },
                fragment={
                    "module": shader,
                    "entry_point": "fragment_main",
                    "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
                },
                primitive={"topology": topology},
                depth_stencil={
                    "format": wgpu.TextureFormat.depth24plus,
                    "depth_write_enabled": True,
                    "depth_compare": wgpu.CompareFunction.less,
                },
                multisample={"count": self.msaa_sample_count},
            )

        self.triangle_pipeline = make_pipeline(wgpu.PrimitiveTopology.triangle_list)
        self.line_pipeline = make_pipeline(wgpu.PrimitiveTopology.line_list)

    def _make_object(self, data: np.ndarray, label: str) -> dict:
        vertices = np.ascontiguousarray(data, dtype=np.float32).reshape(-1, 7)
        vertex_buffer = self.device.create_buffer_with_data(
            data=vertices, usage=wgpu.BufferUsage.VERTEX, label=f"{label}_vertices"
        )
        uniform_buffer = self.device.create_buffer(
            size=64,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label=f"{label}_uniforms",
        )
        bind_group = self.device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": uniform_buffer,
                        "offset": 0,
                        "size": uniform_buffer.size,
                    },
                }
            ],
        )
        return {
            "vertex_buffer": vertex_buffer,
            "uniform_buffer": uniform_buffer,
            "bind_group": bind_group,
            "count": len(vertices),
        }

    def _create_scene(self) -> None:
        cube = PrimData.primitive(Prims.CUBE.value)
        troll = PrimData.primitive(Prims.TROLL.value)
        self.wall_object = self._make_object(
            build_wall_mesh(cube, self.walls), "maze_walls"
        )
        self.wireframe_wall_object = self._make_object(
            build_wireframe_wall_mesh(self.walls), "maze_wireframe_walls"
        )
        self.actor_object = self._make_object(
            build_coloured_mesh(troll, (1.0, 0.0, 0.0, 1.0)), "maze_actor"
        )
        self.ground_object = self._make_object(ground_mesh(40.0, -0.55), "maze_ground")

    def _update_global_transform(self) -> None:
        rot_x = Mat4().rotate_x(self.spin_x_face)
        rot_y = Mat4().rotate_y(self.spin_y_face)
        self.mouse_global_tx = rot_y @ rot_x
        self.mouse_global_tx[3, 0] = self.model_position.x
        self.mouse_global_tx[3, 1] = self.model_position.y
        self.mouse_global_tx[3, 2] = self.model_position.z

    def _update_actor_camera(self) -> None:
        x, y, z = actor_world_position(self.maze, self.actor)
        dx, dy, dz = actor_forward(self.actor)
        self.actor_view = look_at(
            Vec3(x, y, z),
            Vec3(x + dx, y + dy, z + dz),
            Vec3(0.0, 1.0, 0.0),
        )

    def _actor_model(self) -> Mat4:
        x, y, z = actor_world_position(self.maze, self.actor)
        model = Mat4().rotate_y(self.actor.rotation)
        model[3, 0] = x
        model[3, 1] = y
        model[3, 2] = z
        return model

    def _upload_mvp(self, obj: dict, mvp: Mat4) -> None:
        self.device.queue.write_buffer(
            obj["uniform_buffer"], 0, mvp.to_numpy().astype(np.float32).tobytes()
        )

    @staticmethod
    def _draw_object(render_pass, obj: dict) -> None:
        render_pass.set_bind_group(0, obj["bind_group"], [], 0, 999999)
        render_pass.set_vertex_buffer(0, obj["vertex_buffer"])
        render_pass.draw(obj["count"])

    def paintWebGPU(self) -> None:
        self._update_global_transform()
        view = self.top_view if self.active_camera == 1 else self.actor_view
        scene_mvp = self.project @ view @ self.mouse_global_tx
        wall_object = self.wireframe_wall_object if self.wireframe else self.wall_object
        self._upload_mvp(wall_object, scene_mvp)
        self._upload_mvp(
            self.actor_object,
            scene_mvp @ self._actor_model(),
        )
        self._upload_mvp(self.ground_object, scene_mvp)

        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.multisample_texture_view,
                    "resolve_target": self.colour_buffer_texture_view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (1.0, 1.0, 1.0, 1.0),
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_buffer_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )
        render_pass.set_pipeline(
            self.line_pipeline if self.wireframe else self.triangle_pipeline
        )
        self._draw_object(render_pass, wall_object)
        render_pass.set_pipeline(self.triangle_pipeline)
        self._draw_object(render_pass, self.actor_object)
        self._draw_object(render_pass, self.ground_object)
        render_pass.end()
        self.device.queue.submit([encoder.finish()])
        self._update_colour_buffer()

    def resizeWebGPU(self, width: int, height: int) -> None:
        self.project = perspective(
            45.0, width / max(height, 1), 0.5, 50.0, PerspMode.WebGPU
        )
        self.update()

    def keyPressEvent(self, event) -> None:
        directions = {
            Qt.Key_Up: Direction.NORTH,
            Qt.Key_Down: Direction.SOUTH,
            Qt.Key_Left: Direction.WEST,
            Qt.Key_Right: Direction.EAST,
        }
        key = event.key()
        if key in directions:
            self.actor = move_actor(self.maze, self.actor, directions[key])
            self._update_actor_camera()
        elif key == Qt.Key_1:
            self.active_camera = 1
        elif key == Qt.Key_2:
            self.active_camera = 2
        elif key == Qt.Key_W:
            self.wireframe = not self.wireframe
        elif key == Qt.Key_Space:
            self.spin_x_face = 0
            self.spin_y_face = 0
            self.model_position.set(0.0, 0.0, 0.0)
        elif key == Qt.Key_Escape:
            self.close()
        self.update()

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
        position = event.position()
        if self.rotate and event.buttons() == Qt.LeftButton:
            self.spin_x_face += int(0.5 * (position.y() - self.original_y_rotation))
            self.spin_y_face += int(0.5 * (position.x() - self.original_x_rotation))
            self.original_x_rotation = position.x()
            self.original_y_rotation = position.y()
            self.update()
        elif self.translate and event.buttons() == Qt.RightButton:
            self.model_position.x += self.translation_sensitivity * (
                position.x() - self.original_x_pos
            )
            self.model_position.y -= self.translation_sensitivity * (
                position.y() - self.original_y_pos
            )
            self.original_x_pos = position.x()
            self.original_y_pos = position.y()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.rotate = False
        elif event.button() == Qt.RightButton:
            self.translate = False

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta > 0:
            self.model_position.z += self.zoom_sensitivity
        elif delta < 0:
            self.model_position.z -= self.zoom_sensitivity
        self.update()


class DebugApplication(QApplication):
    def __init__(self, argv) -> None:
        super().__init__(argv)
        logger.info("Running in full debug mode")

    def notify(self, receiver, event):
        try:
            return super().notify(receiver, event)
        except Exception:
            traceback.print_exc()
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map",
        type=Path,
        default=MAPS_DIR / "small.png",
        help="maze image (default: maps/small.png)",
    )
    parser.add_argument(
        "--smoketest", nargs="?", const=200, default=None, type=int, metavar="MS"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = DebugApplication(sys.argv) if args.debug else QApplication(sys.argv)
    window = WebGPUScene(args.map)
    window.resize(1024, 720)
    window.show()

    if args.smoketest is not None:
        QTimer.singleShot(args.smoketest, lambda: (print("SMOKETEST OK"), app.quit()))

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
