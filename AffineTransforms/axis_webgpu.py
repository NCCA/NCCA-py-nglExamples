"""RGB axis gizmo for the WebGPU AffineTransforms demo.

The OpenGL sibling (axis.py) draws the gizmo as nine calls to
Primitives.draw(), one per shaft/head, each with its own MVP. WebGPU has no
equivalent of that per-draw uniform push without either a bind group per
part or dynamic offsets, so here the nine transforms are baked into the
vertex data at build time and the whole gizmo goes down as a single draw of
position/colour vertices. Unlit, like the DefaultShader.COLOUR version.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wgpu
from ncca.ngl import PrimData, Transform

# Proportions, as fractions of the axis half-length -- kept in step with
# axis.py so both versions of the demo draw the same gizmo.
_SHAFT_RADIUS = 0.02
_HEAD_RADIUS = 0.07
_HEAD_LENGTH = 0.2

# Rotations take the primitive's own axis onto the drawn one: PrimData's
# cylinder points down +y and its cone down +z.
#   colour, shaft rotation, +head rotation, -head rotation, direction
_AXES = (
    ((1.0, 0.0, 0.0), (0, 0, -90), (0, 90, 0), (0, -90, 0), (1, 0, 0)),
    ((0.0, 1.0, 0.0), (0, 0, 0), (-90, 0, 0), (90, 0, 0), (0, 1, 0)),
    ((0.0, 0.0, 1.0), (90, 0, 0), (0, 0, 0), (0, 180, 0), (0, 0, 1)),
)


def _placed(
    data: np.ndarray,
    scale: tuple[float, float, float],
    rotation: tuple[float, float, float],
    position: tuple[float, float, float],
    colour: tuple[float, float, float],
) -> np.ndarray:
    """
    Transform one primitive's vertices into place and tag them with a colour.

    Parameters
    ----------
        data : np.ndarray
            flat PrimData output, 8 floats per vertex (pos, normal, uv)
        scale : tuple
            x, y, z scale applied before the rotation
        rotation : tuple
            x, y, z rotation in degrees
        position : tuple
            x, y, z translation applied last
        colour : tuple
            r, g, b for every vertex of this part

    Returns
    -------
        np.ndarray
            (n, 6) float32 of position and colour
    """
    tx = Transform()
    tx.set_scale(*scale)
    tx.set_rotation(*rotation)
    tx.set_position(*position)
    matrix = tx.matrix().to_numpy()

    verts = np.asarray(data, dtype=np.float32).reshape(-1, 8)
    points = np.hstack([verts[:, 0:3], np.ones((len(verts), 1), dtype=np.float32)])
    out = np.empty((len(verts), 6), dtype=np.float32)
    out[:, 0:3] = (points @ matrix)[:, 0:3]
    out[:, 3:6] = colour
    return out


def build_geometry(scale: float = 1.5) -> np.ndarray:
    """
    Build the whole gizmo as one interleaved position/colour array.

    Parameters
    ----------
        scale : float
            half-length of each axis, i.e. the arrow tips sit at +/- scale

    Returns
    -------
        np.ndarray
            flat float32 vertex data, 6 floats per vertex
    """
    shaft_data = PrimData.cylinder(1.0, 1.0, 20, 1)
    head_data = PrimData.cone(1.0, 1.0, 20, 1)

    r = _SHAFT_RADIUS * scale
    head_len = _HEAD_LENGTH * scale
    head_r = _HEAD_RADIUS * scale
    # The shaft stops where the heads start so the tips land exactly on
    # +/- scale; the cylinder is centred, hence the full span here.
    shaft_size = (r, 2.0 * (scale - head_len), r)
    head_size = (head_r, head_r, head_len)
    tip = scale - head_len

    parts = []
    for colour, shaft_rot, pos_rot, neg_rot, direction in _AXES:
        offset = tuple(d * tip for d in direction)
        parts.append(_placed(shaft_data, shaft_size, shaft_rot, (0, 0, 0), colour))
        parts.append(_placed(head_data, head_size, pos_rot, offset, colour))
        parts.append(
            _placed(head_data, head_size, neg_rot, tuple(-o for o in offset), colour)
        )
    return np.vstack(parts).reshape(-1)


class AxisPipeline:
    """
    A red/green/blue X/Y/Z axis gizmo drawn at the origin.

    Each axis is a thin shaft through the origin with an arrow head at both
    ends, so the negative half of each axis is as visible as the positive
    one. The gizmo sits at the origin of the scene, not on the object, so
    feed update_uniforms() the view/projection without the object's model
    matrix.

    Attributes
    ----------
        vertex_count : int
            number of vertices in the baked gizmo
    """

    def __init__(self, device, sample_count: int = 4, scale: float = 1.5) -> None:
        self.device = device
        data = build_geometry(scale)
        self.vertex_count = data.size // 6
        self.vertex_buffer = device.create_buffer_with_data(
            data=data, usage=wgpu.BufferUsage.VERTEX
        )

        shader_src = (Path(__file__).parent / "AxisShader.wgsl").read_text()
        shader_module = device.create_shader_module(code=shader_src)
        self.bind_group_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )
        self.pipeline = device.create_render_pipeline(
            layout=device.create_pipeline_layout(
                bind_group_layouts=[self.bind_group_layout]
            ),
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 6 * 4,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x3", "offset": 12, "shader_location": 1},
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
            multisample={"count": sample_count},
        )

        self.uniforms = np.zeros((), dtype=np.dtype([("mvp", np.float32, (4, 4))]))
        self.uniform_buffer = device.create_buffer(
            size=self.uniforms.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.bind_group = device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.uniform_buffer,
                        "offset": 0,
                        "size": self.uniform_buffer.size,
                    },
                }
            ],
        )

    def update_uniforms(self, mvp) -> None:
        """Upload the gizmo's MVP; call before beginning the render pass."""
        self.uniforms["mvp"] = mvp.to_numpy()
        self.device.queue.write_buffer(self.uniform_buffer, 0, self.uniforms.tobytes())

    def render(self, render_pass) -> None:
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(self.vertex_count)
