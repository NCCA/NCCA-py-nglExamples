from enum import IntEnum

import numpy as np
from ncca.ngl import Mat2, Mat3, Mat4, Vec2, Vec3, Vec4

FLOAT_SIZE = np.dtype(np.float32).itemsize


class NGLToWebGPU:
    _strides = {
        "vec2": 2 * FLOAT_SIZE,
        "vec3": 3 * FLOAT_SIZE,
        "vec4": 4 * FLOAT_SIZE,
        "mat2": 4 * FLOAT_SIZE,
        "mat3": 12 * FLOAT_SIZE,
        "mat4": 16 * FLOAT_SIZE,
    }
    _vertex_format = {
        "vec2": "float32x2",
        "vec3": "float32x3",
        "vec4": "float32x4",
    }

    @staticmethod
    def stride_from_type(type: str):
        return NGLToWebGPU._strides[type.lower()]

    @staticmethod
    def vertex_format(type: str):
        return NGLToWebGPU._vertex_format[type.lower()]
