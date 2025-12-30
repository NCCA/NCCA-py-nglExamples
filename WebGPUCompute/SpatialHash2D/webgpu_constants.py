from enum import IntEnum

import numpy as np
from ncca.ngl import Mat2, Mat3, Mat4, Vec2, Vec3, Vec4

FLOAT_SIZE = np.dtype(np.float32).itemsize


class NGLToWebGPU:
    _strides = {
        Vec2: 2 * FLOAT_SIZE,
        Vec3: 3 * FLOAT_SIZE,
        Vec4: 4 * FLOAT_SIZE,
        Mat2: 4 * FLOAT_SIZE,
        Mat3: 12 * FLOAT_SIZE,
        Mat4: 16 * FLOAT_SIZE,
    }
    _vertex_format = {
        Vec2: "float32x2",
        Vec3: "float32x3",
        Vec4: "float32x4",
    }

    @staticmethod
    def stride_from_type(type):
        return NGLToWebGPU._strides[type]

    @staticmethod
    def vertex_format(type):
        return NGLToWebGPU._vertex_format[type]
