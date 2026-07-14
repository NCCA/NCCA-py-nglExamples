"""
This module provides strongly typed enums for OpenGL texture parameters,
mirroring the functionality of the C++ TextureTypes.h header. This ensures
that correct OpenGL enum values are used when creating and configuring textures,
reducing the likelihood of runtime errors.
"""

from enum import Enum

import OpenGL.GL as gl


class GLTextureMinFilter(Enum):
    """
    Specifies the texture minification function.
    """

    NEAREST = gl.GL_NEAREST
    LINEAR = gl.GL_LINEAR
    NEAREST_MIPMAP_NEAREST = gl.GL_NEAREST_MIPMAP_NEAREST
    LINEAR_MIPMAP_NEAREST = gl.GL_LINEAR_MIPMAP_NEAREST
    LINEAR_MIPMAP_LINEAR = gl.GL_LINEAR_MIPMAP_LINEAR


class GLTextureMagFilter(Enum):
    """
    Specifies the texture magnification function.
    """

    NEAREST = gl.GL_NEAREST
    LINEAR = gl.GL_LINEAR


class GLTextureWrap(Enum):
    """
    Specifies the wrapping function for texture coordinates.
    """

    CLAMP_TO_EDGE = gl.GL_CLAMP_TO_EDGE
    CLAMP_TO_BORDER = gl.GL_CLAMP_TO_BORDER
    MIRRORED_REPEAT = gl.GL_MIRRORED_REPEAT
    REPEAT = gl.GL_REPEAT


class GLTextureInternalFormat(Enum):
    """
    Specifies the internal format of a texture.
    """

    DEPTH_COMPONENT = gl.GL_DEPTH_COMPONENT
    DEPTH_STENCIL = gl.GL_DEPTH_STENCIL
    RED = gl.GL_RED
    RG = gl.GL_RG
    RGB = gl.GL_RGB
    RGBA = gl.GL_RGBA
    R8 = gl.GL_R8
    R8_SNORM = gl.GL_R8_SNORM
    R16 = gl.GL_R16
    R16_SNORM = gl.GL_R16_SNORM
    RG8 = gl.GL_RG8
    RG8_SNORM = gl.GL_RG8_SNORM
    RG16 = gl.GL_RG16
    RG16_SNORM = gl.GL_RG16_SNORM
    R3_G3_B2 = gl.GL_R3_G3_B2
    RGB4 = gl.GL_RGB4
    RGB5 = gl.GL_RGB5
    RGB8 = gl.GL_RGB8
    RGB8_SNORM = gl.GL_RGB8_SNORM
    RGB10 = gl.GL_RGB10
    RGB12 = gl.GL_RGB12
    RGB16_SNORM = gl.GL_RGB16_SNORM
    RGBA2 = gl.GL_RGBA2
    RGBA4 = gl.GL_RGBA4
    RGB5_A1 = gl.GL_RGB5_A1
    RGBA8 = gl.GL_RGBA8
    RGBA8_SNORM = gl.GL_RGBA8_SNORM
    RGB10_A2 = gl.GL_RGB10_A2
    RGB10_A2UI = gl.GL_RGB10_A2UI
    RGBA12 = gl.GL_RGBA12
    RGBA16 = gl.GL_RGBA16
    SRGB8 = gl.GL_SRGB8
    SRGB8_ALPHA8 = gl.GL_SRGB8_ALPHA8
    R16F = gl.GL_R16F
    RG16F = gl.GL_RG16F
    RGB16F = gl.GL_RGB16F
    RGBA16F = gl.GL_RGBA16F
    R32F = gl.GL_R32F
    RG32F = gl.GL_RG32F
    RGB32F = gl.GL_RGB32F
    RGBA32F = gl.GL_RGBA32F
    R11F_G11F_B10F = gl.GL_R11F_G11F_B10F
    RGB9_E5 = gl.GL_RGB9_E5
    R8I = gl.GL_R8I
    R8UI = gl.GL_R8UI
    R16I = gl.GL_R16I
    R16UI = gl.GL_R16UI
    R32I = gl.GL_R32I
    R32UI = gl.GL_R32UI
    RG8I = gl.GL_RG8I
    RG8UI = gl.GL_RG8UI
    RG16I = gl.GL_RG16I
    RG16UI = gl.GL_RG16UI
    RG32I = gl.GL_RG32I
    RG32UI = gl.GL_RG32UI
    RGB8I = gl.GL_RGB8I
    RGB8UI = gl.GL_RGB8UI
    RGB16I = gl.GL_RGB16I
    RGB16UI = gl.GL_RGB16UI
    RGB32I = gl.GL_RGB32I
    RGB32UI = gl.GL_RGB32UI
    RGBA8I = gl.GL_RGBA8I
    RGBA8UI = gl.GL_RGBA8UI
    RGBA16I = gl.GL_RGBA16I
    RGBA16UI = gl.GL_RGBA16UI
    RGBA32I = gl.GL_RGBA32I
    RGBA32UI = gl.GL_RGBA32UI


class GLTextureFormat(Enum):
    """
    Specifies the format of the pixel data.
    """

    RED = gl.GL_RED
    RG = gl.GL_RG
    RGB = gl.GL_RGB
    BGR = gl.GL_BGR
    RGBA = gl.GL_RGBA
    BGRA = gl.GL_BGRA
    DEPTH_COMPONENT = gl.GL_DEPTH_COMPONENT
    DEPTH_STENCIL = gl.GL_DEPTH_STENCIL


class GLAttachment(Enum):
    """
    Specifies the attachment point for a texture.
    """

    _0 = gl.GL_COLOR_ATTACHMENT0
    _1 = gl.GL_COLOR_ATTACHMENT1
    _2 = gl.GL_COLOR_ATTACHMENT2
    _3 = gl.GL_COLOR_ATTACHMENT3
    _4 = gl.GL_COLOR_ATTACHMENT4
    _5 = gl.GL_COLOR_ATTACHMENT5
    _6 = gl.GL_COLOR_ATTACHMENT6
    _7 = gl.GL_COLOR_ATTACHMENT7
    _8 = gl.GL_COLOR_ATTACHMENT8


class GLTextureDataType(Enum):
    """
    Specifies the data type of the pixel data.
    """

    UNSIGNED_BYTE = gl.GL_UNSIGNED_BYTE
    BYTE = gl.GL_BYTE
    UNSIGNED_SHORT = gl.GL_UNSIGNED_SHORT
    SHORT = gl.GL_SHORT
    UNSIGNED_INT = gl.GL_UNSIGNED_INT
    INT = gl.GL_INT
    FLOAT = gl.GL_FLOAT
    UNSIGNED_BYTE_3_3_2 = gl.GL_UNSIGNED_BYTE_3_3_2
    UNSIGNED_BYTE_2_3_3_REV = gl.GL_UNSIGNED_BYTE_2_3_3_REV
    UNSIGNED_SHORT_5_6_5 = gl.GL_UNSIGNED_SHORT_5_6_5
    UNSIGNED_SHORT_5_6_5_REV = gl.GL_UNSIGNED_SHORT_5_6_5_REV
    UNSIGNED_SHORT_4_4_4_4 = gl.GL_UNSIGNED_SHORT_4_4_4_4
    UNSIGNED_SHORT_4_4_4_4_REV = gl.GL_UNSIGNED_SHORT_4_4_4_4_REV
    UNSIGNED_SHORT_5_5_5_1 = gl.GL_UNSIGNED_SHORT_5_5_5_1
    UNSIGNED_SHORT_1_5_5_5_REV = gl.GL_UNSIGNED_SHORT_1_5_5_5_REV
    UNSIGNED_INT_8_8_8_8 = gl.GL_UNSIGNED_INT_8_8_8_8
    UNSIGNED_INT_8_8_8_8_REV = gl.GL_UNSIGNED_INT_8_8_8_8_REV
    UNSIGNED_INT_10_10_10_2 = gl.GL_UNSIGNED_INT_10_10_10_2
    UNSIGNED_INT_2_10_10_10_REV = gl.GL_UNSIGNED_INT_2_10_10_10_REV


class GLTextureDepthFormats(Enum):
    """
    Specifies the format of the depth component.
    """

    DEPTH_COMPONENT16 = gl.GL_DEPTH_COMPONENT16
    DEPTH_COMPONENT24 = gl.GL_DEPTH_COMPONENT24
    DEPTH_COMPONENT32 = gl.GL_DEPTH_COMPONENT32
    DEPTH_COMPONENT32F = gl.GL_DEPTH_COMPONENT32F


class GLTextureDepthStencilFormats(Enum):
    """
    Specifies the format of the depth and stencil components.
    """

    DEPTH24_STENCIL8 = gl.GL_DEPTH24_STENCIL8
    DEPTH32F_STENCIL8 = gl.GL_DEPTH32F_STENCIL8


class GLTextureStencilFormats(Enum):
    """
    Specifies the format of the stencil component.
    """

    STENCIL_INDEX8 = gl.GL_STENCIL_INDEX8
