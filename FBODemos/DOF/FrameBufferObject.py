"""
A Python implementation of the NGL FrameBufferObject class, designed to
encapsulate an OpenGL FrameBufferObject and its associated textures.
"""

import dataclasses
from enum import Enum

import OpenGL.GL as gl
from ncca.ngl import Vec2, logger
from TextureTypes import (
    GLAttachment,
    GLTextureDataType,
    GLTextureDepthFormats,
    GLTextureFormat,
    GLTextureInternalFormat,
    GLTextureMagFilter,
    GLTextureMinFilter,
    GLTextureWrap,
)


@dataclasses.dataclass
class TextureAttachment:
    """
    A simple data class to store texture attachment information.
    """

    id: int = 0
    name: str = ""
    samplerLocation: int = 0


class FrameBufferObject:
    """
    The FrameBufferObject class encapsulates an OpenGL FrameBufferObject and the
    associated textures. It allows for creating and managing color and depth
    attachments.
    """

    s_default_fbo: int = 0
    s_copy_fbo: int = 0

    class Target(Enum):
        """
        Specifies the framebuffer target.
        """

        FRAMEBUFFER = gl.GL_FRAMEBUFFER
        DRAW = gl.GL_DRAW_FRAMEBUFFER
        READ = gl.GL_READ_FRAMEBUFFER

    def __init__(self, width: int, height: int, num_attachments: int = 8):
        self._id = gl.glGenFramebuffers(1)
        self._width = int(width)
        self._height = int(height)
        self._attachments = [TextureAttachment() for _ in range(num_attachments)]
        self._depth_buffer_id = 0
        self._bound = False

    @classmethod
    def create(
        cls, width: int, height: int, num_attachments: int = 8
    ) -> "FrameBufferObject":
        """
        Factory method to create a new FrameBufferObject instance.

        Parameters
        ----------
            width : int
                The width of the framebuffer.
            height : int
                The height of the framebuffer.
            num_attachments : int
                The number of color attachments to support.

        Returns
        -------
            FrameBufferObject
                A new instance of FrameBufferObject.
        """
        return cls(width, height, num_attachments)

    def __del__(self):
        """
        Destructor to clean up OpenGL resources.
        """
        gl.glDeleteFramebuffers(1, self._id)
        for t in self._attachments:
            if t.id != 0:
                gl.glDeleteTextures(1, t.id)
        if self._depth_buffer_id != 0:
            gl.glDeleteTextures(1, self._depth_buffer_id)

    def add_depth_buffer(
        self,
        format: GLTextureDepthFormats,
        min_filter: GLTextureMinFilter,
        mag_filter: GLTextureMagFilter,
        swrap: GLTextureWrap,
        twrap: GLTextureWrap,
        immutable: bool = False,
    ) -> bool:
        """
        Adds a depth buffer to the FBO using a texture attachment.

        Parameters
        ----------
            format : GLTextureDepthFormats
                The format for the depth component.
            min_filter : GLTextureMinFilter
                The minification filter mode.
            mag_filter : GLTextureMagFilter
                The magnification filter mode.
            swrap : GLTextureWrap
                The texture wrap mode in s.
            twrap : GLTextureWrap
                The texture wrap mode in t.
            immutable : bool
                Whether to use immutable storage (glTexStorage2D).

        Returns
        -------
            bool
                True if successful, False otherwise.
        """
        if not self._bound:
            logger.error("Trying to add depthbuffer to unbound Framebuffer")
            return False
        self._depth_buffer_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._depth_buffer_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, mag_filter.value)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, min_filter.value)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, swrap.value)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, twrap.value)
        if immutable:
            gl.glTexStorage2D(
                gl.GL_TEXTURE_2D, 1, format.value, self._width, self._height
            )
        else:
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                format.value,
                self._width,
                self._height,
                0,
                gl.GL_DEPTH_COMPONENT,
                gl.GL_FLOAT,
                None,
            )
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_DEPTH_ATTACHMENT,
            gl.GL_TEXTURE_2D,
            self._depth_buffer_id,
            0,
        )
        return True

    def add_colour_attachment(
        self,
        name: str,
        attachment: GLAttachment,
        format: GLTextureFormat,
        iformat: GLTextureInternalFormat,
        type: GLTextureDataType,
        min_filter: GLTextureMinFilter,
        mag_filter: GLTextureMagFilter,
        swrap: GLTextureWrap,
        twrap: GLTextureWrap,
        immutable: bool = False,
    ) -> bool:
        """
        Adds a color attachment to the FBO.

        Parameters
        ----------
            name : str
                The name of the attachment.
            attachment : GLAttachment
                The attachment point.
            format : GLTextureFormat
                The format of the texture.
            iformat : GLTextureInternalFormat
                The internal format of the texture.
            type : GLTextureDataType
                The data type of the texture.
            min_filter : GLTextureMinFilter
                The minification filter.
            mag_filter : GLTextureMagFilter
                The magnification filter.
            swrap : GLTextureWrap
                The texture wrap mode in s.
            twrap : GLTextureWrap
                The texture wrap mode in t.
            immutable : bool
                Whether to use immutable storage.

        Returns
        -------
            bool
                True if successful, False otherwise.
        """
        if not self._bound:
            gl.NGLMessage.addError("Trying to add attachment to unbound Framebuffer")
            return False

        tex_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, mag_filter.value)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, min_filter.value)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, swrap.value)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, twrap.value)

        if immutable:
            gl.glTexStorage2D(
                gl.GL_TEXTURE_2D, 1, iformat.value, self._width, self._height
            )
        else:
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                iformat.value,
                self._width,
                self._height,
                0,
                format.value,
                type.value,
                None,
            )

        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, attachment.value, gl.GL_TEXTURE_2D, tex_id, 0
        )

        t = TextureAttachment(id=tex_id, name=name)
        index = attachment.value - gl.GL_COLOR_ATTACHMENT0
        self._attachments[index] = t
        return True

    def bind(self, target: Target = Target.FRAMEBUFFER) -> None:
        """
        Binds this FBO as the active one.
        """
        gl.glBindFramebuffer(target.value, self._id)
        self._bound = True

    def unbind(self) -> None:
        """
        Unbinds the FBO, binding the default framebuffer instead.
        """
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.s_default_fbo)
        self._bound = False

    def __enter__(self):
        self.bind()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unbind()

    def get_texture_id(self, name: str) -> int:
        """
        Gets the texture ID of a bound attachment by name.

        Parameters
        ----------
            name : str
                The name of the texture attachment.

        Returns
        -------
            int
                The texture ID if found, otherwise 0.
        """
        for attachment in self._attachments:
            if attachment.name == name:
                return attachment.id
        return 0

    def is_complete(self, target: Target = Target.FRAMEBUFFER) -> bool:
        """
        Checks if the framebuffer is complete.

        Parameters
        ----------
            target : Target
                The framebuffer target to check.

        Returns
        -------
            bool
                True if the framebuffer is complete, False otherwise.
        """
        result = gl.glCheckFramebufferStatus(target.value)
        return result == gl.GL_FRAMEBUFFER_COMPLETE

    def get_id(self) -> int:
        """Returns the underlying ID of the FBO."""
        return self._id

    @property
    def depth_texture_id(self) -> int:
        """The texture ID of the depth buffer."""
        return self._depth_buffer_id

    def set_viewport(self) -> None:
        """Calls glViewport using the FBO dimensions."""
        gl.glViewport(0, 0, self._width, self._height)

    @property
    def size(self) -> Vec2:
        """The size of the FBO as a Vec2."""
        return Vec2(self._width, self._height)

    @property
    def width(self) -> int:
        """The width of the FBO."""
        return self._width

    @property
    def id(self) -> int:
        """The id of the FBO."""
        return self._id

    @property
    def height(self) -> int:
        """The height of the FBO."""
        return self._height

    @staticmethod
    def set_default_fbo(fbo_id: int) -> None:
        """
        Sets the default FBO ID to bind to on unbind.
        """
        FrameBufferObject.s_default_fbo = fbo_id

    @staticmethod
    def copy_frame_buffer_texture(
        src_id: int, dst_id: int, width: int, height: int, mode=gl.GL_COLOR_BUFFER_BIT
    ) -> None:
        """
        Copies texture data from one texture to another using an FBO.
        """
        if FrameBufferObject.s_copy_fbo == 0:
            FrameBufferObject.s_copy_fbo = gl.glGenFramebuffers(1)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, FrameBufferObject.s_copy_fbo)
        gl.glFramebufferTexture2D(
            gl.GL_READ_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, src_id, 0
        )
        gl.glFramebufferTexture2D(
            gl.GL_DRAW_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT1, gl.GL_TEXTURE_2D, dst_id, 0
        )
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT1)
        gl.glBlitFramebuffer(
            0, 0, width, height, 0, 0, width, height, mode, gl.GL_NEAREST
        )
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, FrameBufferObject.s_default_fbo)

    def print(self) -> None:
        """
        Prints debug information about the FBO and its attachments, mimicking the C++ version.
        """
        if not self._bound:
            logger.error("Trying to print unbound FrameBufferObject\n")
            return

        max_attachments = gl.glGetIntegerv(gl.GL_MAX_COLOR_ATTACHMENTS)
        logger.info(f"Max Color Attachments: {max_attachments}\n")

        # The C++ version binds the framebuffer here to query it.
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._id)

        i = 0
        while i < max_attachments:
            # This queries which color attachment is bound to the ith draw buffer
            draw_buffer = gl.glGetIntegerv(gl.GL_DRAW_BUFFER0 + i)

            if draw_buffer != gl.GL_NONE:
                attachment_name = (
                    self._attachments[i].name if i < len(self._attachments) else "N/A"
                )

                logger.info(
                    f"{attachment_name} Shader Output Location {i} - color attachment {draw_buffer}"
                )

                # Query attachment parameters using the attachment point
                obj_type = gl.glGetFramebufferAttachmentParameteriv(
                    gl.GL_FRAMEBUFFER,
                    draw_buffer,
                    gl.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE,
                )
                type_str = "Texture" if obj_type == gl.GL_TEXTURE else "Render Buffer"
                logger.info(f"\tAttachment Type : {type_str} ")

                obj_name = gl.glGetFramebufferAttachmentParameteriv(
                    gl.GL_FRAMEBUFFER,
                    draw_buffer,
                    gl.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME,
                )
                logger.info(f"\tAttachment object name :  {obj_name} ")

            i += 1
