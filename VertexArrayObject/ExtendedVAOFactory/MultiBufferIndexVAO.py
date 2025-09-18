import ctypes

import numpy as np
import OpenGL.GL as gl
from ngl import AbstractVAO, VertexData, logger


class MultiBufferIndexVAO(AbstractVAO):
    def __init__(self, mode=gl.GL_TRIANGLES):
        super().__init__(mode)
        self.m_vbo_ids = []
        self.m_index_buffer_id = 0
        self.m_index_type = gl.GL_UNSIGNED_SHORT
        self.m_indicesCount = 0

    def draw(self, start_index=0, amount=-1):
        if self.m_bound and self.m_allocated:
            if amount == -1:
                count = self.m_indicesCount - start_index
            else:
                count = amount

            if count <= 0:
                return
            if self.m_index_type == gl.GL_UNSIGNED_INT:
                offset = start_index * 4
            elif self.m_index_type == gl.GL_UNSIGNED_SHORT:
                offset = start_index * 2
            elif self.m_index_type == gl.GL_UNSIGNED_BYTE:
                offset = start_index * 1
            else:
                logger.error("Unsupported index type")
                return
            gl.glDrawElements(
                self.m_mode, count, self.m_index_type, ctypes.c_void_p(offset)
            )
        else:
            logger.error("MultiBufferIndexVAO is not bound or not allocated")

    def set_data(self, data, index=None):
        if not isinstance(data, VertexData):
            logger.error(
                "MultiBufferIndexVAO.set_data: data must be of type VertexData"
            )
            raise TypeError("data must be of type VertexData")

        if not self.m_bound:
            logger.error("Trying to set VOA data when unbound")
            return

        if index is None:
            index = len(self.m_vbo_ids)

        if index >= len(self.m_vbo_ids):
            new_buffers = index - len(self.m_vbo_ids) + 1
            new_ids = gl.glGenBuffers(new_buffers)
            if isinstance(new_ids, np.ndarray):
                self.m_vbo_ids.extend(new_ids)
            else:
                self.m_vbo_ids.append(new_ids)

        if isinstance(data.data, list):
            vertex_array = np.array(data.data, dtype=np.float32)
        else:
            vertex_array = data.data

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.m_vbo_ids[index])
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, vertex_array.nbytes, vertex_array, data.mode
        )
        self.m_allocated = True

    def set_indices(
        self, data, index_type=gl.GL_UNSIGNED_SHORT, mode=gl.GL_STATIC_DRAW
    ):
        if not isinstance(data, list):
            logger.error("MultiBufferIndexVAO.set_indices: data must be of type List")
            raise TypeError("data must be of type List")

        if not self.m_bound:
            logger.error("Trying to set VOA data when unbound")
            return

        if self.m_index_buffer_id == 0:
            self.m_index_buffer_id = gl.glGenBuffers(1)

        self.m_index_type = index_type
        self.m_indicesCount = len(data)
        if index_type == gl.GL_UNSIGNED_INT:
            index_array = np.array(data, dtype=np.uint32)
        elif index_type == gl.GL_UNSIGNED_SHORT:
            index_array = np.array(data, dtype=np.uint16)
        elif index_type == gl.GL_UNSIGNED_BYTE:
            index_array = np.array(data, dtype=np.uint8)
        else:
            logger.error("Unsupported index type")
            return
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.m_index_buffer_id)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER, index_array.nbytes, index_array, mode
        )

    def remove_vao(self):
        if self.m_bound:
            self.unbind()
        if self.m_allocated:
            if self.m_vbo_ids:
                gl.glDeleteBuffers(len(self.m_vbo_ids), self.m_vbo_ids)
            if self.m_index_buffer_id != 0:
                gl.glDeleteBuffers(1, [self.m_index_buffer_id])

        gl.glDeleteVertexArrays(1, [self.m_id])
        self.m_allocated = False

    def get_buffer_id(self, index=0):
        if index < len(self.m_vbo_ids):
            return self.m_vbo_ids[index]
        return 0

    def map_buffer(self, index=0, access_mode=gl.GL_READ_WRITE):
        if index < len(self.m_vbo_ids):
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.m_vbo_ids[index])
            return gl.glMapBuffer(gl.GL_ARRAY_BUFFER, access_mode)
        return None
