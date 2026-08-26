import numpy as np
import wgpu


class MeshData:
    def __init__(self, device):
        self.device = device
        # 1. Temporary storage for mesh data before processing
        self._raw_meshes = {}

        # 2. Final, consolidated data (host and device)
        self.vertex_data = None
        self.storage_data = None

        self.vertex_buffer = None
        self.storage_buffer = None

        # 3. Metadata for rendering
        self._mesh_info = {}
        self._instance_count = 0

    def add_mesh(self, name: str, prim_data):
        """
        Temporarily stores raw mesh data. Buffers are not created here.
        """
        if name in self._raw_meshes:
            print(
                f"Warning: Mesh with name '{name}' already exists and will be overwritten."
            )
        self._raw_meshes[name] = prim_data
        self._instance_count = len(self._raw_meshes)

    def create_buffers(self):
        """
        Consolidates all raw mesh data into large NumPy arrays and creates
        the corresponding GPU buffers. This is the 'pre-allocation' step for non-indexed meshes.
        """
        if not self._raw_meshes:
            return

        # --- Define dtypes ---
        vertex_dtype = np.dtype(
            [
                ("position", "float32", (3,)),
                ("normal", "float32", (3,)),
                ("texcoord", "float32", (2,)),
            ]
        )

        # --- Pass 1: Calculate total sizes ---
        total_vertices = 0
        for prim_data in self._raw_meshes.values():
            # Convert memoryview to numpy array before using .view()
            total_vertices += (
                np.array(prim_data.data, copy=False).view(vertex_dtype).shape[0]
            )

        # This dtype MUST match the layout in the WGSL shader
        storage_dtype = np.dtype(
            [
                ("model", "float32", (4, 4)),
                ("normal_matrix", "float32", (4, 4)),
                ("colour", "float32", (4)),
            ]
        )

        # --- Allocate host-side (NumPy) arrays ---
        self.vertex_data = np.empty(total_vertices, dtype=vertex_dtype)
        self.storage_data = np.zeros(self._instance_count, dtype=storage_dtype)

        # --- Pass 2: Fill arrays and store metadata ---
        current_vertex = 0
        # Use sorted keys to ensure a deterministic order
        for instance_index, (name, prim_data) in enumerate(
            sorted(self._raw_meshes.items())
        ):
            # Convert memoryview to numpy array before using .view()
            vertex_count = (
                np.array(prim_data.data, copy=False).view(vertex_dtype).shape[0]
            )

            # Information needed for the draw call
            self._mesh_info[name] = {
                "first_vertex": current_vertex,
                "vertex_count": vertex_count,
                "instance_index": instance_index,
            }

            # Copy vertex data
            prim_vertices = np.array(prim_data.data, copy=False).view(vertex_dtype)
            self.vertex_data[current_vertex : current_vertex + vertex_count] = (
                prim_vertices
            )

            current_vertex += vertex_count

        # --- Create device-side (GPU) buffers ---
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=self.vertex_data,
            usage=wgpu.BufferUsage.VERTEX,
            label="consolidated_vertex_buffer",
        )
        self.storage_buffer = self.device.create_buffer(
            size=self.storage_data.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label="consolidated_storage_buffer",
        )

    def update_mesh_data(self, name: str, model, normal_matrix, colour):
        """
        Updates the transformation/color data for a mesh in the host-side NumPy array.
        """
        if name not in self._mesh_info:
            return
        instance_index = self._mesh_info[name]["instance_index"]
        self.storage_data[instance_index]["model"] = model.to_numpy()
        self.storage_data[instance_index]["colour"] = colour
        self.storage_data[instance_index]["normal_matrix"] = normal_matrix.to_numpy()

    def write_buffers(self):
        """
        Uploads the (potentially updated) host-side storage data to the GPU buffer.
        """
        if self.storage_data is not None and self.storage_buffer is not None:
            self.device.queue.write_buffer(self.storage_buffer, 0, self.storage_data)

    def get_mesh_info(self, name: str):
        """
        Returns the rendering metadata for a given mesh.
        """
        return self._mesh_info.get(name)

    @property
    def num_meshes(self):
        return self._instance_count
