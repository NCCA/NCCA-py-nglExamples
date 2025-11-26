from typing import Any, Dict, Optional

import numpy as np
import wgpu
from ncca.ngl import Mat4, PrimData


class MeshData:
    """
    Manages mesh data for a WebGPU scene, allowing for shared geometries.

    This class handles the storage of unique geometries and multiple instances
    of these geometries. It consolidates vertex data into a single buffer and
    instance data (transformations, colors) into another storage buffer for
    efficient rendering.
    """

    def __init__(self, device: wgpu.GPUDevice):
        """
        Initializes the MeshData manager.

        Args:
            device (wgpu.GPUDevice): The WebGPU device used for buffer creation.
        """
        self.device: wgpu.GPUDevice = device
        # Maps instance name to mesh_name
        self._raw_meshes: Dict[str, str] = {}
        # Maps mesh_name to prim_data
        self._raw_geometries: Dict[str, PrimData] = {}

        self.vertex_data: Optional[np.ndarray] = None
        self.storage_data: Optional[np.ndarray] = None

        self.vertex_buffer: Optional[wgpu.GPUBuffer] = None
        self.storage_buffer: Optional[wgpu.GPUBuffer] = None

        # Metadata for rendering
        self._mesh_info: Dict[str, Dict[str, int]] = {}
        self._geometry_info: Dict[str, Dict[str, int]] = {}
        self._instance_count: int = 0

    def add_geometry(self, mesh_name: str, prim_data: PrimData) -> None:
        """
        Temporarily stores raw geometry data (vertices, normals, etc.).

        This data will be consolidated into a single vertex buffer later.

        Args:
            mesh_name (str): The unique name for this geometry (e.g., 'sphere', 'cube').
            prim_data (PrimData): The primitive data containing vertices and other attributes.
        """
        if mesh_name in self._raw_geometries:
            print(
                f"Warning: Geometry with name '{mesh_name}' already exists and will be overwritten."
            )
        self._raw_geometries[mesh_name] = prim_data

    def add_mesh(self, name: str, mesh_name: str) -> None:
        """
        Creates an instance of a mesh, referring to an existing geometry.

        Each instance can have its own unique transformation and material properties,
        while sharing the underlying vertex data with other instances of the same geometry.

        Args:
            name (str): The unique name for this specific instance (e.g., 'light1', 'player_character').
            mesh_name (str): The name of the geometry to use for this instance, which must
                             have been previously added via `add_geometry`.

        Raises:
            ValueError: If the specified `mesh_name` does not correspond to any loaded geometry.
        """
        if name in self._raw_meshes:
            print(f"Warning: Instance with name '{name}' already exists.")
            return

        if mesh_name not in self._raw_geometries:
            raise ValueError(
                f"Geometry '{mesh_name}' not found. Add it first with add_geometry."
            )

        self._raw_meshes[name] = mesh_name
        self._instance_count = len(self._raw_meshes)

    def create_buffers(self) -> None:
        """
        Consolidates all stored geometries and instances into GPU buffers.

        This method performs the following steps:
        1. Calculates the total number of vertices across all unique geometries.
        2. Creates a single, large NumPy array (`vertex_data`) to hold all vertex attributes.
        3. Populates `vertex_data` and records the vertex offset and count for each geometry.
        4. Creates a NumPy array (`storage_data`) to hold per-instance data (model matrix, color).
        5. Records the instance index for each mesh instance.
        6. Creates the `vertex_buffer` and `storage_buffer` on the GPU and uploads the initial data.
        """
        if not self._raw_geometries:
            return

        vertex_dtype = np.dtype(
            [
                ("position", "float32", (3,)),
                ("normal", "float32", (3,)),
                ("texcoord", "float32", (2,)),
            ]
        )
        storage_dtype = np.dtype(
            [
                ("model", "float32", (4, 4)),
                ("normal_matrix", "float32", (4, 4)),
                ("colour", "float32", (4)),
            ]
        )

        total_vertices = sum(
            np.array(pd.data, copy=False).view(vertex_dtype).shape[0]
            for pd in self._raw_geometries.values()
        )

        self.vertex_data = np.empty(total_vertices, dtype=vertex_dtype)
        self.storage_data = np.zeros(self._instance_count, dtype=storage_dtype)

        current_vertex = 0
        for mesh_name, prim_data in sorted(self._raw_geometries.items()):
            vertex_count = (
                np.array(prim_data.data, copy=False).view(vertex_dtype).shape[0]
            )
            self._geometry_info[mesh_name] = {
                "first_vertex": current_vertex,
                "vertex_count": vertex_count,
            }
            prim_vertices = np.array(prim_data.data, copy=False).view(vertex_dtype)
            self.vertex_data[current_vertex : current_vertex + vertex_count] = (
                prim_vertices
            )
            current_vertex += vertex_count

        instance_index = 0
        for name, mesh_name in sorted(self._raw_meshes.items()):
            geom_info = self._geometry_info[mesh_name]
            self._mesh_info[name] = {
                "first_vertex": geom_info["first_vertex"],
                "vertex_count": geom_info["vertex_count"],
                "instance_index": instance_index,
            }
            instance_index += 1

        self.vertex_buffer = self.device.create_buffer_with_data(
            data=self.vertex_data.tobytes(),
            usage=wgpu.BufferUsage.VERTEX,
            label="consolidated_vertex_buffer",
        )
        if self.storage_data.nbytes > 0:
            self.storage_buffer = self.device.create_buffer(
                size=self.storage_data.nbytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
                label="consolidated_storage_buffer",
            )

    def update_mesh_data(
        self, name: str, model: Mat4, normal_matrix: Mat4, colour: tuple
    ) -> None:
        """
        Updates the instance-specific data (transform, color) for a given mesh.

        This updates the host-side NumPy array. `write_buffers` must be called
        to upload the changes to the GPU.

        Args:
            name (str): The name of the mesh instance to update.
            model (Mat4): The new model transformation matrix.
            normal_matrix (Mat4): The new normal matrix.
            colour (tuple): The new color as a tuple of floats (r, g, b, a).
        """
        if name not in self._mesh_info or self.storage_data is None:
            return
        instance_index = self._mesh_info[name]["instance_index"]
        self.storage_data[instance_index]["model"] = model.to_numpy()
        self.storage_data[instance_index]["normal_matrix"] = normal_matrix.to_numpy()
        self.storage_data[instance_index]["colour"] = colour

    def write_buffers(self) -> None:
        """
        Uploads the host-side storage data to the corresponding GPU buffer.

        This should be called each frame after all mesh data has been updated.
        """
        if self.storage_data is not None and self.storage_buffer is not None:
            self.device.queue.write_buffer(
                self.storage_buffer, 0, self.storage_data.tobytes()
            )

    def get_mesh_info(self, name: str) -> Optional[Dict[str, int]]:
        """
        Returns the rendering metadata for a given mesh instance.

        This metadata includes the vertex offset, vertex count, and instance index
        required for a draw call.

        Args:
            name (str): The name of the mesh instance.

        Returns:
            Optional[Dict[str, int]]: A dictionary with rendering info, or None if not found.
        """
        return self._mesh_info.get(name)

    @property
    def num_meshes(self) -> int:
        """
        Returns the total number of mesh instances.
        """
        return self._instance_count
