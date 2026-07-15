"""WebGPU texture-pack management for the PBR demo.

The OpenGL ``texture_pack`` binds five textures to global texture units and
re-binds them before every draw with ``glActiveTexture``. WebGPU has no global
texture-unit state; resources are handed to the pipeline through bind groups.

So the idiom changes: each material loads its five maps (albedo, normal,
metallic, roughness, AO) into their own ``GPUTexture`` and bakes a single
``GPUBindGroup`` holding all five views plus a shared sampler. Selecting a
material at draw time is then just ``set_bind_group`` - no per-unit rebinding.
"""

import json
import os

import numpy as np
import wgpu
from ncca.ngl import Image

# The five maps occupy bindings 0-4 in @group(2); the sampler is binding 5.
# The JSON "location" field already numbers the maps albedo..ao as 0..4.
_SAMPLER_BINDING = 5
_NUM_MAPS = 5


class TexturePack:
    """Loads material texture packs from JSON into WebGPU bind groups."""

    def __init__(
        self,
        device: wgpu.GPUDevice,
        bind_group_layout: wgpu.GPUBindGroupLayout,
    ) -> None:
        """Create a texture-pack manager.

        Args:
            device: The WebGPU device that owns the textures.
            bind_group_layout: The @group(2) layout every pack is built against;
                it must match the material bindings in ``PBRTexture.wgsl``.
        """
        self.device = device
        self.layout = bind_group_layout
        # One sampler is shared by every pack - repeat wrap, linear filtering.
        self.sampler = device.create_sampler(
            address_mode_u=wgpu.AddressMode.repeat,
            address_mode_v=wgpu.AddressMode.repeat,
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            mipmap_filter=wgpu.FilterMode.linear,
        )
        self.packs: dict[str, wgpu.GPUBindGroup] = {}
        # Keep the GPUTexture objects alive for the life of the manager.
        self._textures: dict[str, list[wgpu.GPUTexture]] = {}

    def load_json(self, filename: str) -> bool:
        """Load every texture pack described in a textures.json file.

        The file reuses the OpenGL demo's format, which repeats the
        ``"TexturePack"`` key - invalid JSON that needs pre-processing into an
        array before it will parse.
        """
        try:
            with open(filename, "r") as f:
                content = f.read()

            if '"TexturePack":' not in content:
                # Fall back to standard parsing for a single-pack file.
                data = json.loads(content)
                if "TexturePack" not in data:
                    print("This does not seem to be a valid Texture Pack json file")
                    return False
                data = [data["TexturePack"]]
            else:
                # Strip the repeated key and outer braces, then wrap as an array.
                processed = content.replace('"TexturePack":', "").strip()
                if processed.startswith("{") and processed.endswith("}"):
                    processed = processed[1:-1]
                data = json.loads(f"[{processed}]")
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error opening or parsing json file: {e}")
            return False

        print("*************** Loading WebGPU Texture Packs from JSON ***************")
        for texture_pack_data in data:
            material = texture_pack_data.get("material")
            if not material:
                print("Skipping entry as it has no material")
                continue
            print(f"found material {material}")
            self._load_material(material, texture_pack_data.get("Textures", []))
        return True

    def _load_material(self, material: str, textures: list) -> None:
        """Load one material's maps and bake its bind group."""
        views: list[wgpu.GPUTextureView | None] = [None] * _NUM_MAPS
        gpu_textures: list[wgpu.GPUTexture] = []

        for current in textures:
            location = current.get("location")
            path = current.get("path")
            if location is None or path is None or not 0 <= location < _NUM_MAPS:
                continue
            texture = self._create_texture(path)
            if texture is None:
                continue
            gpu_textures.append(texture)
            views[location] = texture.create_view()

        if any(v is None for v in views):
            print(f"Material '{material}' is missing one or more maps, skipping")
            return

        entries = [{"binding": i, "resource": views[i]} for i in range(_NUM_MAPS)]
        entries.append({"binding": _SAMPLER_BINDING, "resource": self.sampler})
        self.packs[material] = self.device.create_bind_group(
            label=f"{material}_texture_pack", layout=self.layout, entries=entries
        )
        self._textures[material] = gpu_textures

    def _create_texture(self, path: str) -> wgpu.GPUTexture | None:
        """Load an image file and upload it as an rgba8unorm texture."""
        if not os.path.exists(path):
            print(f"Texture file not found at {path}")
            return None

        pixels = Image(path).get_pixels()
        rgba = self._to_rgba(pixels)

        height, width = rgba.shape[0], rgba.shape[1]
        size = (width, height, 1)
        texture = self.device.create_texture(
            size=size,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba8unorm,
            mip_level_count=1,
            sample_count=1,
        )
        self.device.queue.write_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            rgba.tobytes(),
            {"bytes_per_row": width * 4, "rows_per_image": height},
            size,
        )
        return texture

    @staticmethod
    def _to_rgba(pixels: np.ndarray) -> np.ndarray:
        """Expand any image to a contiguous RGBA uint8 array.

        The metallic, roughness and AO maps are single-channel grayscale; the
        shader reads them from the red channel, so grayscale values are
        replicated across RGB with an opaque alpha.
        """
        pixels = np.asarray(pixels, dtype=np.uint8)
        if pixels.ndim == 2:  # grayscale
            pixels = pixels[:, :, np.newaxis]
        height, width, channels = pixels.shape
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255
        if channels == 1:
            rgba[:, :, 0] = rgba[:, :, 1] = rgba[:, :, 2] = pixels[:, :, 0]
        else:
            rgba[:, :, :3] = pixels[:, :, :3]
            if channels == 4:
                rgba[:, :, 3] = pixels[:, :, 3]
        return rgba

    def activate(self, render_pass: wgpu.GPURenderPassEncoder, material: str) -> bool:
        """Bind a material's texture pack into @group(2) for the next draw.

        This replaces the OpenGL ``activate_texture_pack`` five-unit rebind with
        a single ``set_bind_group`` call.
        """
        bind_group = self.packs.get(material)
        if bind_group is None:
            print(f"Texture pack '{material}' not found")
            return False
        render_pass.set_bind_group(2, bind_group)
        return True

    @property
    def materials(self) -> list[str]:
        """The names of every loaded material."""
        return list(self.packs.keys())
