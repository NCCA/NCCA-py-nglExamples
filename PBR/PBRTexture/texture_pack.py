import os
from typing import ClassVar

import OpenGL.GL as gl
from ncca.ngl.opengl import Texture
from texture_pack_parser import parse_texture_packs


class _Texture:
    def __init__(self, location, name, path):
        """
        A class to represent a single texture.
        It loads a texture from a file and creates an OpenGL texture.
        """
        self.location = location
        self.name = name
        self.id = 0

        if not os.path.exists(path):
            print(f"Texture file not found at {path}")
            return

        # This assumes py-ngl has a similar API to the C++ version.
        # 1. Load texture data from file
        texture = Texture(path)

        # 2. Activate texture unit
        gl.glActiveTexture(gl.GL_TEXTURE0 + location)
        # 3. Create OpenGL texture and get its ID
        self.id = texture.set_texture_gl()
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.id)
        # 4. Set texture parameters
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(
            gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST_MIPMAP_LINEAR
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        # Mipmap filtering requires mipmaps to be generated.
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)


class TexturePack:
    """
    A class to manage texture packs loaded from a JSON file.
    """

    s_textures: ClassVar = {}

    @staticmethod
    def load_json(filename):
        """
        Load texture packs from a JSON file.
        """
        data = parse_texture_packs(filename)
        if not data:
            return False

        print("***************Loading Texture Pack from JSON*****************")

        base_path = ""  # os.path.dirname(filename)

        for texture_pack_data in data:
            pack = []
            material = texture_pack_data.material

            print(f"found material {material}")

            for current_texture in texture_pack_data.textures:
                texture_path = os.path.join(base_path, current_texture.path)
                print(
                    f"Found {current_texture.name} "
                    f"{current_texture.location} {texture_path}"
                )

                try:
                    t = _Texture(
                        current_texture.location, current_texture.name, texture_path
                    )
                    if t.id != 0:
                        pack.append(t)
                except Exception as e:  # noqa: BLE001 - image loaders use backend-specific errors.
                    print(f"Error loading texture {texture_path}: {e}")

            TexturePack.s_textures[material] = pack
        return True

    @staticmethod
    def activate_texture_pack(tname):
        """
        Activate a loaded texture pack by name.
        This binds all textures in the pack to their respective texture units.
        """
        pack = TexturePack.s_textures.get(tname)
        if pack:
            for t in pack:
                gl.glActiveTexture(gl.GL_TEXTURE0 + t.location)
                gl.glBindTexture(gl.GL_TEXTURE_2D, t.id)
            return True
        print(f"Texture pack '{tname}' not found")
        return False
