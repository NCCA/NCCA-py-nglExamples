import json
import os

import OpenGL.GL as gl
from ngl import Texture


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

    s_textures = {}

    @staticmethod
    def load_json(filename):
        """
        Load texture packs from a JSON file.
        The JSON file has an invalid structure with duplicate keys,
        so it needs to be pre-processed before parsing.
        """
        try:
            with open(filename, "r") as f:
                content = f.read()

            # The JSON file uses duplicate "TexturePack" keys, which is invalid JSON.
            # A standard parser only reads the last entry.
            # To fix this, we pre-process the text file into a valid JSON array string.

            if '"TexturePack":' not in content:
                print("JSON file does not appear to contain 'TexturePack' data.")
                # Fallback to standard parsing for other json files.
                data = json.loads(content)
                if "TexturePack" not in data:
                    print("This does not seem to be a valid Texture Pack json file")
                    return False
                # if it is a valid file with one texture pack we can process it.
                data = [data["TexturePack"]]

            else:
                # 1. Remove the redundant "TexturePack": key for each entry.
                processed_content = content.replace('"TexturePack":', "")
                # 2. Remove the outer braces of the root object.
                processed_content = processed_content.strip()
                if processed_content.startswith("{") and processed_content.endswith(
                    "}"
                ):
                    processed_content = processed_content[1:-1]
                # 3. Wrap the string with brackets to create a valid JSON array.
                processed_content = f"[{processed_content}]"
                data = json.loads(processed_content)

        except (IOError, json.JSONDecodeError) as e:
            print(f"Error opening or parsing json file: {e}")
            return False

        print("***************Loading Texture Pack from JSON*****************")

        base_path = ""  # os.path.dirname(filename)

        # `data` is now a list of texture pack dictionaries.
        for texture_pack_data in data:
            pack = []
            material = texture_pack_data.get("material")
            if not material:
                print("Skipping entry as it has no material")
                continue

            print(f"found material {material}")

            textures = texture_pack_data.get("Textures", [])
            for current_texture in textures:
                location = current_texture.get("location")
                name = current_texture.get("name")
                path = current_texture.get("path")
                if location is None or name is None or path is None:
                    continue

                texture_path = os.path.join(base_path, path)
                print(f"Found {name} {location} {texture_path}")

                try:
                    t = _Texture(location, name, texture_path)
                    if t.id != 0:
                        pack.append(t)
                except Exception as e:
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
