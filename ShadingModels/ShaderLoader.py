import json
from pathlib import Path
from typing import Any, Dict, List

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, ShaderLib, Vec3, Vec4


class ShaderLoader:
    """A class to load and manage shaders from a JSON definition file."""

    def __init__(self, json_file: str) -> None:
        """
        Initialize the ShaderLoader.

        Args:
            json_file: The path to the JSON file defining the shader.
        """
        self.shader_data: Dict[str, Any] = {}
        self.root_path: Path = Path(json_file).parent
        self.has_normal_matrix: bool = False
        self.has_model_view: bool = False
        self.load_json(json_file)

    @property
    def uniforms(self) -> List[Dict[str, Any]]:
        """
        Get the list of uniforms defined in the shader data.

        Returns:
            A list of dictionaries, where each dictionary describes a uniform.
        """
        return self.shader_data.get("Uniforms", [])

    def load_json(self, json_file: str) -> None:
        """
        Load shader data from a JSON file and create the shader program.

        Args:
            json_file: The path to the JSON file.
        """
        print(f"Loading shader data from {json_file}")
        with open(json_file, "r") as file:
            self.shader_data = json.load(file)

        vert_path = self.root_path / self.shader_data["VertexShader"]
        frag_path = self.root_path / self.shader_data["FragmentShader"]
        shader_name = self.shader_data["ShaderName"]
        ShaderLib.load_shader(shader_name, str(vert_path), str(frag_path))

        ShaderLib.use(shader_name)
        program_id = ShaderLib.get_program_id(shader_name)

        self.has_normal_matrix = (
            gl.glGetUniformLocation(program_id, "normal_matrix") != -1
        )
        self.has_model_view = gl.glGetUniformLocation(program_id, "MV") != -1

    def set_uniforms(self, MVP: Mat4, MV: Mat4, normal_matrix: Mat3) -> None:
        """
        Set the uniforms for the shader.

        This includes standard matrices (MVP, MV, normal_matrix) and any custom
        uniforms defined in the JSON file.

        Args:
            MVP: The Model-View-Projection matrix.
            MV: The Model-View matrix.
            normal_matrix: The normal matrix.
        """
        shader_name = self.shader_data["ShaderName"]
        ShaderLib.use(shader_name)

        ShaderLib.set_uniform("MVP", MVP)
        if self.has_normal_matrix:
            ShaderLib.set_uniform("normal_matrix", normal_matrix)
        if self.has_model_view:
            ShaderLib.set_uniform("MV", MV)

        uniform_name = "<unknown>"
        try:
            for uniform in self.uniforms:
                uniform_name = uniform.get("Name", "<unknown>")
                utype = uniform.get("Type", "")
                uvalue = uniform["Value"]
                value: Any = None

                if utype in ["Vec3", "Colour3"]:
                    value = Vec3(float(uvalue[0]), float(uvalue[1]), float(uvalue[2]))
                elif utype in ["Vec4", "Colour4"]:
                    value = Vec4(
                        float(uvalue[0]),
                        float(uvalue[1]),
                        float(uvalue[2]),
                        float(uvalue[3]),
                    )
                elif utype == "Mat3":
                    value = Mat3.from_list([float(v) for v in uvalue])
                elif utype == "Mat4":
                    value = Mat4.from_list([float(v) for v in uvalue])
                elif str(utype).lower() == "float":
                    try:
                        if isinstance(uvalue, list) and len(uvalue) > 0:
                            value = float(uvalue[0])
                        else:
                            value = float(uvalue)
                    except (ValueError, TypeError):
                        value = 0.0
                else:
                    value = uvalue

                if value is not None:
                    try:
                        ShaderLib.set_uniform(uniform_name, value)
                    except Exception:
                        # Fail silently if the uniform doesn't exist in the shader
                        pass
        except Exception as e:
            print(f"Error processing uniform {uniform_name}: {e}")
