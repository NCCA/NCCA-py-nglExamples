import json
import logging
from pathlib import Path
from typing import Any, Dict

import OpenGL.GL as gl
from ncca.ngl import Mat3, Mat4, ShaderLib, Vec2, Vec3, Vec4


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
        self.uniform_defs: Dict[str, Any] = {}
        self.load_json(json_file)

    def _parse_uniform_value(self, uniform: Dict[str, Any]) -> Any:
        """Parse a uniform value from the raw JSON data."""
        utype = uniform.get("Type", "")
        uvalue = uniform.get("Value")
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
        return value

    def _parse_and_cache_uniforms(self) -> None:
        """
        Parse uniforms from JSON and store them in self.uniform_defs.
        """
        for uniform_data in self.shader_data.get("Uniforms", []):
            name = uniform_data.get("Name")
            if not name:
                continue

            shader_range = uniform_data.get("Range")
            if shader_range:
                shader_range = Vec2(shader_range[0], shader_range[1])

            self.uniform_defs[name] = {
                "type": uniform_data.get("Type"),
                "value": self._parse_uniform_value(uniform_data),
                "range": shader_range,
            }

    def get_uniform_definitions(self) -> Dict[str, Any]:
        """Returns the parsed uniform definitions."""
        return self.uniform_defs

    def set_uniform_value(self, name: str, value: Any) -> None:
        """Update the cached value of a uniform."""
        if name in self.uniform_defs:
            self.uniform_defs[name]["value"] = value
        else:
            logging.warning(f"Attempted to set non-existent uniform '{name}'")

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
        self._parse_and_cache_uniforms()

    def apply_uniforms(self, MVP: Mat4, MV: Mat4, normal_matrix: Mat3) -> None:
        """
        Set all uniforms for the shader before drawing.

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

        for name, definition in self.uniform_defs.items():
            try:
                if definition["value"] is not None:
                    ShaderLib.set_uniform(name, definition["value"])
            except Exception:
                logging.warning(
                    f"Uniform '{name}' defined in JSON but not found in shader '{shader_name}'"
                )
