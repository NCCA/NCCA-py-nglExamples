import json
from pathlib import Path

import OpenGL.GL as gl
from ncca.ngl import (
    Mat3,
    Mat4,
    ShaderLib,
    Vec3,
    Vec4,
)


class ShaderLoader:
    def __init__(self, json_file):
        self.shader_data = {}
        self.root_path = Path(json_file).parent
        self.load_json(json_file)

    @property
    def uniforms(self):
        return self.shader_data.get("Uniforms", [])

    def load_json(self, json_file):
        print(f"Loading shader data from {json_file}")
        with open(json_file, "r") as file:
            self.shader_data = json.load(file)

        vert_path = self.root_path / self.shader_data["VertexShader"]
        frag_path = self.root_path / self.shader_data["FragmentShader"]
        ShaderLib.load_shader(self.shader_data["ShaderName"], vert_path, frag_path)

        id = ShaderLib.get_program_id(self.shader_data["ShaderName"])

        if gl.glGetUniformLocation(id, "normal_matrix") != -1:
            self.has_normal_matrix = True
        else:
            self.has_normal_matrix = False

        if gl.glGetUniformLocation(id, "MV") != -1:
            self.has_model_view = True
        else:
            self.has_model_view = False

    def set_uniforms(self, MVP: Mat4, MV: Mat4, normal_matrix: Mat3):
        ShaderLib.use(self.shader_data["ShaderName"])

        ShaderLib.set_uniform("MVP", MVP)
        if self.has_normal_matrix:
            ShaderLib.set_uniform("normal_matrix", normal_matrix)
        if self.has_model_view:
            ShaderLib.set_uniform("MV", MV)
        try:
            for key, val in self.shader_data.items():
                if key == "Uniforms":
                    for uniform in val:
                        if uniform["Type"] in ["Vec3", "Colour3"]:
                            value = Vec3(
                                float(uniform["Value"][0]),
                                float(uniform["Value"][1]),
                                float(uniform["Value"][2]),
                            )
                        elif uniform["Type"] in ["Vec4", "Colour4"]:
                            value = Vec4(
                                float(uniform["Value"][0]),
                                float(uniform["Value"][1]),
                                float(uniform["Value"][2]),
                                float(uniform["Value"][3]),
                            )
                            print("vec4 ", value, type(value))
                        elif uniform["Type"] == "Mat3":
                            value = Mat3.from_list(
                                [
                                    float(uniform["Value"][0]),
                                    float(uniform["Value"][1]),
                                    float(uniform["Value"][2]),
                                    float(uniform["Value"][3]),
                                    float(uniform["Value"][4]),
                                    float(uniform["Value"][5]),
                                    float(uniform["Value"][6]),
                                    float(uniform["Value"][7]),
                                    float(uniform["Value"][8]),
                                ],
                            )
                        elif uniform["Type"] == "Mat4":
                            value = Mat4.from_list(
                                [
                                    float(uniform["Value"][0]),
                                    float(uniform["Value"][1]),
                                    float(uniform["Value"][2]),
                                    float(uniform["Value"][3]),
                                    float(uniform["Value"][4]),
                                    float(uniform["Value"][5]),
                                    float(uniform["Value"][6]),
                                    float(uniform["Value"][7]),
                                    float(uniform["Value"][8]),
                                    float(uniform["Value"][9]),
                                    float(uniform["Value"][10]),
                                    float(uniform["Value"][11]),
                                    float(uniform["Value"][12]),
                                    float(uniform["Value"][13]),
                                    float(uniform["Value"][14]),
                                    float(uniform["Value"][15]),
                                ]
                            )
                        else:
                            value = uniform["Value"]

                        try:
                            ShaderLib.set_uniform(uniform["Name"], value)
                        except Exception:
                            pass
        except Exception as e:
            print(f"Error setting uniform {uniform['Name']}: {e}")
