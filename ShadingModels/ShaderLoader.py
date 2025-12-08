import json
from pathlib import Path

from ncca.ngl import (
    Mat3,
    Mat4,
    Prims,
    ShaderLib,
    Vec3,
    Vec4,
)


class ShaderLoader:
    def __init__(self, json_file):
        self.shader_data = {}
        self.root_path = Path(json_file).parent
        self.load_json(json_file)

    def load_json(self, json_file):
        print(f"Loading shader data from {json_file}")
        with open(json_file, "r") as file:
            self.shader_data = json.load(file)
        print(self.shader_data)

        vert_path = self.root_path / self.shader_data["VertexShader"]
        frag_path = self.root_path / self.shader_data["FragmentShader"]
        print(f"Loading shader from {vert_path} and {frag_path}")
        ShaderLib.load_shader(self.shader_data["ShaderName"], vert_path, frag_path)

    def set_uniforms(self, MVP: Mat4):
        ShaderLib.use(self.shader_data["ShaderName"])
        ShaderLib.set_uniform("MVP", MVP)
        try:
            for key, val in self.shader_data.items():
                if key == "Uniforms":
                    for uniform in val:
                        if uniform["Type"] == "Vec3":
                            value = Vec3(
                                float(uniform["Value"][0]),
                                float(uniform["Value"][1]),
                                float(uniform["Value"][2]),
                            )
                            print("vec3 ", value, type(value))
                        elif uniform["Type"] == "Vec4":
                            value = Vec4(
                                float(uniform["Value"][0]),
                                float(uniform["Value"][1]),
                                float(uniform["Value"][2]),
                                float(uniform["Value"][3]),
                            )
                            print("vec4 ", value, type(value))
                        elif uniform["Type"] == "Mat3":
                            value = Mat3(
                                float(uniform["Value"][0]),
                                float(uniform["Value"][1]),
                                float(uniform["Value"][2]),
                                float(uniform["Value"][3]),
                                float(uniform["Value"][4]),
                                float(uniform["Value"][5]),
                                float(uniform["Value"][6]),
                                float(uniform["Value"][7]),
                                float(uniform["Value"][8]),
                            )
                        elif uniform["Type"] == "Mat4":
                            value = Mat4(
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
                            )
                        else:
                            value = uniform["Value"]

                        try:
                            ShaderLib.set_uniform(uniform["Name"], value)
                            print(value, type(value))
                            print(f"setting {uniform['Name']} {uniform['Value']}")
                        except Exception:
                            pass
        except Exception as e:
            print(f"Error setting uniform {uniform['Name']}: {e}")
