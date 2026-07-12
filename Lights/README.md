# Lights


Simple demo showing how to use the use structures for lights, this demo also demonstrates the editShader methods to dynamically edit the shader source and update how many lights are being used.

Use the keys 1/2 to add and remove lights (note there is a limit of max  lights due to the  Implementation limit of 128 varying components. )

## References

- [LearnOpenGL — Multiple lights](https://learnopengl.com/Lighting/Multiple-lights) — structuring GLSL light structs/arrays and summing their contributions.
- [OpenGL Wiki — Uniform (GLSL)](https://www.khronos.org/opengl/wiki/Uniform_(GLSL)) — uniform blocks, structs and the varying-component limits that cap the light count.
- B. T. Phong, "Illumination for Computer Generated Pictures", CACM 1975 — [ACM](https://dl.acm.org/doi/10.1145/360825.360839) — the lighting model evaluated per light.
