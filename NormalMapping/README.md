# Normal Mapping

![](NormalMapping.png)
This demo uses the ngl::Obj and ngl::VertexArrayObject classes to read in a mesh then construct an extended VAO passing in Tangents and Bi-Tangents (BiNormals) to glsl as attributes. This is then used to do normal mapping along the lines of [this](http://www.ozone3d.net/tutorials/bump_mapping.php)

## References

- J. F. Blinn, "Simulation of Wrinkled Surfaces", SIGGRAPH 1978 — [ACM](https://dl.acm.org/doi/10.1145/965139.507101) — the original bump-mapping (perturbed normal) paper.
- [LearnOpenGL — Normal Mapping](https://learnopengl.com/Advanced-Lighting/Normal-Mapping) — tangent-space normal mapping as implemented here.
- [Computing Tangent Space Basis Vectors for an Arbitrary Mesh (Eric Lengyel)](https://terathon.com/blog/tangent-space.html) — the per-vertex tangent/bitangent construction passed as attributes.
- [opengl-tutorial — Normal Mapping](http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/) — another worked TBN-matrix walkthrough.
