# SimplePBR

This demo shows how to render using Physically Based Rendering (PBR) techniques as outlined in https://learnopengl.com/PBR/Lighting

This demo uses a simple albedo model to demonstrate the basics of PBR rendering / metallic workflow.

## References

- [LearnOpenGL — PBR Theory](https://learnopengl.com/PBR/Theory) and [PBR Lighting](https://learnopengl.com/PBR/Lighting) — the Cook-Torrance metallic workflow implemented here.
- R. L. Cook & K. E. Torrance, "A Reflectance Model for Computer Graphics", ACM TOG 1982 — [ACM](https://dl.acm.org/doi/10.1145/357290.357293) — the microfacet BRDF underlying modern PBR.
- B. Karis, "Real Shading in Unreal Engine 4", SIGGRAPH 2013 course — [course page](https://blog.selfshadow.com/publications/s2013-shading-course/) — the GGX/Smith/Schlick term choices most real-time PBR (including this shader) follows.
