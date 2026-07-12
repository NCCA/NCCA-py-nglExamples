# PBRTexture


This demo shows how to render using Physically Based Rendering (PBR) techniques as outlined in https://learnopengl.com/PBR/Lighting

In particular this demo uses a class called texture_pack which can read in a json file containing a list of textures and locations for easy texture management.

## References

- [LearnOpenGL — PBR Theory](https://learnopengl.com/PBR/Theory) and [PBR Lighting](https://learnopengl.com/PBR/Lighting) — the textured metallic/roughness workflow implemented here.
- R. L. Cook & K. E. Torrance, "A Reflectance Model for Computer Graphics", ACM TOG 1982 — [ACM](https://dl.acm.org/doi/10.1145/357290.357293) — the microfacet BRDF underlying modern PBR.
- B. Karis, "Real Shading in Unreal Engine 4", SIGGRAPH 2013 course — [course page](https://blog.selfshadow.com/publications/s2013-shading-course/) — the GGX/Smith/Schlick term choices most real-time PBR follows.
- [FreePBR](https://freepbr.com/) — source of the albedo/normal/metallic/roughness/AO texture sets.
