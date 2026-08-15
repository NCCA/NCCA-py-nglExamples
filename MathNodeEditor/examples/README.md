# MathNodeEditor examples

These are saved graphs for trying the PyNGL `Vec`, `Mat` and `Quaternion`
classes in a few problems we meet in graphics. Open one with `File` → `Open...`
and change the input values; the output nodes update as you type.

## Vectors and linear algebra

| Example | What it shows | Try changing |
| :--- | :--- | :--- |
| [vec3_multiply_demo.json](vec3_multiply_demo.json) | Component-wise `Vec3` multiplication | Either input vector |
| [vector_arithmetic_demo.json](vector_arithmetic_demo.json) | Position updates, displacement, squared distance and a unit direction | Object and target positions |
| [triangle_normal_demo.json](triangle_normal_demo.json) | Two triangle edges and their cross product | Swap points B and C to reverse the normal |
| [lambert_diffuse_demo.json](lambert_diffuse_demo.json) | The normalised `N dot L` term used by diffuse lighting | Move the light below the surface to get a negative result |
| [mat2_rotation_demo.json](mat2_rotation_demo.json) | A `Vec2 @ Mat2` rotation and its transpose undoing the rotation | The point or matrix entries |
| [homogeneous_coordinates_demo.json](homogeneous_coordinates_demo.json) | Why a point has `w=1` and a direction has `w=0` | The translation values |

The Lambert graph leaves the dot product unclamped so we can see the actual
linear algebra result. In a shader we would normally use `max(0.0, dot(N, L))`.

## Matrices and transforms

| Example | What it shows | Try changing |
| :--- | :--- | :--- |
| [transform_order_demo.json](transform_order_demo.json) | Matrix multiplication is not commutative | The scale and translation values |
| [normal_matrix_demo.json](normal_matrix_demo.json) | A naive normal transform compared with the inverse-transpose matrix | Make the scale uniform and compare the two outputs |
| [mvp_demo.json](mvp_demo.json) | `Projection @ View @ Model` composition | Model rotation or camera position |
| [projection_comparison_demo.json](projection_comparison_demo.json) | Perspective, orthographic and asymmetric frustum matrices | Near/far planes and frustum bounds |

## Quaternions

| Example | What it shows | Try changing |
| :--- | :--- | :--- |
| [quaternion_rotation_demo.json](quaternion_rotation_demo.json) | Rotating a `Vec3` with an axis-angle quaternion | Axis and angle |
| [quaternion_slerp_demo.json](quaternion_slerp_demo.json) | Spherical interpolation between two orientations | Blend factor from `0` to `1` |

## Meshes

| Example | What it shows | Try changing |
| :--- | :--- | :--- |
| [mesh_pipeline_demo.json](mesh_pipeline_demo.json) | Rotating vertices and building the normal matrix by hand | Y rotation |
| [mvp_mesh_demo.json](mvp_mesh_demo.json) | Applying a model transform to a displayed cube | Position, rotation and scale |

The mesh examples use [cube.obj](cube.obj), which is stored beside the graphs.
