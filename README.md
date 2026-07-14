# Py-NGL Demos

This is work in progress examples for the new all python version of the NGL library. As well as a Demo Launcher / Runner. 

It is expected you will use uv to run all the python applications. All demos can be launched by running the RunDemos.py file.

![](DemoApp.png)

Whilst I will mainly be using WebGPU for my teaching I thought it would be good to also have an OpenGL example for quick demos and easier coding in some cases.

You can see the Source for the [PyNGL](https://github.com/NCCA/PyNGL) on GitHub and it can be installed via [PyPi](https://pypi.org/project/ncca-ngl/) Full details of PyNGL and how to use it can be found at [https://ncca.github.io/PyNGL/](https://ncca.github.io/PyNGL/)

Each demo lives in its own folder with a `README.md` explaining it. Click a preview or demo name below to open that folder.

## Contents

- [Getting Started / Templates](#getting-started--templates)
- [OpenGL Fundamentals](#opengl-fundamentals)
- [Vertex Array Objects](#vertex-array-objects)
- [Instancing &amp; Performance](#instancing--performance)
- [Geometry &amp; Meshes](#geometry--meshes)
- [Transforms &amp; Hierarchy](#transforms--hierarchy)
- [Animation](#animation)
- [Textures &amp; Materials](#textures--materials)
- [Blending &amp; Transparency](#blending--transparency)
- [Environment &amp; Sky](#environment--sky)
- [Curves &amp; Interpolation](#curves--interpolation)
- [Selection &amp; Picking](#selection--picking)
- [Framebuffers &amp; Post Processing](#framebuffers--post-processing)
- [Lighting &amp; Shadows](#lighting--shadows)
- [Ray Marching](#ray-marching)
- [Geometry &amp; Tessellation Shaders](#geometry--tessellation-shaders)
- [Uniforms &amp; Buffers](#uniforms--buffers)
- [Compute Shaders](#compute-shaders)
- [WebGPU](#webgpu)
- [Particles &amp; Points](#particles--points)
- [GUI](#gui)

## Getting Started / Templates

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="BlankPySide6NGL"><img src="BlankPySide6NGL/BlankPySideNGL.png" width="220"></a> | [BlankPySide6NGL](BlankPySide6NGL) | Minimal PySide6 + NGL OpenGL window template |
| <a href="BlankPySDL3"><img src="BlankPySDL3/BlankPySDL3.png" width="220"></a> | [BlankPySDL3](BlankPySDL3) | Minimal PySDL3 + NGL OpenGL window template |
| <a href="BlankWebGPU"><img src="BlankWebGPU/BlankWebGPU.png" width="220"></a> | [BlankWebGPU](BlankWebGPU) | Minimal WebGPU window template |
| <a href="SimplePyNGL"><img src="SimplePyNGL/PySDL3NGLDemo.png" width="220"></a> | [SimplePyNGL](SimplePyNGL) | Simple first PyNGL examples |
| <a href="SimpleWebGPU"><img src="SimpleWebGPU/WebGPUNGL.png" width="220"></a> | [SimpleWebGPU](SimpleWebGPU) | Simple first WebGPU example |

## OpenGL Fundamentals

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="2DDrawingOpenGL"><img src="2DDrawingOpenGL/2D.png" width="220"></a> | [2DDrawingOpenGL](2DDrawingOpenGL) | 2D drawing with OpenGL |
| <a href="Camera"><img src="Camera/Camera.png" width="220"></a> | [Camera](Camera) | Camera / view and projection setup |
| <a href="Lights"><img src="Lights/Lights.png" width="220"></a> | [Lights](Lights) | Basic lighting |
| <a href="ShadingModels"><img src="ShadingModels/ShadingModels.png" width="220"></a> | [ShadingModels](ShadingModels) | Comparison of shading models |
| <a href="ScreenTri"><img src="ScreenTri/ScreenTri.png" width="220"></a> | [ScreenTri](ScreenTri) | Full-screen triangle technique |
| <a href="OpenGLPrimRestart"><img src="OpenGLPrimRestart/PrimRestart.png" width="220"></a> | [OpenGLPrimRestart](OpenGLPrimRestart) | Primitive restart index |
| <a href="FrustumCull"><img src="FrustumCull/FrustumCull.png" width="220"></a> | [FrustumCull](FrustumCull) | View frustum culling |
| <a href="FontRendering"><img src="FontRendering/FontRender.png" width="220"></a> | [FontRendering](FontRendering) | Text / font rendering |

## Vertex Array Objects

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="VAOPrimitives"><img src="VAOPrimitives/VAOPrimitives.png" width="220"></a> | [VAOPrimitives](VAOPrimitives) | Built-in VAO primitives |
| <a href="VertexArrayObject/Sphere"><img src="VertexArrayObject/Sphere/Sphere.png" width="220"></a> | [VertexArrayObject/Sphere](VertexArrayObject/Sphere) | Generating a sphere VAO |
| <a href="VertexArrayObject/Boid"><img src="VertexArrayObject/Boid/Boid.png" width="220"></a> | [VertexArrayObject/Boid](VertexArrayObject/Boid) | Simple Boid VAO |
| <a href="VertexArrayObject/BoidShaded"><img src="VertexArrayObject/BoidShaded/BoidShaded.png" width="220"></a> | [VertexArrayObject/BoidShaded](VertexArrayObject/BoidShaded) | Shaded Boid VAO |
| <a href="VertexArrayObject/ChangingVAO"><img src="VertexArrayObject/ChangingVAO/ChangingVAO.png" width="220"></a> | [VertexArrayObject/ChangingVAO](VertexArrayObject/ChangingVAO) | Updating VAO data |
| <a href="VertexArrayObject/ChangingVAOMultiBuffer"><img src="VertexArrayObject/ChangingVAOMultiBuffer/ChangingVAO.png" width="220"></a> | [VertexArrayObject/ChangingVAOMultiBuffer](VertexArrayObject/ChangingVAOMultiBuffer) | Updating a multi-buffer VAO |
| <a href="VertexArrayObject/MultiBufferVAO"><img src="VertexArrayObject/MultiBufferVAO/MBBoid.png" width="220"></a> | [VertexArrayObject/MultiBufferVAO](VertexArrayObject/MultiBufferVAO) | Multi-buffer VAO |
| <a href="VertexArrayObject/SimpleIndexVAOFactory"><img src="VertexArrayObject/SimpleIndexVAOFactory/IndexVAO.png" width="220"></a> | [VertexArrayObject/SimpleIndexVAOFactory](VertexArrayObject/SimpleIndexVAOFactory) | Indexed VAO factory |
| <a href="VertexArrayObject/ExtendedVAOFactory"><img src="VertexArrayObject/ExtendedVAOFactory/ExtendedVAO.png" width="220"></a> | [VertexArrayObject/ExtendedVAOFactory](VertexArrayObject/ExtendedVAOFactory) | Custom / extended VAO factory |

## Instancing &amp; Performance

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="Instancing"><img src="Instancing/Instancing.png" width="220"></a> | [Instancing](Instancing) | GPU instancing vs a Python draw-call loop, with an on-screen frame-time HUD (OpenGL + WebGPU) |

## Geometry &amp; Meshes

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="ObjViewer"><img src="ObjViewer/ObjDemo.png" width="220"></a> | [ObjViewer](ObjViewer) | Load and view Obj meshes |
| <a href="ColourObj"><img src="ColourObj/ColourObj.png" width="220"></a> | [ColourObj](ColourObj) | Obj mesh with per-vertex colour |
| <a href="Obj2Numpy"><img src="Obj2Numpy/Obj2Numpy.png" width="220"></a> | [Obj2Numpy](Obj2Numpy) | Convert Obj data to NumPy arrays |
| <a href="KleinBottle"><img src="KleinBottle/KleinBottle.png" width="220"></a> | [KleinBottle](KleinBottle) | Procedural Klein bottle |
| <a href="MarchingCubes"><img src="MarchingCubes/MarchingCubes.png" width="220"></a> | [MarchingCubes](MarchingCubes) | Metaballs polygonised into a mesh every frame with vectorised Marching Cubes |

## Transforms &amp; Hierarchy

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="SceneGraph"><img src="SceneGraph/SceneGraph.png" width="220"></a> | [SceneGraph](SceneGraph) | A robot arm built from a minimal, unit-tested transform-hierarchy `Node` class |

## Animation

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="SkeletalAnimation"><img src="SkeletalAnimation/SkeletalAnimation.png" width="220"></a> | [SkeletalAnimation](SkeletalAnimation) | Linear blend skinning vs dual-quaternion skinning, and the "candy wrapper" artefact that tells them apart (OpenGL) |

## Textures &amp; Materials

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="SimpleTexture"><img src="SimpleTexture/Texture.png" width="220"></a> | [SimpleTexture](SimpleTexture) | Basic texture mapping |
| <a href="AnimatedTextures"><img src="AnimatedTextures/AnimatedTextures.png" width="220"></a> | [AnimatedTextures](AnimatedTextures) | Animated / scrolling textures |
| <a href="ShowMipmap"><img src="ShowMipmap/MipMap.png" width="220"></a> | [ShowMipmap](ShowMipmap) | Visualising mipmap levels |
| <a href="ImageHeightMap"><img src="ImageHeightMap/ImageHeightMap.png" width="220"></a> | [ImageHeightMap](ImageHeightMap) | Displacement from a height map image |
| <a href="NormalMapping"><img src="NormalMapping/NormalMapping.png" width="220"></a> | [NormalMapping](NormalMapping) | Tangent-space normal mapping |
| <a href="PBR/SimplePBR"><img src="PBR/SimplePBR/SimplePBR.png" width="220"></a> | [PBR/SimplePBR](PBR/SimplePBR) | Simple physically based rendering |
| <a href="PBR/PBRTexture"><img src="PBR/PBRTexture/PBRTexture.png" width="220"></a> | [PBR/PBRTexture](PBR/PBRTexture) | Textured physically based rendering |
| <a href="Billboards"><img src="Billboards/Billboards.png" width="220"></a> | [Billboards](Billboards) | Camera-facing quads: fixed / cylindrical / spherical billboarding modes (OpenGL) |

## Blending &amp; Transparency

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="Blending"><img src="Blending/Blending.png" width="220"></a> | [Blending](Blending) | Alpha blending, depth write and sorting toggles (OpenGL + WebGPU) |
| <a href="OITransparency"><img src="OITransparency/OIT.png" width="220"></a> | [OITransparency](OITransparency) | Weighted blended order-independent transparency (OpenGL + WebGPU) |

## Environment &amp; Sky

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="SkyBoxEnvMap"><img src="SkyBoxEnvMap/SkyBoxEnvMap.png" width="220"></a> | [SkyBoxEnvMap](SkyBoxEnvMap) | Procedural cubemap skybox with reflect/refract/Fresnel teapot (OpenGL + WebGPU) |

## Curves &amp; Interpolation

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="CurveDemos"><img src="CurveDemos/CurveDemos.png" width="220"></a> | [CurveDemos](CurveDemos) | Curve types and evaluation |
| <a href="EasingFunctions"><img src="EasingFunctions/EasingFunctions.png" width="220"></a> | [EasingFunctions](EasingFunctions) | Full Penner easing set with combo box selection and live matplotlib graph |
| <a href="Interpolation"><img src="Interpolation/Interpolation.png" width="220"></a> | [Interpolation](Interpolation) | Interpolation techniques |
| <a href="QuatSlerp"><img src="QuatSlerp/QuatSlerp.png" width="220"></a> | [QuatSlerp](QuatSlerp) | Quaternion spherical interpolation |

## Selection &amp; Picking

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="ColourSelectionOpenGL"><img src="ColourSelectionOpenGL/ColourSelect.png" width="220"></a> | [ColourSelectionOpenGL](ColourSelectionOpenGL) | Unique colour-ID picking |
| <a href="RayPickingSelection"><img src="RayPickingSelection/RayPickingSelection.png" width="220"></a> | [RayPickingSelection](RayPickingSelection) | Ray-cast selection and manipulation |
| <a href="SelectionManipulator"><img src="SelectionManipulator/SelectionManipulator.png" width="220"></a> | [SelectionManipulator](SelectionManipulator) | Maya-style manipulator (OpenGL) |
| <a href="SelectionManipulatorWebGPU"><img src="SelectionManipulatorWebGPU/SelectionManipulatorWebGPU.png" width="220"></a> | [SelectionManipulatorWebGPU](SelectionManipulatorWebGPU) | Manipulator (WebGPU) |
| <a href="WebGPUComputePicking"><img src="WebGPUComputePicking/WebGPUPick.png" width="220"></a> | [WebGPUComputePicking](WebGPUComputePicking) | Compute-shader picking (WebGPU) |
| <a href="StencilOutline"><img src="StencilOutline/StencilOutline.png" width="220"></a> | [StencilOutline](StencilOutline) | Maya-style two-pass stencil-buffer selection outline (OpenGL) |

## Framebuffers &amp; Post Processing

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="FBODemos/SimpleFBO"><img src="FBODemos/SimpleFBO/SimpleFBO.png" width="220"></a> | [FBODemos/SimpleFBO](FBODemos/SimpleFBO) | Simple framebuffer object |
| <a href="FBODemos/Blit"><img src="FBODemos/Blit/Blit.png" width="220"></a> | [FBODemos/Blit](FBODemos/Blit) | Blitting between framebuffers |
| <a href="FBODemos/DOF"><img src="FBODemos/DOF/DOF.png" width="220"></a> | [FBODemos/DOF](FBODemos/DOF) | Depth of field |
| <a href="FBODemos/WebGPURenderToTexture"><img src="FBODemos/WebGPURenderToTexture/WebGPURenderToTexture.png" width="220"></a> | [FBODemos/WebGPURenderToTexture](FBODemos/WebGPURenderToTexture) | Render to texture (WebGPU) |
| <a href="PostProcessChain"><img src="PostProcessChain/PostProcessChain.png" width="220"></a> | [PostProcessChain](PostProcessChain) | HDR bloom + tonemap post-process chain (OpenGL) |

## Lighting &amp; Shadows

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="DefferedLighting"><img src="DefferedLighting/WebGPUNGL.png" width="220"></a> | [DefferedLighting](DefferedLighting) | Deferred lighting (WebGPU) |
| <a href="WebGPUShadows"><img src="WebGPUShadows/WebGPUShadows.png" width="220"></a> | [WebGPUShadows](WebGPUShadows) | PCF shadow mapping (WebGPU) |
| <a href="ShadowMapping"><img src="ShadowMapping/ShadowMapping.png" width="220"></a> | [ShadowMapping](ShadowMapping) | Two-pass depth-map shadows with PCF, bias and culling toggles (OpenGL) |

## Ray Marching

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="RayMarchingSDF"><img src="RayMarchingSDF/RayMarchingSDF.png" width="220"></a> | [RayMarchingSDF](RayMarchingSDF) | Sphere-traced signed distance fields, smooth-blended primitives, soft shadows + AO (OpenGL &amp; WebGPU) |

## Geometry &amp; Tessellation Shaders

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="GeometryTessellation"><img src="GeometryTessellation/GeometryTessellation.png" width="220"></a> | [GeometryTessellation](GeometryTessellation) | Geometry-shader normal visualiser + distance-LOD tessellated noise plane (OpenGL) |

## Uniforms &amp; Buffers

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="UBOStorageBuffers"><img src="UBOStorageBuffers/UBOStorageBuffers.png" width="220"></a> | [UBOStorageBuffers](UBOStorageBuffers) | UBO shared across two shader programs + std140 padding trap (OpenGL); runtime-sized storage-buffer point lights (WebGPU) |

## Compute Shaders

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="SimpleComputeWebGPU"><img src="SimpleComputeWebGPU/SimpleComputeWebGPU.png" width="220"></a> | [SimpleComputeWebGPU](SimpleComputeWebGPU) | Simple compute shader (WebGPU) |
| <a href="WebGPUCompute/SpatialHash2D"><img src="WebGPUCompute/SpatialHash2D/SpatialHash2D.png" width="220"></a> | [WebGPUCompute/SpatialHash2D](WebGPUCompute/SpatialHash2D) | 2D spatial hashing on the GPU |
| <a href="WebGPUCompute/SpatialHash3D"><img src="WebGPUCompute/SpatialHash3D/SpatialHash3D.png" width="220"></a> | [WebGPUCompute/SpatialHash3D](WebGPUCompute/SpatialHash3D) | 3D spatial hashing on the GPU |
| <a href="BoidsCompute"><img src="BoidsCompute/BoidsCompute.png" width="220"></a> | [BoidsCompute](BoidsCompute) | Reynolds flocking with compute shaders + instanced rendering |

## WebGPU

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="WebGPUMultiGeo"><img src="WebGPUMultiGeo/WebGPUMulti.png" width="220"></a> | [WebGPUMultiGeo](WebGPUMultiGeo) | Multiple geometry in one WebGPU scene |

## Particles &amp; Points

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="Particles/ParticleQuads"><img src="Particles/ParticleQuads/ParticleQuads.png" width="220"></a> | [Particles/ParticleQuads](Particles/ParticleQuads) | Billboarded particle quads |
| <a href="PointCloud"><img src="PointCloud/PointCloud.png" width="220"></a> | [PointCloud](PointCloud) | Rendering a point cloud |
| <a href="Voxels"><img src="Voxels/Voxels.png" width="220"></a> | [Voxels](Voxels) | Voxel rendering |

## GUI

| Preview | Demo | Description |
| :---: | :--- | :--- |
| <a href="GUIDemos/PySideGUIOpenGL"><img src="GUIDemos/PySideGUIOpenGL/PySideGUI.png" width="220"></a> | [GUIDemos/PySideGUIOpenGL](GUIDemos/PySideGUIOpenGL) | PySide GUI driving an OpenGL widget |
| <a href="GUIDemos/NGLWidgetsOpenGL"><img src="GUIDemos/NGLWidgetsOpenGL/PySideGUI.png" width="220"></a> | [GUIDemos/NGLWidgetsOpenGL](GUIDemos/NGLWidgetsOpenGL) | NGL widgets with OpenGL |
| <a href="GUIDemos/WebGPUGUI"><img src="GUIDemos/WebGPUGUI/WebGPUGUI.png" width="220"></a> | [GUIDemos/WebGPUGUI](GUIDemos/WebGPUGUI) | GUI driving a WebGPU widget |
| <a href="GUIDemos/QMLOverlayApp"><img src="GUIDemos/QMLOverlayApp/QMLOverlayApp.png" width="220"></a> | [GUIDemos/QMLOverlayApp](GUIDemos/QMLOverlayApp) | QWidget OpenGL viewport with a transparent QQuickWidget overlay of floating ncca.ngl.qml panels |
| <a href="GUIDemos/QMLWebGPUOverlay"><img src="GUIDemos/QMLWebGPUOverlay/QMLWebGPUOverlay.png" width="220"></a> | [GUIDemos/QMLWebGPUOverlay](GUIDemos/QMLWebGPUOverlay) | WebGPU (offscreen) viewport with a transparent QQuickWidget overlay of floating ncca.ngl.qml panels |
| <a href="SciFiUI"><img src="SciFiUI/SciFiUI.png" width="220"></a> | [SciFiUI](SciFiUI) | Retro sci-fi CRT terminal: Unknown Pleasures style wireframe terrain fly-over, clickable buttons, scrolling log and a two-pass CRT post process |
