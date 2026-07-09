# WebGPU 3D Spatial Hashing Demo

This demo extends the 2D spatial hashing collision detection system to 3D, showcasing efficient particle collision detection in three-dimensional space using WebGPU compute shaders.

## Features

### Core Features
- **3D Spatial Hashing**: Efficient collision detection using a 3D grid-based spatial hash
- **Particle System**: Thousands of particles with physics simulation
- **Collision Detection**: Elastic collisions between particles
- **3D Camera Controls**: Full 3D navigation with rotation and zoom
- **Real-time Performance**: GPU-accelerated compute shaders for parallel processing

### Visual Features
- **Instanced Diffuse Spheres**: Every particle is an instanced, diffuse-lit sphere (single draw call via `MULTI_COLOURED_INSTANCED_GEOMETRY`) rather than a flat point sprite
- **3D Grid Visualization**: Toggle-able grid showing spatial hash structure
- **Particle Colors**: Randomly colored particles for easy tracking
- **3D Perspective**: Proper perspective projection for 3D rendering
- **Interactive Camera**: Mouse-based camera rotation and zoom

## Files

- `CollisionCompute3D.wgsl` - WebGPU compute shader with 3D spatial hashing
- `WebGPU3D.py` - Core 3D WebGPU application
- `WebGPU3DGui.py` - GUI version with control panel
- `README.md` - This documentation

## Key Differences from 2D Version

### Shader Changes
1. **Particle Structure**: Extended from `vec2` to `vec3` for positions and velocities
2. **Grid Hashing**: 3D coordinate hashing for spatial cell lookup
3. **Collision Detection**: Checks 27 neighboring cells (3×3×3 cube) instead of 9
4. **Boundary Handling**: 3D boundary wrapping

### Application Changes
1. **Camera System**: Full 3D camera with perspective projection
2. **Mouse Controls**: Left-drag rotation, scroll wheel zoom
3. **Rendering**: Instanced, diffuse-lit sphere rendering with depth buffer (a single shared sphere mesh, generated via `PrimData.sphere()`, drawn once per particle)
4. **UI Controls**: Additional controls for Z-axis wind and camera distance

## Running the Demo

### Basic Version
```bash
# Run with 1000 particles (default)
python WebGPU3D.py

# Run with 5000 particles, equispaced distribution
python WebGPU3D.py -p 5000 --equispaced

# Run in debug mode
python WebGPU3D.py -d
```

### GUI Version
```bash
# Run GUI version
python WebGPU3DGui.py

# Run GUI with specific particle count
python WebGPU3DGui.py -p 2000 --random
```

## Controls

### Mouse
- **Left Drag**: Rotate camera around scene
- **Scroll Wheel**: Zoom in/out

### Keyboard
- **A**: Toggle animation
- **G**: Toggle grid display
- **Space**: Reset camera and wind
- **Arrow Keys**: Adjust wind in X/Y directions
- **Page Up/Down**: Adjust wind in Z direction
- **ESC**: Exit application

## GUI Controls

The GUI version provides comprehensive controls for:

### Simulation Parameters
- Particle count (100-2,500,000)
- Distribution type (random/equispaced)
- Simulation dimensions (width, height, depth)
- Grid cell size
- Particle radius

### Physics Parameters
- 3D wind controls (X, Y, Z axes)
- Animation toggle

### Display Parameters
- Grid visibility
- Sphere scale adjustment

### Camera Controls
- Camera distance
- Rotation X and Y angles
- Reset camera button

## Technical Details

### Spatial Hashing Algorithm
1. **Grid Division**: 3D space divided into uniform cells
2. **Particle Assignment**: Each particle hashed to a grid cell based on position
3. **Collision Detection**: Only check particles in neighboring cells
4. **Performance**: O(n) complexity vs O(n²) for brute force

### 3D Hash Function
```wgsl
fn grid_hash(grid_pos: vec3<i32>) -> u32 {
    let x = clamp(grid_pos.x, 0, grid_width - 1);
    let y = clamp(grid_pos.y, 0, grid_height - 1);
    let z = clamp(grid_pos.z, 0, grid_depth - 1);
    return u32((z * grid_height + y) * grid_width + x);
}
```

### Compute Pipeline
The simulation uses 6 compute passes:
1. **Clear Grid**: Reset cell counters and offsets
2. **Count Particles**: Count particles per cell
3. **Build Offsets**: Create prefix sum for particle indexing
4. **Fill Grid**: Populate grid with particle indices
5. **Detect Collisions**: Check and resolve particle collisions
6. **Update Physics**: Update positions and handle boundaries

## Performance Considerations

- **Grid Cell Size**: Should be proportional to particle radius for optimal performance
- **Particle Count**: Scales well to millions of particles due to spatial hashing
- **Memory Usage**: O(n) for particles + O(grid_cells) for spatial hash
- **GPU Utilization**: Compute shaders process particles in parallel

## Extensions

Possible extensions to explore:
- Different particle shapes (boxes, capsules)
- Variable particle sizes
- Force fields and attractions
- Obstacle collision
- Multi-threaded grid updates
- Level-of-detail rendering for distant particles