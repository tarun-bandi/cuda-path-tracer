# CPU Mandelbrot

A CPU-based volumetric path tracer with support for participating media.

## Dependencies

### Required Dependencies

1. **CMake** (3.10 or later)
   - Download from [CMake Downloads](https://cmake.org/download/)
   - Or install via package manager:
     ```bash
     # Ubuntu/Debian
     sudo apt-get install cmake
     
     # macOS
     brew install cmake
     ```

2. **OpenGL and GLFW**
   - Ubuntu/Debian:
     ```bash
     sudo apt-get install libgl1-mesa-dev libglfw3-dev
     ```
   - macOS:
     ```bash
     brew install glfw
     ```

3. **GLEW**
   - Ubuntu/Debian:
     ```bash
     sudo apt-get install libglew-dev
     ```
   - macOS:
     ```bash
     brew install glew
     ```

## Building

1. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```

2. Configure with CMake:
   ```bash
   cmake ..
   ```

3. Build the project:
   ```bash
   make
   ```

## Running

Run the executable:
```bash
./cuda_mandlebrot
```

For headless rendering:
```bash
./cuda_mandlebrot --headless
```

## Features

- CPU-based path tracing
- Volumetric rendering
- Multiple scattering
- Depth of field
- Subsurface scattering
- OpenGL integration for real-time display

## Controls

- WASD: Move camera
- Mouse: Look around
- Q/E: Move up/down
- Scroll: Adjust FOV
- ESC: Exit

## License

This project is licensed under the MIT License - see the LICENSE file for details.
