# CUDA Volumetric Path Tracer

A real-time volumetric path tracer implemented in CUDA, featuring:
- Interactive camera controls
- Volumetric rendering with Henyey-Greenstein phase function
- Multiple importance sampling
- Russian roulette for unbiased rendering
- OpenGL integration for real-time display

## Running in Google Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Upload the `setup_colab.ipynb` notebook
3. Make sure to select a GPU runtime:
   - Click "Runtime" -> "Change runtime type"
   - Select "GPU" as the hardware accelerator
4. Run all cells in the notebook

## Features

- Real-time interactive rendering
- Volumetric light transport
- Multiple scattering
- Depth of field
- Subsurface scattering
- Denoising

## Scene Setup

The default scene includes:
- A ground plane
- A glass sphere
- A light source
- A volumetric smoke plume

## Controls

- WASD: Move camera
- Mouse: Look around
- Q/E: Move up/down
- Scroll: Adjust FOV
- ESC: Exit

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit
- GLFW3
- GLM
- CMake

## Building from Source

```bash
mkdir build
cd build
cmake ..
make
```

## License

MIT License # Performance improvements
# Add volume rendering support
# Add advanced materials
# Add interactive features
# Performance improvements
# Add volume rendering support
# Add advanced materials
# Add interactive features
