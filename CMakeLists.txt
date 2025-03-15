cmake_minimum_required(VERSION 3.18)
project(VolumetricPathTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

# Include directories
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external
)

# Source files
set(CUDA_SOURCES
    src/kernels/path_tracing.cu
    src/kernels/volume_integration.cu
    src/kernels/subsurface_scattering.cu
    src/kernels/denoising.cu
)

set(CPP_SOURCES
    src/main.cpp
    src/core/scene.cpp
    src/core/camera.cpp
    src/acceleration/bvh.cpp
    src/acceleration/volume_grid.cpp
    src/renderer.cpp
    src/window.cpp
)

set(HEADERS
    src/core/scene.h
    src/core/camera.h
    src/core/materials.h
    src/core/volume.h
    src/math/sampling.cuh
    src/math/monte_carlo.cuh
    src/math/phase_functions.cuh
    src/acceleration/bvh.h
    src/acceleration/volume_grid.h
    src/renderer.h
    src/window.h
    src/common.h
)

# Create executable
add_executable(VolumetricPathTracer ${CPP_SOURCES} ${CUDA_SOURCES} ${HEADERS})

# Set CUDA properties
set_property(TARGET VolumetricPathTracer PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link libraries
target_link_libraries(VolumetricPathTracer
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    glfw
    curand
)

# Compiler flags
target_compile_options(VolumetricPathTracer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --relocatable-device-code=true
        -Xcompiler=-Wall
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra -O3
    >
)

# Create assets directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/assets)

# Copy assets if they exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets")
    file(COPY assets DESTINATION ${CMAKE_BINARY_DIR})
    message(STATUS "Assets directory copied successfully")
else()
    message(WARNING "Assets directory not found, continuing without assets")
endif() 