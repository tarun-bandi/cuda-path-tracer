#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include "core/camera.h"
#include "core/scene.h"
#include "core/materials.h"
#include "core/volume.h"
#include "acceleration/bvh.h"
#include "acceleration/volume_grid.h"
#include "kernels/path_tracing.cu"
#include "window.h"

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Global variables
GLFWwindow* window = nullptr;
cudaGraphicsResource* cuda_pbo_resource = nullptr;
GLuint pbo = 0;
float3* d_output = nullptr;
int width = 1280;
int height = 720;
int samples_per_pixel = 1024;
bool render_complete = false;
bool headless_mode = false;

// Function declarations
void setup_scene(Scene& scene, Camera& camera, BVH& bvh, VolumeGrid& volume_grid);
void render();
void display();
void cleanup();

// Scene setup
void setup_scene(Scene& scene, Camera& camera, BVH& bvh, VolumeGrid& volume_grid) {
    // Camera setup
    camera = Camera(
        glm::vec3(0, 2, 5),  // position
        glm::vec3(0, 0, 0),  // target
        glm::vec3(0, 1, 0),  // up
        45.0f,               // fov
        float(width) / height, // aspect ratio
        0.1f,                // aperture
        10.0f                // focus distance
    );
    
    // Materials
    Material ground_material = {
        glm::vec3(0.8f, 0.8f, 0.8f),  // albedo
        0.0f,                         // metallic
        0.5f,                         // roughness
        1.5f,                         // ior
        0.0f,                         // transmission
        glm::vec3(0.0f)              // emission
    };
    
    Material glass_material = {
        glm::vec3(1.0f),             // albedo
        0.0f,                         // metallic
        0.0f,                         // roughness
        1.5f,                         // ior
        1.0f,                         // transmission
        glm::vec3(0.0f)              // emission
    };
    
    Material light_material = {
        glm::vec3(0.0f),             // albedo
        0.0f,                         // metallic
        0.0f,                         // roughness
        1.0f,                         // ior
        0.0f,                         // transmission
        glm::vec3(10.0f)             // emission
    };
    
    // Add materials to scene
    scene.materials.push_back(ground_material);
    scene.materials.push_back(glass_material);
    scene.materials.push_back(light_material);
    
    // Create ground plane
    scene.add_plane(glm::vec3(0, 0, 0), glm::vec3(0, 1, 0), 0);
    
    // Create glass sphere
    scene.add_sphere(glm::vec3(0, 1, 0), 1.0f, 1);
    
    // Create light
    scene.add_sphere(glm::vec3(0, 5, 0), 0.5f, 2);
    
    // Setup volume grid
    volume_grid.init(
        glm::vec3(-2, 0, -2),  // min bounds
        glm::vec3(2, 4, 2),    // max bounds
        make_int3(32, 32, 32)  // resolution
    );
    
    // Add some volumetric media
    Volume smoke = {
        glm::vec3(0.1f),       // sigma_a
        glm::vec3(0.8f),       // sigma_s
        glm::vec3(0.9f),       // sigma_t
        0.2f,                  // g
        0.5f                   // density
    };
    
    // Add smoke to grid
    for (int z = 0; z < 32; ++z) {
        for (int y = 0; y < 32; ++y) {
            for (int x = 0; x < 32; ++x) {
                float3 pos = make_float3(
                    -2.0f + 4.0f * x / 31.0f,
                    4.0f * y / 31.0f,
                    -2.0f + 4.0f * z / 31.0f
                );
                
                // Create a smoke plume
                float dist = length(pos - make_float3(0, 2, 0));
                float density = 0.5f * expf(-dist * dist);
                
                if (density > 0.01f) {
                    smoke.density = density;
                    volume_grid.set_volume(x, y, z, smoke);
                }
            }
        }
    }
    
    // Build BVH
    bvh.build(scene);
}

// Main render loop
void render() {
    if (render_complete) return;
    
    // Map PBO to CUDA
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_pbo_resource));
    size_t num_bytes;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_output, &num_bytes, cuda_pbo_resource));
    
    // Launch path tracing kernel
    launch_path_trace(
        d_output,
        &camera,
        &scene,
        &bvh,
        &volume_grid,
        width,
        height,
        samples_per_pixel
    );
    
    // Unmap PBO
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_pbo_resource));
    
    render_complete = true;
}

// Display callback
void display() {
    render();
    
    // Display the rendered image
    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, 0);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(-1, -1);
    glTexCoord2f(1, 0); glVertex2f(1, -1);
    glTexCoord2f(1, 1); glVertex2f(1, 1);
    glTexCoord2f(0, 1); glVertex2f(-1, 1);
    glEnd();
    
    glfwSwapBuffers(window);
}

int main(int argc, char** argv) {
    // Check for headless mode
    if (argc > 1 && std::string(argv[1]) == "--headless") {
        headless_mode = true;
    }

    if (!headless_mode) {
        // Initialize GLFW
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            return -1;
        }

        // Create window
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        window = glfwCreateWindow(width, height, "Volumetric Path Tracer", nullptr, nullptr);
        if (!window) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return -1;
        }

        glfwMakeContextCurrent(window);
        init_window();
    }

    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));

    // Create PBO
    if (!headless_mode) {
        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(float3), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // Register PBO with CUDA
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));
    } else {
        // Allocate device memory for headless mode
        CUDA_CHECK(cudaMalloc(&d_output, width * height * sizeof(float3)));
    }

    // Create texture
    GLuint texture;
    if (!headless_mode) {
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, nullptr);
    }

    // Setup scene
    Scene scene;
    Camera camera;
    BVH bvh;
    VolumeGrid volume_grid;
    setup_scene(scene, camera, bvh, volume_grid);

    if (headless_mode) {
        // Render in headless mode
        render();
        
        // Save output image
        std::vector<float3> h_output(width * height);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, width * height * sizeof(float3), cudaMemcpyDeviceToHost));
        
        // Save as PPM
        std::ofstream out("output.ppm");
        out << "P3\n" << width << " " << height << "\n255\n";
        for (int i = 0; i < width * height; i++) {
            float3 pixel = h_output[i];
            int r = static_cast<int>(255.99f * sqrtf(pixel.x));
            int g = static_cast<int>(255.99f * sqrtf(pixel.y));
            int b = static_cast<int>(255.99f * sqrtf(pixel.z));
            out << r << " " << g << " " << b << "\n";
        }
        out.close();
    } else {
        // Main loop
        while (!glfwWindowShouldClose(window)) {
            display();
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    // Cleanup
    cleanup();
    return 0;
}

void cleanup() {
    if (!headless_mode) {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &texture);
        glfwDestroyWindow(window);
        glfwTerminate();
    } else {
        cudaFree(d_output);
    }
} 