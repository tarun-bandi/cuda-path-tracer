#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include "../common.h"
#include "../core/scene.h"
#include "../core/camera.h"
#include "../core/materials.h"
#include "../core/volume.h"
#include "../acceleration/bvh.h"
#include "../acceleration/volume_grid.h"

// Device functions for ray-scene intersection
__device__ bool hitScene(const Ray& r, float t_min, float t_max, HitRecord& rec,
                        const Sphere* spheres, int num_spheres,
                        const Box* boxes, int num_boxes,
                        const Plane* planes, int num_planes) {
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    
    // Test spheres
    for (int i = 0; i < num_spheres; i++) {
        if (spheres[i].hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    
    // Test boxes
    for (int i = 0; i < num_boxes; i++) {
        if (boxes[i].hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    
    // Test planes
    for (int i = 0; i < num_planes; i++) {
        if (planes[i].hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    
    return hit_anything;
}

// Check if a point is inside any volume
__device__ VolumeRecord sampleVolume(const Ray& r, float t_min, float t_max,
                                   const Volume* volumes, int num_volumes,
                                   curandState* state) {
    VolumeRecord vol_rec;
    
    // Find which volume the ray is in
    for (int i = 0; i < num_volumes; i++) {
        const Volume& vol = volumes[i];
        
        // Check if ray intersects volume bounds
        float3 inv_d = make_float3(1.0f / r.direction.x, 1.0f / r.direction.y, 1.0f / r.direction.z);
        float3 t0 = (vol.min_bounds - r.origin) * inv_d;
        float3 t1 = (vol.max_bounds - r.origin) * inv_d;
        
        float3 t_near = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
        float3 t_far = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));
        
        float t_entry = fmaxf(fmaxf(t_near.x, t_near.y), t_near.z);
        float t_exit = fminf(fminf(t_far.x, t_far.y), t_far.z);
        
        if (t_entry <= t_exit && t_exit > t_min && t_entry < t_max) {
            // Ray intersects volume, sample a point inside
            float t_start = fmaxf(t_entry, t_min);
            float t_end = fminf(t_exit, t_max);
            
            // Delta tracking / Woodcock tracking
            float t = t_start;
            float max_sigma_t = fmaxf(fmaxf(vol.sigma_t.x, vol.sigma_t.y), vol.sigma_t.z);
            
            while (t < t_end) {
                float dt = sampleExponential(random_float(state), max_sigma_t);
                t += dt;
                
                if (t >= t_end) break;
                
                float3 sample_point = r.at(t);
                float3 sigma_t = vol.getSigmaT(sample_point);
                float prob = fmaxf(fmaxf(sigma_t.x, sigma_t.y), sigma_t.z) / max_sigma_t;
                
                if (random_float(state) < prob) {
                    // Real scattering event
                    vol_rec.point = sample_point;
                    vol_rec.t = t;
                    vol_rec.volume_id = i;
                    vol_rec.sigma_t = sigma_t;
                    vol_rec.sigma_s = vol.getSigmaS(sample_point);
                    vol_rec.g = vol.g;
                    vol_rec.valid = true;
                    return vol_rec;
                }
                // Null collision, continue
            }
        }
    }
    
    return vol_rec;  // No volume interaction
}

// Material scattering functions
__device__ bool scatter(const Ray& r_in, const HitRecord& rec, float3& attenuation, Ray& scattered,
                       const Material& material, curandState* state) {
    switch (material.type) {
        case MATERIAL_LAMBERTIAN: {
            float3 scatter_direction = rec.normal + normalize(random_in_unit_sphere(state));
            if (length_squared(scatter_direction) < 1e-6f) {
                scatter_direction = rec.normal;
            }
            scattered = Ray(rec.point, scatter_direction);
            attenuation = material.albedo;
            return true;
        }
        
        case MATERIAL_METAL: {
            float3 reflected = reflect(normalize(r_in.direction), rec.normal);
            reflected = reflected + material.roughness * random_in_unit_sphere(state);
            scattered = Ray(rec.point, reflected);
            attenuation = material.albedo;
            return dot(scattered.direction, rec.normal) > 0;
        }
        
        case MATERIAL_GLASS: {
            attenuation = make_float3(1.0f, 1.0f, 1.0f);
            float refraction_ratio = rec.front_face ? (1.0f / material.ior) : material.ior;
            
            float3 unit_direction = normalize(r_in.direction);
            float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
            float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
            
            bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
            float3 direction;
            
            if (cannot_refract || schlick(cos_theta, refraction_ratio) > random_float(state)) {
                direction = reflect(unit_direction, rec.normal);
            } else {
                direction = refract(unit_direction, rec.normal, refraction_ratio);
            }
            
            scattered = Ray(rec.point, direction);
            return true;
        }
        
        case MATERIAL_EMISSIVE: {
            return false;  // Emissive materials don't scatter
        }
        
        default:
            return false;
    }
}

// Volume scattering function
__device__ bool scatterVolume(const Ray& r_in, const VolumeRecord& vol_rec, float3& attenuation, 
                             Ray& scattered, curandState* state) {
    // Sample scattering direction using Henyey-Greenstein phase function
    float2 xi = random_float2(state);
    float3 local_direction = sampleHenyeyGreenstein(vol_rec.g, xi);
    
    // Transform to world coordinates
    float3 scatter_direction = localToWorld(local_direction, normalize(r_in.direction));
    
    scattered = Ray(vol_rec.point, scatter_direction);
    
    // Compute attenuation (scattering albedo)
    attenuation = vol_rec.sigma_s / vol_rec.sigma_t;
    
    return true;
}

// Get emission from material
__device__ float3 emitted(const Material& material, float u, float v, const float3& point) {
    if (material.type == MATERIAL_EMISSIVE) {
        return material.emission;
    }
    return make_float3(0.0f, 0.0f, 0.0f);
}

// Main path tracing function
__device__ float3 rayColor(Ray r, 
                          const Sphere* spheres, int num_spheres,
                          const Box* boxes, int num_boxes,
                          const Plane* planes, int num_planes,
                          const Material* materials,
                          const Volume* volumes, int num_volumes,
                          const Light* lights, int num_lights,
                          curandState* state,
                          int max_depth) {
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    
    for (int depth = 0; depth < max_depth; depth++) {
        HitRecord rec;
        VolumeRecord vol_rec;
        
        // Check for volume interactions first
        vol_rec = sampleVolume(r, r.tmin, r.tmax, volumes, num_volumes, state);
        
        if (vol_rec.valid) {
            // Volume scattering event
            float3 attenuation;
            Ray scattered;
            
            if (scatterVolume(r, vol_rec, attenuation, scattered, state)) {
                throughput *= attenuation;
                r = scattered;
                
                // Russian roulette for path termination
                float p = fmaxf(fmaxf(throughput.x, throughput.y), throughput.z);
                if (depth > 3 && random_float(state) > p) {
                    break;
                }
                throughput /= p;
                continue;
            } else {
                break;
            }
        }
        
        // Check for surface interactions
        if (hitScene(r, r.tmin, 1e30f, rec, spheres, num_spheres, boxes, num_boxes, planes, num_planes)) {
            const Material& material = materials[rec.material_id];
            
            // Add emission
            color += throughput * emitted(material, rec.u, rec.v, rec.point);
            
            // Scatter
            float3 attenuation;
            Ray scattered;
            
            if (scatter(r, rec, attenuation, scattered, material, state)) {
                throughput *= attenuation;
                r = scattered;
                
                // Russian roulette
                float p = fmaxf(fmaxf(throughput.x, throughput.y), throughput.z);
                if (depth > 3 && random_float(state) > p) {
                    break;
                }
                throughput /= p;
            } else {
                break;
            }
        } else {
            // Hit sky/environment
            float3 unit_direction = normalize(r.direction);
            float t = 0.5f * (unit_direction.y + 1.0f);
            float3 sky_color = (1.0f - t) * make_float3(1.0f, 1.0f, 1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);
            color += throughput * sky_color * 0.5f;  // Dim sky
            break;
        }
    }
    
    return color;
}

// CUDA kernel declarations
__global__ void curand_init_kernel(curandState* states, int n, unsigned int seed);
__global__ void path_trace_kernel(
    float3* output,
    const Camera* camera,
    const Scene* scene,
    const BVH* bvh,
    const VolumeGrid* volume_grid,
    int width,
    int height,
    int samples_per_pixel,
    curandState* states
);

// Helper functions
__device__ float3 sample_material(const Material& material, const float3& wo, const float3& normal, float3& wi, curandState* state, float* pdf);
__device__ bool near_zero(const float3& v);

// CUDA kernel implementations
__global__ void curand_init_kernel(curandState* states, int n, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void path_trace_kernel(
    float3* output,
    const Camera* camera,
    const Scene* scene,
    const BVH* bvh,
    const VolumeGrid* volume_grid,
    int width,
    int height,
    int samples_per_pixel,
    curandState* states
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    curandState* state = &states[idx];
    float3 pixel_color = make_float3(0.0f);

    for (int s = 0; s < samples_per_pixel; ++s) {
        float u = (x + curand_uniform(state)) / width;
        float v = (y + curand_uniform(state)) / height;
        Ray ray = camera->get_ray(u, v, state);
        float3 path_throughput = make_float3(1.0f);
        float3 path_radiance = make_float3(0.0f);

        for (int depth = 0; depth < 8; ++depth) {
            Intersection isect;
            if (bvh->intersect(ray, isect)) {
                const Material& material = scene->materials[isect.material_id];
                path_radiance += path_throughput * material.emission;

                float3 wi;
                float pdf;
                float3 f = sample_material(material, -ray.direction, isect.normal, wi, state, &pdf);

                if (pdf > 0.0f && !near_zero(f)) {
                    path_throughput *= f * fabsf(dot(wi, isect.normal)) / pdf;
                    ray = Ray(isect.position, wi);
                } else {
                    break;
                }
            } else {
                path_radiance += path_throughput * scene->environment_map.sample(ray.direction);
                break;
            }

            // Russian roulette
            if (depth > 3) {
                float q = 1.0f - std::min(1.0f, std::max(path_throughput.x, std::max(path_throughput.y, path_throughput.z)));
                if (curand_uniform(state) < q) break;
                path_throughput /= (1.0f - q);
            }
        }

        pixel_color += path_radiance;
    }

    output[idx] = pixel_color / samples_per_pixel;
}

// Helper function implementations
__device__ float3 sample_material(const Material& material, const float3& wo, const float3& normal, float3& wi, curandState* state, float* pdf) {
    // Implement material sampling here
    // This is a placeholder - you'll need to implement the actual material sampling logic
    return make_float3(1.0f);
}

__device__ bool near_zero(const float3& v) {
    const float s = 1e-8;
    return (fabsf(v.x) < s) && (fabsf(v.y) < s) && (fabsf(v.z) < s);
}

// Host function to launch the kernel
void launch_path_trace(
    float3* output,
    const Camera* camera,
    const Scene* scene,
    const BVH* bvh,
    const VolumeGrid* volume_grid,
    int width,
    int height,
    int samples_per_pixel
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    curandState* states;
    cudaMalloc(&states, width * height * sizeof(curandState));

    curand_init_kernel<<<grid, block>>>(states, width * height, time(NULL));
    path_trace_kernel<<<grid, block>>>(
        output,
        camera,
        scene,
        bvh,
        volume_grid,
        width,
        height,
        samples_per_pixel,
        states
    );

    cudaFree(states);
} 