#include "../common.h"
#include "../core/scene.h"
#include "../math/sampling.cuh"

// Subsurface scattering using the dipole approximation
__device__ float3 dipoleProfile(float r, const float3& sigma_s_prime, const float3& sigma_a) {
    float3 sigma_t_prime = sigma_s_prime + sigma_a;
    float3 alpha_prime = sigma_s_prime / sigma_t_prime;
    
    float3 D = make_float3(
        1.0f / (3.0f * sigma_t_prime.x),
        1.0f / (3.0f * sigma_t_prime.y),
        1.0f / (3.0f * sigma_t_prime.z)
    );
    
    float3 sigma_tr = make_float3(
        sqrtf(sigma_a.x / D.x),
        sqrtf(sigma_a.y / D.y),
        sqrtf(sigma_a.z / D.z)
    );
    
    // Extrapolation boundary
    float3 A = make_float3(
        (1.0f + 3.0f * D.x) / (1.0f + 2.0f * D.x),
        (1.0f + 3.0f * D.y) / (1.0f + 2.0f * D.y),
        (1.0f + 3.0f * D.z) / (1.0f + 2.0f * D.z)
    );
    
    float3 z_b = 2.0f * A * D;
    
    // Real source distance
    float z_r = 1.0f / sigma_t_prime.x;  // Using first component for simplicity
    float d_r = sqrtf(r * r + z_r * z_r);
    
    // Virtual source distance
    float z_v = z_r + 2.0f * z_b.x;
    float d_v = sqrtf(r * r + z_v * z_v);
    
    // Dipole profile
    float3 profile = make_float3(
        alpha_prime.x / (4.0f * M_PI * D.x) * (expf(-sigma_tr.x * d_r) / (d_r * d_r) - expf(-sigma_tr.x * d_v) / (d_v * d_v)),
        alpha_prime.y / (4.0f * M_PI * D.y) * (expf(-sigma_tr.y * d_r) / (d_r * d_r) - expf(-sigma_tr.y * d_v) / (d_v * d_v)),
        alpha_prime.z / (4.0f * M_PI * D.z) * (expf(-sigma_tr.z * d_r) / (d_r * d_r) - expf(-sigma_tr.z * d_v) / (d_v * d_v))
    );
    
    return profile;
}

// Sample subsurface scattering using importance sampling
__device__ float3 sampleSubsurface(const float3& hit_point, const float3& hit_normal,
                                  const Material& material, curandState* state,
                                  const Sphere* spheres, int num_spheres,
                                  const Box* boxes, int num_boxes,
                                  const Plane* planes, int num_planes) {
    // Sample entry point on the surface
    float2 xi = random_float2(state);
    float3 tangent, bitangent;
    buildOrthonormalBasis(hit_normal, tangent, bitangent);
    
    // Sample disk around entry point
    float2 disk_sample = sampleDisk(xi);
    float radius = material.subsurface_radius * sqrtf(random_float(state));
    float3 entry_offset = radius * (disk_sample.x * tangent + disk_sample.y * bitangent);
    float3 entry_point = hit_point + entry_offset;
    
    // Perform random walk inside the material
    float3 current_pos = entry_point;
    float3 total_transmittance = make_float3(1.0f, 1.0f, 1.0f);
    int max_bounces = 16;
    
    for (int bounce = 0; bounce < max_bounces; bounce++) {
        // Sample step length
        float3 sigma_t = material.absorption + material.scattering;
        float max_sigma = fmaxf(fmaxf(sigma_t.x, sigma_t.y), sigma_t.z);
        float step_length = sampleExponential(random_float(state), max_sigma);
        
        // Sample direction (isotropic scattering inside material)
        float3 scatter_direction = normalize(random_in_unit_sphere(state));
        float3 next_pos = current_pos + step_length * scatter_direction;
        
        // Check if we exit the material
        Ray exit_ray(current_pos, scatter_direction, 0.001f, step_length + 0.001f);
        HitRecord exit_rec;
        
        if (hitScene(exit_ray, exit_ray.tmin, exit_ray.tmax, exit_rec,
                    spheres, num_spheres, boxes, num_boxes, planes, num_planes)) {
            // Found exit point
            float distance = exit_rec.t;
            float3 exit_point = exit_rec.point;
            
            // Compute transmittance
            float3 transmittance = make_float3(
                expf(-sigma_t.x * distance),
                expf(-sigma_t.y * distance),
                expf(-sigma_t.z * distance)
            );
            
            total_transmittance *= transmittance;
            
            // Apply dipole approximation
            float exit_distance = length(exit_point - hit_point);
            float3 dipole_contrib = dipoleProfile(exit_distance, material.scattering, material.absorption);
            
            return total_transmittance * dipole_contrib * material.albedo;
        }
        
        // Continue random walk
        current_pos = next_pos;
        
        // Apply absorption
        float3 step_transmittance = make_float3(
            expf(-material.absorption.x * step_length),
            expf(-material.absorption.y * step_length),
            expf(-material.absorption.z * step_length)
        );
        
        total_transmittance *= step_transmittance;
        
        // Russian roulette termination
        float max_transmittance = fmaxf(fmaxf(total_transmittance.x, total_transmittance.y), total_transmittance.z);
        if (max_transmittance < 0.1f && random_float(state) > max_transmittance) {
            break;
        }
    }
    
    return make_float3(0.0f, 0.0f, 0.0f);
}

// Improved subsurface scattering with multiple importance sampling
__device__ float3 advancedSubsurface(const Ray& r_in, const HitRecord& rec,
                                    const Material& material, curandState* state,
                                    const Sphere* spheres, int num_spheres,
                                    const Box* boxes, int num_boxes,
                                    const Plane* planes, int num_planes,
                                    const Light* lights, int num_lights) {
    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    int num_samples = 4;  // Number of subsurface samples
    
    for (int i = 0; i < num_samples; i++) {
        // Sample subsurface contribution
        float3 subsurface_contrib = sampleSubsurface(rec.point, rec.normal, material, state,
                                                    spheres, num_spheres, boxes, num_boxes, planes, num_planes);
        
        // Sample exit point and compute lighting
        float2 xi = random_float2(state);
        float3 tangent, bitangent;
        buildOrthonormalBasis(rec.normal, tangent, bitangent);
        
        // Sample exit point on hemisphere
        float3 local_dir = sampleCosineHemisphere(xi);
        float3 exit_direction = localToWorld(local_dir, rec.normal);
        
        // Trace ray to find exit point
        Ray subsurface_ray(rec.point + 0.001f * rec.normal, exit_direction);
        HitRecord exit_rec;
        
        if (hitScene(subsurface_ray, subsurface_ray.tmin, material.subsurface_radius * 2.0f, exit_rec,
                    spheres, num_spheres, boxes, num_boxes, planes, num_planes)) {
            
            // Direct lighting at exit point
            float3 direct_lighting = make_float3(0.0f, 0.0f, 0.0f);
            
            for (int light_idx = 0; light_idx < num_lights; light_idx++) {
                const Light& light = lights[light_idx];
                
                // Sample light direction
                float3 light_dir = normalize(light.position - exit_rec.point);
                float light_distance = length(light.position - exit_rec.point);
                
                // Check for shadows
                Ray shadow_ray(exit_rec.point + 0.001f * exit_rec.normal, light_dir, 0.001f, light_distance - 0.001f);
                HitRecord shadow_rec;
                
                if (!hitScene(shadow_ray, shadow_ray.tmin, shadow_ray.tmax, shadow_rec,
                             spheres, num_spheres, boxes, num_boxes, planes, num_planes)) {
                    // No shadow, add lighting contribution
                    float cos_theta = fmaxf(0.0f, dot(exit_rec.normal, light_dir));
                    float attenuation = 1.0f / (light_distance * light_distance);
                    direct_lighting += light.emission * cos_theta * attenuation;
                }
            }
            
            result += subsurface_contrib * direct_lighting;
        }
    }
    
    return result / float(num_samples);
}

// Jensen's photon mapping approach for subsurface scattering
__device__ float3 photonMappingSubsurface(const float3& hit_point, const float3& hit_normal,
                                         const Material& material, curandState* state) {
    // Simplified photon mapping for subsurface scattering
    // In a full implementation, this would use pre-computed photon maps
    
    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    int num_photons = 64;  // Simulated photon samples
    
    for (int i = 0; i < num_photons; i++) {
        // Sample random entry point
        float2 xi = random_float2(state);
        float3 tangent, bitangent;
        buildOrthonormalBasis(hit_normal, tangent, bitangent);
        
        float2 disk_sample = sampleDisk(xi);
        float radius = material.subsurface_radius * sqrtf(random_float(state));
        float3 entry_point = hit_point + radius * (disk_sample.x * tangent + disk_sample.y * bitangent);
        
        // Simulate photon random walk
        float3 photon_pos = entry_point;
        float3 photon_power = make_float3(1.0f, 1.0f, 1.0f);
        
        for (int bounce = 0; bounce < 8; bounce++) {
            // Sample scattering direction
            float3 scatter_dir = normalize(random_in_unit_sphere(state));
            
            // Sample step length
            float3 sigma_t = material.absorption + material.scattering;
            float step = sampleExponential(random_float(state), fmaxf(fmaxf(sigma_t.x, sigma_t.y), sigma_t.z));
            
            photon_pos += step * scatter_dir;
            
            // Apply absorption
            photon_power *= make_float3(
                expf(-material.absorption.x * step),
                expf(-material.absorption.y * step),
                expf(-material.absorption.z * step)
            );
            
            // Check if photon reaches query point
            float distance_to_query = length(photon_pos - hit_point);
            if (distance_to_query < material.subsurface_radius * 0.1f) {
                // Photon contributes to the query point
                float3 contribution = photon_power * material.scattering / (4.0f * M_PI * distance_to_query * distance_to_query);
                result += contribution;
                break;
            }
            
            // Russian roulette
            float max_power = fmaxf(fmaxf(photon_power.x, photon_power.y), photon_power.z);
            if (max_power < 0.1f && random_float(state) > max_power) {
                break;
            }
        }
    }
    
    return result * material.albedo / float(num_photons);
}

// Kernel for computing subsurface scattering contributions
__global__ void subsurfaceScatteringKernel(float3* subsurface_buffer, int width, int height,
                                          const Ray* primary_rays, const HitRecord* hit_records,
                                          const Sphere* spheres, int num_spheres,
                                          const Box* boxes, int num_boxes,
                                          const Plane* planes, int num_planes,
                                          const Material* materials,
                                          const Light* lights, int num_lights,
                                          curandState* states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= width || j >= height) return;
    
    int pixel_index = j * width + i;
    curandState* state = &states[pixel_index];
    
    const HitRecord& rec = hit_records[pixel_index];
    const Ray& ray = primary_rays[pixel_index];
    
    if (rec.material_id >= 0) {
        const Material& material = materials[rec.material_id];
        
        if (material.type == MATERIAL_SUBSURFACE) {
            // Compute subsurface scattering contribution
            float3 subsurface_color = advancedSubsurface(ray, rec, material, state,
                                                        spheres, num_spheres, boxes, num_boxes, planes, num_planes,
                                                        lights, num_lights);
            
            subsurface_buffer[pixel_index] = subsurface_color;
        } else {
            subsurface_buffer[pixel_index] = make_float3(0.0f, 0.0f, 0.0f);
        }
    } else {
        subsurface_buffer[pixel_index] = make_float3(0.0f, 0.0f, 0.0f);
    }
}

// Precompute subsurface scattering lookup table
__global__ void precomputeSubsurfaceLUT(float* lut, int lut_size, 
                                       float3 sigma_s, float3 sigma_a, float max_radius) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= lut_size) return;
    
    float r = (float(i) / float(lut_size - 1)) * max_radius;
    float3 profile = dipoleProfile(r, sigma_s, sigma_a);
    
    // Store luminance value
    lut[i] = 0.299f * profile.x + 0.587f * profile.y + 0.114f * profile.z;
} 