#include "../common.h"
#include "../core/volume.h"
#include "../math/phase_functions.cuh"

__device__ float3 integrate_volume(
    const Ray& ray,
    const Volume& volume,
    curandState* state,
    float t_min,
    float t_max
) {
    float3 L = make_float3(0.0f);
    float3 Tr = make_float3(1.0f);
    float t = t_min;

    while (true) {
        // Sample distance to next scattering event
        float3 sigma_t = volume.sigma_t * volume.density;
        float t_scatter = -logf(1.0f - random_float(state)) / 
            (sigma_t.x + sigma_t.y + sigma_t.z);

        // Check if we hit a surface before scattering
        if (t + t_scatter >= t_max) {
            // Add contribution from light source at surface
            Tr *= expf(-sigma_t * (t_max - t));
            break;
        }

        // Move to scattering point
        t += t_scatter;
        float3 p = ray.point_at(t);

        // Update transmittance
        Tr *= expf(-sigma_t * t_scatter);

        // Sample phase function
        float3 wo = -ray.direction;
        float3 wi;
        float pdf;
        float3 phase = sample_henyey_greenstein(wo, wi, volume.g, state, &pdf);

        // Compute scattering contribution
        float3 scatter = volume.sigma_s * volume.density * phase / pdf;
        L += Tr * scatter;

        // Russian roulette
        float q = 1.0f - min(1.0f, max(Tr.x, max(Tr.y, Tr.z)));
        if (random_float(state) < q) break;
        Tr /= (1.0f - q);
    }

    return L;
}

__device__ float3 estimate_volume_radiance(
    const Ray& ray,
    const Volume& volume,
    curandState* state,
    float t_min,
    float t_max
) {
    // Multiple importance sampling between direct lighting and volume integration
    float3 L = make_float3(0.0f);
    
    // Direct lighting contribution
    float3 direct_L = integrate_volume(ray, volume, state, t_min, t_max);
    
    // Volume integration contribution
    float3 volume_L = make_float3(0.0f);
    float t = t_min;
    float3 Tr = make_float3(1.0f);
    
    while (true) {
        // Sample distance to next scattering event
        float3 sigma_t = volume.sigma_t * volume.density;
        float t_scatter = -logf(1.0f - random_float(state)) / 
            (sigma_t.x + sigma_t.y + sigma_t.z);

        if (t + t_scatter >= t_max) {
            Tr *= expf(-sigma_t * (t_max - t));
            break;
        }

        t += t_scatter;
        float3 p = ray.point_at(t);
        Tr *= expf(-sigma_t * t_scatter);

        // Sample light source
        float3 wi;
        float light_pdf;
        float3 Li = sample_light(p, wi, state, &light_pdf);
        
        if (light_pdf > 0.0f) {
            float3 phase = henyey_greenstein(wi, -ray.direction, volume.g);
            float3 scatter = volume.sigma_s * volume.density * phase;
            volume_L += Tr * scatter * Li / light_pdf;
        }

        // Russian roulette
        float q = 1.0f - min(1.0f, max(Tr.x, max(Tr.y, Tr.z)));
        if (random_float(state) < q) break;
        Tr /= (1.0f - q);
    }

    // Combine contributions using power heuristic
    float w_direct = 1.0f;
    float w_volume = 1.0f;
    L = (w_direct * direct_L + w_volume * volume_L) / (w_direct + w_volume);

    return L;
} 