#pragma once

#include "../common.h"

enum MaterialType {
    MATERIAL_LAMBERTIAN = 0,
    MATERIAL_METAL,
    MATERIAL_GLASS,
    MATERIAL_SUBSURFACE,
    MATERIAL_PARTICIPATING,
    MATERIAL_EMISSIVE
};

struct Material {
    float3 albedo;
    float3 emission;
    float3 absorption;    // For subsurface scattering (sigma_a)
    float3 scattering;    // Scattering coefficient (sigma_s)
    float asymmetry;      // Henyey-Greenstein g parameter
    float ior;            // Index of refraction
    float roughness;
    float subsurface_radius;
    int type;
    int volume_id;        // Reference to volume if participating medium
    
    __host__ __device__ Material() :
        albedo(make_float3(0.7f, 0.7f, 0.7f)),
        emission(make_float3(0.0f, 0.0f, 0.0f)),
        absorption(make_float3(0.1f, 0.1f, 0.1f)),
        scattering(make_float3(1.0f, 1.0f, 1.0f)),
        asymmetry(0.0f),
        ior(1.5f),
        roughness(0.0f),
        subsurface_radius(0.1f),
        type(MATERIAL_LAMBERTIAN),
        volume_id(-1)
    {}
    
    __host__ __device__ static Material createLambertian(const float3& color) {
        Material mat;
        mat.albedo = color;
        mat.type = MATERIAL_LAMBERTIAN;
        return mat;
    }
    
    __host__ __device__ static Material createMetal(const float3& color, float roughness) {
        Material mat;
        mat.albedo = color;
        mat.roughness = roughness;
        mat.type = MATERIAL_METAL;
        return mat;
    }
    
    __host__ __device__ static Material createGlass(float ior) {
        Material mat;
        mat.albedo = make_float3(1.0f, 1.0f, 1.0f);
        mat.ior = ior;
        mat.type = MATERIAL_GLASS;
        return mat;
    }
    
    __host__ __device__ static Material createSubsurface(const float3& albedo, 
                                                          const float3& absorption,
                                                          const float3& scattering,
                                                          float radius) {
        Material mat;
        mat.albedo = albedo;
        mat.absorption = absorption;
        mat.scattering = scattering;
        mat.subsurface_radius = radius;
        mat.type = MATERIAL_SUBSURFACE;
        return mat;
    }
    
    __host__ __device__ static Material createEmissive(const float3& emission) {
        Material mat;
        mat.emission = emission;
        mat.type = MATERIAL_EMISSIVE;
        return mat;
    }
    
    __host__ __device__ static Material createParticipating(int volume_id) {
        Material mat;
        mat.volume_id = volume_id;
        mat.type = MATERIAL_PARTICIPATING;
        return mat;
    }
};

struct Volume {
    float3 sigma_a;       // Absorption coefficient
    float3 sigma_s;       // Scattering coefficient  
    float3 sigma_t;       // Extinction (absorption + scattering)
    float3 le;            // Emission
    float g;              // Phase function parameter
    float density;        // For heterogeneous volumes
    float3 min_bounds;    // Volume bounding box
    float3 max_bounds;
    
    __host__ __device__ Volume() :
        sigma_a(make_float3(0.01f, 0.01f, 0.01f)),
        sigma_s(make_float3(0.1f, 0.1f, 0.1f)),
        le(make_float3(0.0f, 0.0f, 0.0f)),
        g(0.0f),
        density(1.0f),
        min_bounds(make_float3(-1.0f, -1.0f, -1.0f)),
        max_bounds(make_float3(1.0f, 1.0f, 1.0f))
    {
        sigma_t = sigma_a + sigma_s;
    }
    
    __host__ __device__ Volume(const float3& absorption, 
                               const float3& scattering, 
                               float phase_g,
                               const float3& bounds_min,
                               const float3& bounds_max) :
        sigma_a(absorption),
        sigma_s(scattering),
        le(make_float3(0.0f, 0.0f, 0.0f)),
        g(phase_g),
        density(1.0f),
        min_bounds(bounds_min),
        max_bounds(bounds_max)
    {
        sigma_t = sigma_a + sigma_s;
    }
    
    __host__ __device__ bool inside(const float3& p) const {
        return p.x >= min_bounds.x && p.x <= max_bounds.x &&
               p.y >= min_bounds.y && p.y <= max_bounds.y &&
               p.z >= min_bounds.z && p.z <= max_bounds.z;
    }
    
    __host__ __device__ float getDensity(const float3& p) const {
        if (!inside(p)) return 0.0f;
        // For now, uniform density. Can be extended to procedural density
        return density;
    }
    
    __host__ __device__ float3 getSigmaT(const float3& p) const {
        return sigma_t * getDensity(p);
    }
    
    __host__ __device__ float3 getSigmaA(const float3& p) const {
        return sigma_a * getDensity(p);
    }
    
    __host__ __device__ float3 getSigmaS(const float3& p) const {
        return sigma_s * getDensity(p);
    }
};

// Predefined materials for common use cases
namespace Materials {
    __host__ __device__ inline Material whiteLambertian() {
        return Material::createLambertian(make_float3(0.73f, 0.73f, 0.73f));
    }
    
    __host__ __device__ inline Material redLambertian() {
        return Material::createLambertian(make_float3(0.65f, 0.05f, 0.05f));
    }
    
    __host__ __device__ inline Material greenLambertian() {
        return Material::createLambertian(make_float3(0.12f, 0.45f, 0.15f));
    }
    
    __host__ __device__ inline Material skin() {
        return Material::createSubsurface(
            make_float3(0.92f, 0.78f, 0.62f),
            make_float3(0.74f, 0.88f, 1.01f),
            make_float3(2.55f, 3.21f, 3.77f),
            0.1f
        );
    }
    
    __host__ __device__ inline Material marble() {
        return Material::createSubsurface(
            make_float3(0.95f, 0.95f, 0.95f),
            make_float3(0.0021f, 0.0041f, 0.0071f),
            make_float3(2.19f, 2.62f, 3.00f),
            0.2f
        );
    }
    
    __host__ __device__ inline Material wax() {
        return Material::createSubsurface(
            make_float3(0.95f, 0.87f, 0.69f),
            make_float3(0.013f, 0.070f, 0.145f),
            make_float3(2.55f, 3.21f, 3.77f),
            0.15f
        );
    }
    
    __host__ __device__ inline Material glass() {
        return Material::createGlass(1.5f);
    }
    
    __host__ __device__ inline Material aluminum() {
        return Material::createMetal(make_float3(0.91f, 0.92f, 0.92f), 0.1f);
    }
    
    __host__ __device__ inline Material gold() {
        return Material::createMetal(make_float3(1.0f, 0.71f, 0.29f), 0.05f);
    }
    
    __host__ __device__ inline Material light(float intensity = 10.0f) {
        return Material::createEmissive(make_float3(intensity, intensity, intensity));
    }
}

namespace Volumes {
    __host__ __device__ inline Volume fog() {
        return Volume(
            make_float3(0.01f, 0.01f, 0.01f),  // absorption
            make_float3(0.1f, 0.1f, 0.1f),     // scattering
            0.0f,                               // isotropic
            make_float3(-5.0f, -5.0f, -5.0f),
            make_float3(5.0f, 5.0f, 5.0f)
        );
    }
    
    __host__ __device__ inline Volume smoke() {
        return Volume(
            make_float3(0.05f, 0.05f, 0.05f),
            make_float3(0.2f, 0.2f, 0.2f),
            0.8f,  // forward scattering
            make_float3(-2.0f, -2.0f, -2.0f),
            make_float3(2.0f, 2.0f, 2.0f)
        );
    }
} 