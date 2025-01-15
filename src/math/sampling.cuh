#pragma once

#include "../common.h"

// Random number generation utilities
__device__ inline float random_float(curandState* state) {
    return curand_uniform(state);
}

__device__ inline float random_float(curandState* state, float min, float max) {
    return min + (max - min) * random_float(state);
}

__device__ inline float2 random_float2(curandState* state) {
    return make_float2(random_float(state), random_float(state));
}

// Sampling functions for path tracing

// Sample a point on the unit sphere
__device__ inline float3 random_in_unit_sphere(curandState* state) {
    float3 p;
    do {
        p = 2.0f * make_float3(random_float(state), random_float(state), random_float(state)) - make_float3(1.0f, 1.0f, 1.0f);
    } while (length_squared(p) >= 1.0f);
    return p;
}

// Sample a point on the unit hemisphere
__device__ inline float3 random_in_unit_hemisphere(curandState* state, const float3& normal) {
    float3 in_unit_sphere = random_in_unit_sphere(state);
    if (dot(in_unit_sphere, normal) > 0.0f) {
        return in_unit_sphere;
    } else {
        return -in_unit_sphere;
    }
}

// Sample a cosine-weighted hemisphere direction
__device__ inline float3 sampleCosineHemisphere(float2 xi) {
    float cos_theta = sqrtf(xi.x);
    float sin_theta = sqrtf(1.0f - xi.x);
    float phi = 2.0f * M_PI * xi.y;
    
    return make_float3(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta
    );
}

// Sample a uniform hemisphere direction
__device__ inline float3 sampleUniformHemisphere(float2 xi) {
    float cos_theta = xi.x;
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = 2.0f * M_PI * xi.y;
    
    return make_float3(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta
    );
}

// Sample Henyey-Greenstein phase function
__device__ inline float3 sampleHenyeyGreenstein(float g, float2 xi) {
    float cos_theta;
    
    if (fabsf(g) < 1e-3f) {
        // Isotropic case
        cos_theta = 1.0f - 2.0f * xi.x;
    } else {
        float sqr_term = (1.0f - g * g) / (1.0f - g + 2.0f * g * xi.x);
        cos_theta = (1.0f + g * g - sqr_term * sqr_term) / (2.0f * g);
    }
    
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = 2.0f * M_PI * xi.y;
    
    return make_float3(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta
    );
}

// Sample a point on a disk
__device__ inline float2 sampleDisk(float2 xi) {
    float r = sqrtf(xi.x);
    float theta = 2.0f * M_PI * xi.y;
    return make_float2(r * cos(theta), r * sin(theta));
}

// Sample exponential distribution for distance sampling
__device__ inline float sampleExponential(float xi, float lambda) {
    return -logf(1.0f - xi) / lambda;
}

// Sample a point on a triangle
__device__ inline float3 sampleTriangle(float2 xi) {
    float sqrt_xi1 = sqrtf(xi.x);
    float u = 1.0f - sqrt_xi1;
    float v = xi.y * sqrt_xi1;
    float w = 1.0f - u - v;
    return make_float3(u, v, w);
}

// Build orthonormal basis from a normal vector
__device__ inline void buildOrthonormalBasis(const float3& n, float3& tangent, float3& bitangent) {
    if (fabsf(n.x) > 0.9f) {
        tangent = make_float3(0.0f, 1.0f, 0.0f);
    } else {
        tangent = make_float3(1.0f, 0.0f, 0.0f);
    }
    
    tangent = normalize(cross(n, tangent));
    bitangent = cross(n, tangent);
}

// Transform direction from local to world coordinates
__device__ inline float3 localToWorld(const float3& local, const float3& normal) {
    float3 tangent, bitangent;
    buildOrthonormalBasis(normal, tangent, bitangent);
    
    return local.x * tangent + local.y * bitangent + local.z * normal;
}

// Transform direction from world to local coordinates
__device__ inline float3 worldToLocal(const float3& world, const float3& normal) {
    float3 tangent, bitangent;
    buildOrthonormalBasis(normal, tangent, bitangent);
    
    return make_float3(
        dot(world, tangent),
        dot(world, bitangent),
        dot(world, normal)
    );
}

// PDF functions

__device__ inline float cosineHemispherePdf(float cos_theta) {
    return cos_theta / M_PI;
}

__device__ inline float uniformHemispherePdf() {
    return 1.0f / (2.0f * M_PI);
}

__device__ inline float henyeyGreensteinPdf(float g, float cos_theta) {
    float denom = 1.0f + g * g + 2.0f * g * cos_theta;
    return (1.0f - g * g) / (4.0f * M_PI * powf(denom, 1.5f));
}

// Multiple importance sampling utilities

__device__ inline float powerHeuristic(float n_f, float pdf_f, float n_g, float pdf_g) {
    float f = n_f * pdf_f;
    float g = n_g * pdf_g;
    return (f * f) / (f * f + g * g);
}

__device__ inline float balanceHeuristic(float n_f, float pdf_f, float n_g, float pdf_g) {
    return (n_f * pdf_f) / (n_f * pdf_f + n_g * pdf_g);
}

// Low discrepancy sequences (simplified Sobol)
__device__ inline float sobol(int i, int dim) {
    // Simplified Sobol sequence - for production use proper implementation
    int x = i;
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    
    // Different scrambling for different dimensions
    for (int d = 0; d < dim; d++) {
        x = ((x >> 1) ^ (-(x & 1) & 0xd0000001));
    }
    
    return (x & 0xffffff) / 16777216.0f;
}

// Stratified sampling
__device__ inline float2 stratifiedSample(int i, int j, int nx, int ny, curandState* state) {
    float u = (i + random_float(state)) / nx;
    float v = (j + random_float(state)) / ny;
    return make_float2(u, v);
}

// Sample a light source
__device__ inline float3 sampleLight(curandState* state, const float3& position, 
                                    const float3& light_pos, float light_radius) {
    // Sample point on sphere light
    float3 to_light = light_pos - position;
    float distance = length(to_light);
    float3 direction = to_light / distance;
    
    if (distance < light_radius) {
        // Inside the light
        return random_in_unit_sphere(state);
    }
    
    // Sample cone towards light
    float cos_theta_max = sqrtf(1.0f - light_radius * light_radius / (distance * distance));
    float cos_theta = 1.0f - random_float(state) * (1.0f - cos_theta_max);
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
    float phi = 2.0f * M_PI * random_float(state);
    
    float3 local_dir = make_float3(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta
    );
    
    return localToWorld(local_dir, direction);
} 