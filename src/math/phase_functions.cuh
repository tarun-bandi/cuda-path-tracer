#pragma once

#include "../common.h"

__device__ inline float3 henyey_greenstein(
    const float3& wo,
    const float3& wi,
    float g
) {
    float cos_theta = dot(wo, wi);
    float denom = 1.0f + g * g - 2.0f * g * cos_theta;
    float phase = (1.0f - g * g) / (denom * sqrtf(denom));
    return make_float3(phase);
}

__device__ inline float3 sample_henyey_greenstein(
    const float3& wo,
    float3& wi,
    float g,
    curandState* state,
    float* pdf
) {
    float cos_theta;
    if (fabsf(g) < 1e-3f) {
        // Isotropic scattering
        cos_theta = 1.0f - 2.0f * random_float(state);
    } else {
        float sqr_term = (1.0f - g * g) / (1.0f - g + 2.0f * g * random_float(state));
        cos_theta = (1.0f + g * g - sqr_term * sqr_term) / (2.0f * g);
    }

    // Compute direction
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = 2.0f * PI * random_float(state);

    // Create coordinate system
    float3 u, v;
    if (fabsf(wo.x) > fabsf(wo.y)) {
        u = make_float3(-wo.z, 0.0f, wo.x) / sqrtf(wo.x * wo.x + wo.z * wo.z);
    } else {
        u = make_float3(0.0f, wo.z, -wo.y) / sqrtf(wo.y * wo.y + wo.z * wo.z);
    }
    v = cross(wo, u);

    // Compute scattered direction
    wi = u * (sin_theta * cosf(phi)) +
         v * (sin_theta * sinf(phi)) +
         wo * cos_theta;

    // Compute PDF
    *pdf = henyey_greenstein(wo, wi, g).x;

    return henyey_greenstein(wo, wi, g);
}

__device__ inline float3 sample_light(
    const float3& p,
    float3& wi,
    curandState* state,
    float* pdf
) {
    // TODO: Implement light sampling
    // This is a placeholder that returns zero radiance
    *pdf = 0.0f;
    return make_float3(0.0f);
} 