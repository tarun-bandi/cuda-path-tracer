#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Vector operations
__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float3 operator*(float s, const float3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ inline float3 operator/(const float3& a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}

__host__ __device__ inline float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__host__ __device__ inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

__host__ __device__ inline float3& operator*=(float3& a, float s) {
    a.x *= s; a.y *= s; a.z *= s;
    return a;
}

__host__ __device__ inline float3& operator/=(float3& a, float s) {
    a.x /= s; a.y /= s; a.z /= s;
    return a;
}

__host__ __device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

__host__ __device__ inline float length(const float3& v) {
    return sqrtf(dot(v, v));
}

__host__ __device__ inline float length_squared(const float3& v) {
    return dot(v, v);
}

__host__ __device__ inline float3 normalize(const float3& v) {
    float len = length(v);
    return len > 0.0f ? v / len : make_float3(0.0f);
}

__host__ __device__ inline float3 reflect(const float3& v, const float3& n) {
    return v - 2.0f * dot(v, n) * n;
}

__host__ __device__ inline float3 refract(const float3& uv, const float3& n, float etai_over_etat) {
    float cos_theta = fminf(dot(-uv, n), 1.0f);
    float3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    float3 r_out_parallel = -sqrtf(fabsf(1.0f - length_squared(r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

__host__ __device__ inline float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

__host__ __device__ inline float clamp(float x, float min_val, float max_val) {
    return fmaxf(min_val, fminf(max_val, x));
}

__host__ __device__ inline float3 clamp(const float3& v, float min_val, float max_val) {
    return make_float3(clamp(v.x, min_val, max_val),
                       clamp(v.y, min_val, max_val),
                       clamp(v.z, min_val, max_val));
}

__host__ __device__ inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__host__ __device__ inline float3 lerp(const float3& a, const float3& b, float t) {
    return a + t * (b - a);
}

// Basic data structures
struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
    float tmin;
    float tmax;

    __device__ glm::vec3 point_at(float t) const {
        return origin + t * direction;
    }
};

struct Intersection {
    float t;
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 tangent;
    glm::vec3 bitangent;
    glm::vec2 uv;
    uint32_t material_id;
    uint32_t volume_id;
    bool is_volume;
};

struct Material {
    glm::vec3 albedo;
    float metallic;
    float roughness;
    float ior;
    float transmission;
    glm::vec3 emission;
};

struct Volume {
    glm::vec3 sigma_a;  // Absorption coefficient
    glm::vec3 sigma_s;  // Scattering coefficient
    glm::vec3 sigma_t;  // Total extinction coefficient
    float g;            // Phase function parameter (Henyey-Greenstein)
    float density;      // Volume density
};

// Constants
constexpr float PI = 3.14159265358979323846f;
constexpr float EPSILON = 1e-5f;
constexpr int MAX_DEPTH = 8;
constexpr int SAMPLES_PER_PIXEL = 1024;

// Utility functions
__device__ inline float random_float(curandState* state) {
    return curand_uniform(state);
}

__device__ inline glm::vec3 random_unit_vector(curandState* state) {
    float z = random_float(state) * 2.0f - 1.0f;
    float r = sqrtf(1.0f - z * z);
    float phi = random_float(state) * 2.0f * PI;
    return glm::vec3(r * cosf(phi), r * sinf(phi), z);
}

__device__ inline glm::vec3 random_in_unit_sphere(curandState* state) {
    while (true) {
        glm::vec3 p = glm::vec3(
            random_float(state) * 2.0f - 1.0f,
            random_float(state) * 2.0f - 1.0f,
            random_float(state) * 2.0f - 1.0f
        );
        if (glm::dot(p, p) < 1.0f) return p;
    }
}

__device__ inline bool near_zero(const glm::vec3& v) {
    return (fabsf(v.x) < EPSILON) && (fabsf(v.y) < EPSILON) && (fabsf(v.z) < EPSILON);
}

__device__ inline glm::vec3 reflect(const glm::vec3& v, const glm::vec3& n) {
    return v - 2.0f * glm::dot(v, n) * n;
}

__device__ inline glm::vec3 refract(const glm::vec3& uv, const glm::vec3& n, float etai_over_etat) {
    float cos_theta = fminf(glm::dot(-uv, n), 1.0f);
    glm::vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    glm::vec3 r_out_parallel = -sqrtf(fabsf(1.0f - glm::dot(r_out_perp, r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
} 