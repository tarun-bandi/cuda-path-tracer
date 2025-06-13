#pragma once

#include <vector>
#include <memory>
#include <random>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Vector operations
inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

inline float3 operator*(float s, const float3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

inline float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline float3 operator/(const float3& a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}

inline float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}

inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

inline float3& operator*=(float3& a, float s) {
    a.x *= s; a.y *= s; a.z *= s;
    return a;
}

inline float3& operator/=(float3& a, float s) {
    a.x /= s; a.y /= s; a.z /= s;
    return a;
}

inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

inline float length(const float3& v) {
    return sqrtf(dot(v, v));
}

inline float length_squared(const float3& v) {
    return dot(v, v);
}

inline float3 normalize(const float3& v) {
    float len = length(v);
    return len > 0.0f ? v / len : make_float3(0.0f, 0.0f, 0.0f);
}

inline float3 reflect(const float3& v, const float3& n) {
    return v - 2.0f * dot(v, n) * n;
}

inline float3 refract(const float3& uv, const float3& n, float etai_over_etat) {
    float cos_theta = fminf(dot(-uv, n), 1.0f);
    float3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    float3 r_out_parallel = -sqrtf(fabsf(1.0f - length_squared(r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

inline float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

inline float clamp(float x, float min_val, float max_val) {
    return fmaxf(min_val, fminf(max_val, x));
}

inline float3 clamp(const float3& v, float min_val, float max_val) {
    return make_float3(clamp(v.x, min_val, max_val),
                       clamp(v.y, min_val, max_val),
                       clamp(v.z, min_val, max_val));
}

inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

inline float3 lerp(const float3& a, const float3& b, float t) {
    return a + t * (b - a);
}

// Basic data structures
struct Ray {
    float3 origin;
    float3 direction;
    float tmin;
    float tmax;

    float3 point_at(float t) const {
        return origin + t * direction;
    }
};

struct Intersection {
    float t;
    float3 position;
    float3 normal;
    float3 tangent;
    float3 bitangent;
    float2 uv;
    uint32_t material_id;
    uint32_t volume_id;
    bool is_volume;
};

struct Material {
    float3 albedo;
    float metallic;
    float roughness;
    float ior;
    float transmission;
    float3 emission;
};

struct Volume {
    float3 sigma_a;  // Absorption coefficient
    float3 sigma_s;  // Scattering coefficient
    float3 sigma_t;  // Total extinction coefficient
    float g;         // Phase function parameter (Henyey-Greenstein)
    float density;   // Volume density
};

// Constants
constexpr float PI = 3.14159265358979323846f;
constexpr float EPSILON = 1e-5f;
constexpr int MAX_DEPTH = 8;
constexpr int SAMPLES_PER_PIXEL = 1024;

// Random number generation
class Random {
public:
    Random() : gen(std::random_device{}()) {}
    
    float random_float() {
        return std::uniform_real_distribution<float>(0.0f, 1.0f)(gen);
    }
    
    float3 random_unit_vector() {
        float z = random_float() * 2.0f - 1.0f;
        float r = sqrtf(1.0f - z * z);
        float phi = random_float() * 2.0f * PI;
        return make_float3(r * cosf(phi), r * sinf(phi), z);
    }
    
    float3 random_in_unit_sphere() {
        while (true) {
            float3 p = make_float3(
                random_float() * 2.0f - 1.0f,
                random_float() * 2.0f - 1.0f,
                random_float() * 2.0f - 1.0f
            );
            if (length_squared(p) < 1.0f) return p;
        }
    }
    
private:
    std::mt19937 gen;
}; 