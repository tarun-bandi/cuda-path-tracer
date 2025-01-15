#pragma once

#include "../common.h"
#include "materials.h"

struct Ray {
    float3 origin;
    float3 direction;
    float tmin, tmax;
    float3 throughput;
    int depth;
    
    __host__ __device__ Ray() : 
        origin(make_float3(0.0f)), 
        direction(make_float3(0.0f, 0.0f, 1.0f)),
        tmin(0.001f), 
        tmax(1e30f),
        throughput(make_float3(1.0f)),
        depth(0) 
    {}
    
    __host__ __device__ Ray(const float3& o, const float3& d, float t_min = 0.001f, float t_max = 1e30f) :
        origin(o), direction(d), tmin(t_min), tmax(t_max), throughput(make_float3(1.0f)), depth(0) {}
    
    __host__ __device__ float3 at(float t) const {
        return origin + t * direction;
    }
};

struct HitRecord {
    float3 point;
    float3 normal;
    float t;
    bool front_face;
    int material_id;
    float u, v;  // Texture coordinates
    
    __host__ __device__ void setFaceNormal(const Ray& r, const float3& outward_normal) {
        front_face = dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

struct VolumeRecord {
    float3 point;
    float t;
    int volume_id;
    float3 sigma_t;
    float3 sigma_s;
    float g;  // Phase function parameter
    bool valid;
    
    __host__ __device__ VolumeRecord() : valid(false) {}
};

// Basic sphere geometry
struct Sphere {
    float3 center;
    float radius;
    int material_id;
    
    __host__ __device__ Sphere(const float3& c, float r, int mat_id) :
        center(c), radius(r), material_id(mat_id) {}
    
    __host__ __device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        float3 oc = r.origin - center;
        float a = length_squared(r.direction);
        float half_b = dot(oc, r.direction);
        float c = length_squared(oc) - radius * radius;
        float discriminant = half_b * half_b - a * c;
        
        if (discriminant < 0) return false;
        
        float sqrt_d = sqrtf(discriminant);
        float root = (-half_b - sqrt_d) / a;
        
        if (root < t_min || t_max < root) {
            root = (-half_b + sqrt_d) / a;
            if (root < t_min || t_max < root) {
                return false;
            }
        }
        
        rec.t = root;
        rec.point = r.at(rec.t);
        float3 outward_normal = (rec.point - center) / radius;
        rec.setFaceNormal(r, outward_normal);
        rec.material_id = material_id;
        
        // Compute UV coordinates for sphere
        float theta = acosf(-outward_normal.y);
        float phi = atan2f(-outward_normal.z, outward_normal.x) + M_PI;
        rec.u = phi / (2.0f * M_PI);
        rec.v = theta / M_PI;
        
        return true;
    }
};

// Axis-aligned box
struct Box {
    float3 min_corner;
    float3 max_corner;
    int material_id;
    
    __host__ __device__ Box(const float3& min_c, const float3& max_c, int mat_id) :
        min_corner(min_c), max_corner(max_c), material_id(mat_id) {}
    
    __host__ __device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        float3 inv_d = make_float3(1.0f / r.direction.x, 1.0f / r.direction.y, 1.0f / r.direction.z);
        
        float3 t0 = (min_corner - r.origin) * inv_d;
        float3 t1 = (max_corner - r.origin) * inv_d;
        
        float3 t_near = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
        float3 t_far = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));
        
        float t_entry = fmaxf(fmaxf(t_near.x, t_near.y), t_near.z);
        float t_exit = fminf(fminf(t_far.x, t_far.y), t_far.z);
        
        if (t_entry > t_exit || t_exit < t_min || t_entry > t_max) {
            return false;
        }
        
        float t_hit = (t_entry > t_min) ? t_entry : t_exit;
        if (t_hit < t_min || t_hit > t_max) {
            return false;
        }
        
        rec.t = t_hit;
        rec.point = r.at(rec.t);
        rec.material_id = material_id;
        
        // Compute normal based on which face was hit
        float3 center = (min_corner + max_corner) * 0.5f;
        float3 local_point = rec.point - center;
        float3 d = (max_corner - min_corner) * 0.5f;
        
        float bias = 1.000001f;
        float3 normal = make_float3(
            (fabsf(local_point.x) > d.x * bias) ? ((local_point.x > 0) ? 1.0f : -1.0f) : 0.0f,
            (fabsf(local_point.y) > d.y * bias) ? ((local_point.y > 0) ? 1.0f : -1.0f) : 0.0f,
            (fabsf(local_point.z) > d.z * bias) ? ((local_point.z > 0) ? 1.0f : -1.0f) : 0.0f
        );
        
        rec.setFaceNormal(r, normal);
        
        return true;
    }
};

// Plane geometry
struct Plane {
    float3 point;
    float3 normal;
    int material_id;
    
    __host__ __device__ Plane(const float3& p, const float3& n, int mat_id) :
        point(p), normal(normalize(n)), material_id(mat_id) {}
    
    __host__ __device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        float denom = dot(normal, r.direction);
        if (fabsf(denom) < 1e-6f) return false;  // Ray parallel to plane
        
        float t = dot(point - r.origin, normal) / denom;
        if (t < t_min || t > t_max) return false;
        
        rec.t = t;
        rec.point = r.at(rec.t);
        rec.setFaceNormal(r, normal);
        rec.material_id = material_id;
        
        return true;
    }
};

// Light source
struct Light {
    float3 position;
    float3 emission;
    float radius;
    int type;  // 0 = point, 1 = sphere, 2 = area
    
    __host__ __device__ Light(const float3& pos, const float3& emit, float r = 0.1f, int t = 1) :
        position(pos), emission(emit), radius(r), type(t) {}
};

// Scene structure containing all geometry and materials
class Scene {
public:
    std::vector<Sphere> spheres;
    std::vector<Box> boxes;
    std::vector<Plane> planes;
    std::vector<Material> materials;
    std::vector<Volume> volumes;
    std::vector<Light> lights;
    
    // Device copies
    Sphere* d_spheres;
    Box* d_boxes;
    Plane* d_planes;
    Material* d_materials;
    Volume* d_volumes;
    Light* d_lights;
    
    int num_spheres, num_boxes, num_planes;
    int num_materials, num_volumes, num_lights;
    
    Scene() : d_spheres(nullptr), d_boxes(nullptr), d_planes(nullptr),
              d_materials(nullptr), d_volumes(nullptr), d_lights(nullptr),
              num_spheres(0), num_boxes(0), num_planes(0),
              num_materials(0), num_volumes(0), num_lights(0) {}
    
    ~Scene() {
        cleanup();
    }
    
    void addSphere(const float3& center, float radius, int material_id) {
        spheres.emplace_back(center, radius, material_id);
    }
    
    void addBox(const float3& min_corner, const float3& max_corner, int material_id) {
        boxes.emplace_back(min_corner, max_corner, material_id);
    }
    
    void addPlane(const float3& point, const float3& normal, int material_id) {
        planes.emplace_back(point, normal, material_id);
    }
    
    int addMaterial(const Material& material) {
        materials.push_back(material);
        return materials.size() - 1;
    }
    
    int addVolume(const Volume& volume) {
        volumes.push_back(volume);
        return volumes.size() - 1;
    }
    
    void addLight(const float3& position, const float3& emission, float radius = 0.1f) {
        lights.emplace_back(position, emission, radius);
    }
    
    void copyToDevice() {
        num_spheres = spheres.size();
        num_boxes = boxes.size();
        num_planes = planes.size();
        num_materials = materials.size();
        num_volumes = volumes.size();
        num_lights = lights.size();
        
        // Allocate and copy spheres
        if (num_spheres > 0) {
            CUDA_CHECK(cudaMalloc(&d_spheres, num_spheres * sizeof(Sphere)));
            CUDA_CHECK(cudaMemcpy(d_spheres, spheres.data(), num_spheres * sizeof(Sphere), cudaMemcpyHostToDevice));
        }
        
        // Allocate and copy boxes
        if (num_boxes > 0) {
            CUDA_CHECK(cudaMalloc(&d_boxes, num_boxes * sizeof(Box)));
            CUDA_CHECK(cudaMemcpy(d_boxes, boxes.data(), num_boxes * sizeof(Box), cudaMemcpyHostToDevice));
        }
        
        // Allocate and copy planes
        if (num_planes > 0) {
            CUDA_CHECK(cudaMalloc(&d_planes, num_planes * sizeof(Plane)));
            CUDA_CHECK(cudaMemcpy(d_planes, planes.data(), num_planes * sizeof(Plane), cudaMemcpyHostToDevice));
        }
        
        // Allocate and copy materials
        if (num_materials > 0) {
            CUDA_CHECK(cudaMalloc(&d_materials, num_materials * sizeof(Material)));
            CUDA_CHECK(cudaMemcpy(d_materials, materials.data(), num_materials * sizeof(Material), cudaMemcpyHostToDevice));
        }
        
        // Allocate and copy volumes
        if (num_volumes > 0) {
            CUDA_CHECK(cudaMalloc(&d_volumes, num_volumes * sizeof(Volume)));
            CUDA_CHECK(cudaMemcpy(d_volumes, volumes.data(), num_volumes * sizeof(Volume), cudaMemcpyHostToDevice));
        }
        
        // Allocate and copy lights
        if (num_lights > 0) {
            CUDA_CHECK(cudaMalloc(&d_lights, num_lights * sizeof(Light)));
            CUDA_CHECK(cudaMemcpy(d_lights, lights.data(), num_lights * sizeof(Light), cudaMemcpyHostToDevice));
        }
    }
    
    void cleanup() {
        if (d_spheres) { cudaFree(d_spheres); d_spheres = nullptr; }
        if (d_boxes) { cudaFree(d_boxes); d_boxes = nullptr; }
        if (d_planes) { cudaFree(d_planes); d_planes = nullptr; }
        if (d_materials) { cudaFree(d_materials); d_materials = nullptr; }
        if (d_volumes) { cudaFree(d_volumes); d_volumes = nullptr; }
        if (d_lights) { cudaFree(d_lights); d_lights = nullptr; }
    }
}; 