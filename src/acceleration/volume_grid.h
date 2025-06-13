#pragma once

#include "../common.h"
#include "../core/volume.h"

struct VolumeIntersection {
    float t_enter;
    float t_exit;
    Volume volume;
};

class VolumeGrid {
public:
    VolumeGrid() : resolution(0), bounds_min(0), bounds_max(0), data(nullptr) {}
    
    void init(const float3& min_bounds, const float3& max_bounds, int3 res) {
        resolution = res;
        bounds_min = min_bounds;
        bounds_max = max_bounds;
        cell_size = (bounds_max - bounds_min) / make_float3(float(resolution.x), float(resolution.y), float(resolution.z));
        
        // Allocate grid data
        size_t num_cells = resolution.x * resolution.y * resolution.z;
        cudaMalloc(&data, num_cells * sizeof(Volume));
    }
    
    void set_volume(int x, int y, int z, const Volume& volume) {
        if (x < 0 || x >= resolution.x || y < 0 || y >= resolution.y || z < 0 || z >= resolution.z) {
            return;
        }
        int index = z * (resolution.x * resolution.y) + y * resolution.x + x;
        data[index] = volume;
    }
    
    __device__ bool intersect(const Ray& ray, VolumeIntersection& isect) const {
        // Compute ray-box intersection
        float3 inv_dir = make_float3(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);
        float3 t_min = (bounds_min - ray.origin) * inv_dir;
        float3 t_max = (bounds_max - ray.origin) * inv_dir;
        
        float3 t1 = make_float3(fminf(t_min.x, t_max.x), fminf(t_min.y, t_max.y), fminf(t_min.z, t_max.z));
        float3 t2 = make_float3(fmaxf(t_min.x, t_max.x), fmaxf(t_min.y, t_max.y), fmaxf(t_min.z, t_max.z));
        
        float t_enter = fmaxf(fmaxf(t1.x, t1.y), t1.z);
        float t_exit = fminf(fminf(t2.x, t2.y), t2.z);
        
        if (t_exit < t_enter || t_exit < 0.0f) {
            return false;
        }
        
        t_enter = fmaxf(t_enter, ray.tmin);
        t_exit = fminf(t_exit, ray.tmax);
        
        if (t_enter >= t_exit) {
            return false;
        }
        
        // Get cell coordinates
        float3 p_enter = ray.point_at(t_enter);
        float3 p_exit = ray.point_at(t_exit);
        
        int3 cell_min = get_cell_coords(p_enter);
        int3 cell_max = get_cell_coords(p_exit);
        
        // Clamp to grid bounds
        cell_min = make_int3(
            fmaxf(0, fminf(resolution.x - 1, cell_min.x)),
            fmaxf(0, fminf(resolution.y - 1, cell_min.y)),
            fmaxf(0, fminf(resolution.z - 1, cell_min.z))
        );
        
        cell_max = make_int3(
            fmaxf(0, fminf(resolution.x - 1, cell_max.x)),
            fmaxf(0, fminf(resolution.y - 1, cell_max.y)),
            fmaxf(0, fminf(resolution.z - 1, cell_max.z))
        );
        
        // Find first non-empty cell
        for (int z = cell_min.z; z <= cell_max.z; ++z) {
            for (int y = cell_min.y; y <= cell_max.y; ++y) {
                for (int x = cell_min.x; x <= cell_max.x; ++x) {
                    int index = z * (resolution.x * resolution.y) + y * resolution.x + x;
                    if (data[index].density > 0.0f) {
                        isect.t_enter = t_enter;
                        isect.t_exit = t_exit;
                        isect.volume = data[index];
                        return true;
                    }
                }
            }
        }
        
        return false;
    }
    
    void cleanup() {
        if (data) {
            cudaFree(data);
            data = nullptr;
        }
    }
    
private:
    int3 resolution;
    float3 bounds_min;
    float3 bounds_max;
    float3 cell_size;
    Volume* data;
    
    __device__ int3 get_cell_coords(const float3& p) const {
        float3 local_p = p - bounds_min;
        return make_int3(
            (int)(local_p.x / cell_size.x),
            (int)(local_p.y / cell_size.y),
            (int)(local_p.z / cell_size.z)
        );
    }
}; 