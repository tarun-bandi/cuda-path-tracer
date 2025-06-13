#include "../common.h"
#include "../core/scene.h"
#include "../core/camera.h"
#include "../acceleration/bvh.h"
#include "../acceleration/volume_grid.h"
#include <thread>
#include <vector>
#include <atomic>

// Thread pool for parallel rendering
class ThreadPool {
public:
    ThreadPool(size_t num_threads) : stop(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { 
                            return stop || !tasks.empty(); 
                        });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            worker.join();
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// Path tracing implementation
void trace_path(const Ray& ray, const Scene& scene, const BVH& bvh, 
                const VolumeGrid& volume_grid, Random& rng, float3& color, 
                int depth = 0) {
    if (depth >= MAX_DEPTH) return;

    // Check for volume intersection
    VolumeIntersection vol_isect;
    if (volume_grid.intersect(ray, vol_isect)) {
        // Handle volume scattering
        float t = vol_isect.t_enter + rng.random_float() * (vol_isect.t_exit - vol_isect.t_enter);
        float3 p = ray.point_at(t);
        
        // Sample phase function
        float3 scatter_dir = rng.random_unit_vector();
        Ray scatter_ray(p, scatter_dir);
        
        float3 scatter_color;
        trace_path(scatter_ray, scene, bvh, volume_grid, rng, scatter_color, depth + 1);
        
        color += vol_isect.volume.sigma_s * scatter_color;
        return;
    }

    // Check for surface intersection
    HitRecord hit;
    if (bvh.intersect(ray, hit)) {
        const Material& material = scene.materials[hit.material_id];
        
        // Handle emission
        color += material.emission;
        
        // Handle scattering
        if (material.transmission > 0.0f) {
            // Handle transmission
            float3 scatter_dir = rng.random_unit_vector();
            Ray scatter_ray(hit.point, scatter_dir);
            
            float3 scatter_color;
            trace_path(scatter_ray, scene, bvh, volume_grid, rng, scatter_color, depth + 1);
            
            color += material.albedo * scatter_color;
        } else {
            // Handle reflection
            float3 scatter_dir = rng.random_unit_vector();
            Ray scatter_ray(hit.point, scatter_dir);
            
            float3 scatter_color;
            trace_path(scatter_ray, scene, bvh, volume_grid, rng, scatter_color, depth + 1);
            
            color += material.albedo * scatter_color;
        }
    } else {
        // Handle environment map
        color += scene.environment_map.sample(ray.direction);
    }
}

// Main rendering function
void render(float3* output, const Camera* camera, const Scene* scene,
           const BVH* bvh, const VolumeGrid* volume_grid,
           int width, int height, int samples_per_pixel) {
    // Create thread pool
    size_t num_threads = std::thread::hardware_concurrency();
    ThreadPool pool(num_threads);
    
    // Create progress counter
    std::atomic<int> pixels_rendered(0);
    int total_pixels = width * height;
    
    // Render each pixel
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            pool.enqueue([=, &pixels_rendered]() {
                float3 pixel_color = make_float3(0.0f, 0.0f, 0.0f);
                Random rng;
                
                // Sample pixel
                for (int s = 0; s < samples_per_pixel; ++s) {
                    float u = (i + rng.random_float()) / width;
                    float v = (j + rng.random_float()) / height;
                    Ray ray = camera->get_ray(u, v);
                    
                    float3 sample_color;
                    trace_path(ray, *scene, *bvh, *volume_grid, rng, sample_color);
                    pixel_color += sample_color;
                }
                
                // Average samples
                pixel_color = pixel_color / float(samples_per_pixel);
                
                // Store result
                output[j * width + i] = pixel_color;
                
                // Update progress
                pixels_rendered++;
                if (pixels_rendered % 100 == 0) {
                    float progress = float(pixels_rendered) / total_pixels * 100.0f;
                    printf("\rRendering progress: %.1f%%", progress);
                    fflush(stdout);
                }
            });
        }
    }
    
    // Wait for all threads to complete
    pool.~ThreadPool();
    printf("\nRendering complete!\n");
} 