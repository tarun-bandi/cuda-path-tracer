#pragma once

#include "../common.h"
#include "../core/scene.h"

struct BVHNode {
    AABB bounds;
    int first_primitive;
    int primitive_count;
    int right_child;
    bool is_leaf;
};

class BVH {
public:
    std::vector<BVHNode> nodes;
    std::vector<int> primitive_indices;

    void build(const Scene& scene) {
        // Clear existing data
        nodes.clear();
        primitive_indices.clear();

        // Create leaf node for each primitive
        std::vector<AABB> primitive_bounds;
        primitive_bounds.reserve(scene.primitives.size());
        primitive_indices.reserve(scene.primitives.size());

        for (size_t i = 0; i < scene.primitives.size(); ++i) {
            primitive_bounds.push_back(scene.primitives[i].bounds());
            primitive_indices.push_back(i);
        }

        // Recursively build BVH
        BVHNode root;
        root.first_primitive = 0;
        root.primitive_count = primitive_indices.size();
        root.is_leaf = false;
        nodes.push_back(root);

        build_node(0, primitive_bounds, 0, primitive_indices.size());
    }

    __device__ bool intersect(const Ray& ray, Intersection& isect) const {
        float t_min = ray.tmin;
        float t_max = ray.tmax;
        int node_idx = 0;
        bool hit = false;

        while (node_idx >= 0) {
            const BVHNode& node = nodes[node_idx];

            if (!node.bounds.intersect(ray, t_min, t_max)) {
                node_idx = -1;
                continue;
            }

            if (node.is_leaf) {
                // Check all primitives in leaf node
                for (int i = 0; i < node.primitive_count; ++i) {
                    int prim_idx = primitive_indices[node.first_primitive + i];
                    if (scene.primitives[prim_idx].intersect(ray, isect)) {
                        hit = true;
                        t_max = isect.t;
                    }
                }
                node_idx = -1;
            } else {
                // Traverse to child nodes
                node_idx = node.right_child;
            }
        }

        return hit;
    }

private:
    void build_node(int node_idx, const std::vector<AABB>& primitive_bounds, int start, int end) {
        BVHNode& node = nodes[node_idx];
        node.bounds = compute_bounds(primitive_bounds, start, end);

        if (end - start <= 2) {
            // Create leaf node
            node.is_leaf = true;
            return;
        }

        // Find split axis and position
        int split_axis = find_split_axis(primitive_bounds, start, end);
        int split_idx = find_split_index(primitive_bounds, start, end, split_axis);

        // Create child nodes
        BVHNode left_child;
        left_child.first_primitive = start;
        left_child.primitive_count = split_idx - start;
        left_child.is_leaf = false;
        nodes.push_back(left_child);

        BVHNode right_child;
        right_child.first_primitive = split_idx;
        right_child.primitive_count = end - split_idx;
        right_child.is_leaf = false;
        nodes.push_back(right_child);

        node.right_child = nodes.size() - 2;

        // Recursively build child nodes
        build_node(nodes.size() - 2, primitive_bounds, start, split_idx);
        build_node(nodes.size() - 1, primitive_bounds, split_idx, end);
    }

    AABB compute_bounds(const std::vector<AABB>& primitive_bounds, int start, int end) {
        AABB bounds = primitive_bounds[start];
        for (int i = start + 1; i < end; ++i) {
            bounds = bounds.union_with(primitive_bounds[i]);
        }
        return bounds;
    }

    int find_split_axis(const std::vector<AABB>& primitive_bounds, int start, int end) {
        AABB bounds = compute_bounds(primitive_bounds, start, end);
        float3 extent = bounds.max - bounds.min;
        if (extent.x > extent.y && extent.x > extent.z) return 0;
        if (extent.y > extent.z) return 1;
        return 2;
    }

    int find_split_index(const std::vector<AABB>& primitive_bounds, int start, int end, int axis) {
        // Simple median split
        return start + (end - start) / 2;
    }
};
