#pragma once

#include "../common.h"
#include <glm/gtc/matrix_transform.hpp>

class Camera {
public:
    // FPS camera parameters
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 world_up;
    float yaw;
    float pitch;
    float fov;
    float aspect_ratio;
    float aperture;
    float focus_distance;

    // Ray tracing parameters
    glm::vec3 origin;
    glm::vec3 lower_left_corner;
    glm::vec3 horizontal;
    glm::vec3 vertical;
    glm::vec3 u, v, w;
    float lens_radius;

    Camera(
        const glm::vec3& position = glm::vec3(0.0f, 0.0f, 0.0f),
        const glm::vec3& target = glm::vec3(0.0f, 0.0f, -1.0f),
        const glm::vec3& up = glm::vec3(0.0f, 1.0f, 0.0f),
        float fov = 45.0f,
        float aspect_ratio = 16.0f/9.0f,
        float aperture = 0.0f,
        float focus_distance = 1.0f
    ) : position(position), world_up(up), fov(fov), aspect_ratio(aspect_ratio),
        aperture(aperture), focus_distance(focus_distance), yaw(-90.0f), pitch(0.0f) {
        update_camera_vectors();
        update_ray_tracing_parameters();
    }

    void update_camera_vectors() {
        glm::vec3 front;
        front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(pitch));
        front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        this->front = glm::normalize(front);
        this->right = glm::normalize(glm::cross(this->front, world_up));
        this->up = glm::normalize(glm::cross(this->right, this->front));
    }

    void update_ray_tracing_parameters() {
        float theta = glm::radians(fov);
        float h = tan(theta / 2.0f);
        float viewport_height = 2.0f * h;
        float viewport_width = aspect_ratio * viewport_height;

        w = glm::normalize(position - (position + front));
        u = glm::normalize(glm::cross(up, w));
        v = glm::cross(w, u);

        origin = position;
        horizontal = focus_distance * viewport_width * u;
        vertical = focus_distance * viewport_height * v;
        lower_left_corner = origin - horizontal/2.0f - vertical/2.0f - focus_distance * w;

        lens_radius = aperture / 2.0f;
    }

    __device__ Ray get_ray(float s, float t, curandState* state) const {
        glm::vec3 rd = lens_radius * random_in_unit_disk(state);
        glm::vec3 offset = u * rd.x + v * rd.y;

        return Ray{
            origin + offset,
            lower_left_corner + s * horizontal + t * vertical - origin - offset,
            0.0f,
            INFINITY
        };
    }

private:
    __device__ glm::vec3 random_in_unit_disk(curandState* state) const {
        while (true) {
            glm::vec3 p = glm::vec3(
                random_float(state) * 2.0f - 1.0f,
                random_float(state) * 2.0f - 1.0f,
                0.0f
            );
            if (glm::dot(p, p) < 1.0f) return p;
        }
    }
}; 