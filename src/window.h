#pragma once

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "core/camera.h"

// Global variables
extern GLFWwindow* window;
extern Camera camera;
extern bool render_complete;

// Callback functions
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    
    // Reset rendering when camera moves
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        float move_speed = 0.1f;
        
        switch (key) {
            case GLFW_KEY_W:
                camera.position += camera.front * move_speed;
                camera.update_ray_tracing_parameters();
                render_complete = false;
                break;
            case GLFW_KEY_S:
                camera.position -= camera.front * move_speed;
                camera.update_ray_tracing_parameters();
                render_complete = false;
                break;
            case GLFW_KEY_A:
                camera.position -= camera.right * move_speed;
                camera.update_ray_tracing_parameters();
                render_complete = false;
                break;
            case GLFW_KEY_D:
                camera.position += camera.right * move_speed;
                camera.update_ray_tracing_parameters();
                render_complete = false;
                break;
            case GLFW_KEY_Q:
                camera.position += camera.up * move_speed;
                camera.update_ray_tracing_parameters();
                render_complete = false;
                break;
            case GLFW_KEY_E:
                camera.position -= camera.up * move_speed;
                camera.update_ray_tracing_parameters();
                render_complete = false;
                break;
        }
    }
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    static double last_x = 0.0;
    static double last_y = 0.0;
    static bool first_mouse = true;
    
    if (first_mouse) {
        last_x = xpos;
        last_y = ypos;
        first_mouse = false;
    }
    
    float xoffset = xpos - last_x;
    float yoffset = last_y - ypos;
    last_x = xpos;
    last_y = ypos;
    
    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;
    
    camera.yaw += xoffset;
    camera.pitch += yoffset;
    
    // Constrain pitch
    if (camera.pitch > 89.0f) camera.pitch = 89.0f;
    if (camera.pitch < -89.0f) camera.pitch = -89.0f;
    
    camera.update_camera_vectors();
    camera.update_ray_tracing_parameters();
    render_complete = false;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    camera.fov -= yoffset;
    if (camera.fov < 1.0f) camera.fov = 1.0f;
    if (camera.fov > 45.0f) camera.fov = 45.0f;
    camera.update_ray_tracing_parameters();
    render_complete = false;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    camera.aspect_ratio = float(width) / height;
    camera.update_ray_tracing_parameters();
    render_complete = false;
}

// Initialize window and callbacks
void init_window() {
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    
    // Capture mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
} 