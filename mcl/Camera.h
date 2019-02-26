#pragma once
#include <Eigen/Dense>

namespace mcl {
    struct Camera {
        Eigen::Matrix3f matrix;
        Eigen::Matrix4f transform_rf_parent;
        float lambda_near, lambda_far;
        float width, height;

        inline bool is_valid(float u, float v) const {
            return 0.0f <= u && u < width && 0.0f <= v && v < height;
        }

        inline bool is_valid(const Eigen::Vector2f& uv) const {
            return is_valid(uv[0], uv[1]);
        }
    };
}
