#pragma once
#include <Eigen/Dense>

namespace mcl {
    struct Camera {
        Eigen::Matrix3f matrix;
        Eigen::Matrix4f transform_rf_parent;
        float lambda_near, lambda_far;
        float width, height;

        inline bool is_uv_valid(float u, float v) const {
            return 0.0f <= u && u < width && 0.0f <= v && v < height;
        }

        inline bool is_uv_valid(const Eigen::Vector2f& uv) const {
            return is_uv_valid(uv[0], uv[1]);
        }

        inline bool is_pcam_valid(const Eigen::Vector3f& pcam) const {
            return lambda_near <= pcam.z() && pcam.z() <= lambda_far;
        }
    };
}
