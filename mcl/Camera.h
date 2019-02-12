#pragma once
#include <Eigen/Dense>

namespace mcl {
    struct Camera {
        Eigen::Matrix3f matrix;
        Eigen::Matrix4f transform_rf_parent;
        float lambda_near, lambda_far;
        float width, height;
    };
}
