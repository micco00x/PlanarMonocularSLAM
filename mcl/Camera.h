#pragma once
#include <Eigen/Dense>

namespace mcl {
    struct Camera {
        Eigen::Matrix3f matrix;
        float lambda_near, lambda_far;
        float width, height;
    };
}
