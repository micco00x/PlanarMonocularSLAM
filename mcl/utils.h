#pragma once

#include <Eigen/Dense>

namespace mcl {
    const int APPEARANCE_SIZE = 10;

    Eigen::Matrix3f skew(const Eigen::Vector3f& v);
    Eigen::Matrix3f v2t(const Eigen::Vector3f& pose);
    Eigen::Vector3f t2v(const Eigen::Matrix3f& T);
    //Eigen::Matrix4f Rz_4f(float angle);
    Eigen::Matrix3f Rz(float angle);
    Eigen::Matrix3f Ry(float angle);
    Eigen::Matrix3f Rx(float angle);
}
