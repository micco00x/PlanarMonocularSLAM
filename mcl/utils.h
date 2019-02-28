#pragma once

#include <Eigen/Dense>

namespace mcl {
    const int APPEARANCE_SIZE = 10;

    Eigen::Vector4f to_homogeneous(const Eigen::Vector3f& v);
    Eigen::Vector3f to_inhomogeneous(const Eigen::Vector4f& v_hom);
    Eigen::Matrix3f skew(const Eigen::Vector3f& v);
    Eigen::Vector3f vector_from_skew(const Eigen::Matrix3f& S);
    Eigen::Matrix4f T_from_Rt(const Eigen::Matrix3f& R, const Eigen::Vector3f& t);
    Eigen::Matrix3f v2t(const Eigen::Vector3f& pose);
    Eigen::Matrix4f planar_v2t(const Eigen::Vector3f& pose);
    Eigen::Vector3f t2v(const Eigen::Matrix3f& T);
    Eigen::Matrix4f Rz_4f(float angle);
    Eigen::Matrix4f Ry_4f(float angle);
    Eigen::Matrix4f Rx_4f(float angle);
    Eigen::Matrix3f Rz(float angle);
    Eigen::Matrix3f Ry(float angle);
    Eigen::Matrix3f Rx(float angle);
}
