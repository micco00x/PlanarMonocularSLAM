#include "utils.h"

namespace mcl {
    Eigen::Vector4f to_homogeneous(const Eigen::Vector3f& v) {
        return Eigen::Vector4f(v.x(), v.y(), v.z(), 1.0f);
    }

    Eigen::Vector3f to_inhomogeneous(const Eigen::Vector4f& v_hom) {
        return v_hom.head<3>() / v_hom[3];
    }

    Eigen::Matrix3f skew(const Eigen::Vector3f& v) {
        Eigen::Matrix3f S;
        S <<   0.0f, -v.z(),  v.y(),
              v.z(),   0.0f, -v.x(),
             -v.y(),  v.x(),   0.0f;
        return S;
    }

    // pose = (x, y, theta):
    Eigen::Matrix3f v2t(const Eigen::Vector3f& pose) {
        Eigen::Matrix3f T = mcl::Rz(pose[2]);
        T.block<2, 1>(0, 2) = pose.head<2>();
        return T;
    }

    // as v2t in 2d but considering 3d space and x-y plane (z=0):
    Eigen::Matrix4f planar_v2t(const Eigen::Vector3f& pose) {
        Eigen::Matrix4f T = mcl::Rz_4f(pose[2]);
        T.block<2, 1>(0, 3) = pose.head<2>();
        return T;
    }

    Eigen::Vector3f t2v(const Eigen::Matrix3f& T) {
        Eigen::Vector3f pose;
        pose << T(0, 2), T(1, 2), std::atan2(T(1, 0), T(1, 1));
        return pose;
    }

    // pose = (x, y, z, roll, pitch, yaw):
    /*Eigen::Matrix4f v2t(const mcl::Vector6f& pose) {
        Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
        transform.block<3, 3>(0, 0) = Rz(pose[5]) * Ry(pose[4]) * Rx(pose[3]);
        transform.block<3, 1>(0, 3) = pose.head<3>();
        return transform;
    }*/

    Eigen::Matrix4f Rz_4f(float angle) {
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T.block<3, 3>(0, 0) = mcl::Rz(angle);
        return T;
    }

    Eigen::Matrix3f Rz(float angle) {
        Eigen::Matrix3f R;
        R << std::cos(angle), -std::sin(angle), 0.0f,
             std::sin(angle),  std::cos(angle), 0.0f,
                        0.0f,             0.0f, 1.0f;
        return R;
    }

    Eigen::Matrix3f Ry(float angle) {
        Eigen::Matrix3f R;
        R <<  std::cos(angle), 0.0f, std::sin(angle),
                         0.0f, 1.0f,            0.0f,
             -std::sin(angle), 0.0f, std::cos(angle);
        return R;
    }

    Eigen::Matrix3f Rx(float angle) {
        Eigen::Matrix3f R;
        R << 1.0f,            0.0f,             0.0f,
             0.0f, std::cos(angle), -std::sin(angle),
             0.0f, std::sin(angle),  std::cos(angle);
        return R;
    }
}
