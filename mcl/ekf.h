#include <Eigen/Dense>

#include "Camera.h"
#include "Landmark.h"
#include "unicycle.h"

namespace mcl {
    namespace ekf {
        // estimated_pose: configuration of the unicycle (x, y, theta)
        // covariance_estimate: convariance matrix of estimated_pose
        // displacement: (norm(x_{t+1}-x_{t}), theta_{t+1}-theta{t})
        void predict(Eigen::Vector3f& estimated_pose,
                     Eigen::Matrix3f& covariance_estimate,
                     const Eigen::Vector2f& displacement) {

            Eigen::Matrix3f jacobian_transition;
            Eigen::Matrix<float, 3, 2> jacobian_controls;
            Eigen::Matrix2f covariance_controls;
            const float control_noise = 0.001f; // sigma_u^2

            jacobian_transition << 1.0f, 0.0f, -displacement[0] * std::sin(estimated_pose[2]),
                                   0.0f, 1.0f,  displacement[0] * std::cos(estimated_pose[2]),
                                   0.0f, 0.0f,                                           1.0f;

            jacobian_controls << std::cos(estimated_pose[2]), 0.0f,
                                 std::sin(estimated_pose[2]), 0.0f,
                                                        0.0f, 1.0f;

            covariance_controls << std::pow(displacement[0], 2.0f) + control_noise, 0.0f,
                                   0.0f, std::pow(displacement[1], 2.0f) + control_noise;

            // Perform prediciton step updating estimated_pose and covariance_estimate:
            estimated_pose = mcl::unicycle::transition(estimated_pose, displacement);
            covariance_estimate = jacobian_transition * covariance_estimate * jacobian_transition.transpose() +
                                  jacobian_controls * covariance_controls * jacobian_controls.transpose();
        }

        // estimated_pose: configuration of the unicycle (x, y, theta)
        // covariance_estimate: convariance matrix of estimated_pose
        // landmarks: vector containing all the information about the landmarks
        // measurements: vector containing all the information about the measurements (current step)
        void update(Eigen::Vector3f& gt_pose,
                    Eigen::Vector3f& estimated_pose,
                    Eigen::Matrix3f& covariance_estimate,
                    const std::vector<mcl::Landmark>& landmarks,
                    const std::vector<mcl::Measurement>& measurements,
                    const mcl::Camera& camera) {

            Eigen::VectorXf gt_bearings(measurements.size());
            Eigen::VectorXf estimated_bearings(measurements.size());

            Eigen::MatrixXf jacobian_measurements((measurements.size()), 3);

            //const Eigen::Matrix3f rotation_camera2robot = camera.transform_rf_parent.topLeftCorner(3, 3);
            //const Eigen::Vector3f translation_camera2robot = camera.transform_rf_parent.topRightCorner(3, 1);

            // Estimated transform:
            Eigen::Matrix2f estimated_rotation_robot2world;
            estimated_rotation_robot2world << std::cos(estimated_pose[2]), -std::sin(estimated_pose[2]),
                                              std::sin(estimated_pose[2]),  std::cos(estimated_pose[2]);
            Eigen::Matrix2f estimated_rotation_world2robot = estimated_rotation_robot2world.transpose();
            Eigen::Matrix2f estimated_derivative_transpose_rotation_world2robot;
            estimated_derivative_transpose_rotation_world2robot << -std::sin(estimated_pose[2]),  std::cos(estimated_pose[2]),
                                                                   -std::cos(estimated_pose[2]), -std::sin(estimated_pose[2]);

            // GT transform:
            Eigen::Matrix4f gt_transform_robot2world;
            gt_transform_robot2world <<  std::cos(gt_pose[2]), -std::sin(gt_pose[2]), 0.0f, gt_pose[0],
                                         std::sin(gt_pose[2]),  std::cos(gt_pose[2]), 0.0f, gt_pose[1],
                                                         0.0f,                  0.0f, 1.0f,       0.0f,
                                                         0.0f,                  0.0f, 0.0f,       1.0f;
            Eigen::Matrix4f gt_transform_world2camera = (gt_transform_robot2world * camera.transform_rf_parent).inverse();

            for (const auto& meas : measurements) {
                int meas_idx = &meas - &measurements[0];

                // Compute bearing from estimate:
                Eigen::Vector2f gt_landmark_position(landmarks[meas.gt_landmark_id].position.x(), landmarks[meas.gt_landmark_id].position.y());
                Eigen::Vector2f estimated_landmark_position_rf_robot = estimated_rotation_world2robot * (gt_landmark_position - estimated_pose.head(2));
                float estimated_bearing = std::atan2(estimated_landmark_position_rf_robot.y(), estimated_landmark_position_rf_robot.x());
                estimated_bearings[meas_idx] = estimated_bearing;

                // Compute bearing from measurement (simulating a depth sensor):
                float f  = camera.matrix(0, 0);
                float u0 = camera.matrix(0, 2);
                float v0 = camera.matrix(1, 2);
                float depth = (gt_transform_world2camera *
                    Eigen::Vector4f(landmarks[meas.gt_landmark_id].position.x(),
                                    landmarks[meas.gt_landmark_id].position.y(),
                                    landmarks[meas.gt_landmark_id].position.z(),
                                    1.0f)).z();
                Eigen::Vector4f gt_landmark_position_rf_camera((meas.u - u0) * depth / f, (meas.v - v0) * depth / f, depth, 1.0f);
                Eigen::Vector4f gt_landmark_position_rf_robot = camera.transform_rf_parent * gt_landmark_position_rf_camera;
                float gt_bearing = std::atan2(gt_landmark_position_rf_robot.y(), gt_landmark_position_rf_robot.x());
                gt_bearings[meas_idx] = gt_bearing;

                Eigen::Matrix<float, 2, 3> mr;
                mr.block<2, 2>(0, 0) = -estimated_rotation_world2robot;
                mr.block<2, 1>(0, 2) = estimated_derivative_transpose_rotation_world2robot * (gt_landmark_position - estimated_pose.head(2));
                Eigen::Matrix<float, 1, 3> jacobian_meas = 1.0f /
                    (std::pow(estimated_landmark_position_rf_robot.x(), 2.0f) + std::pow(estimated_landmark_position_rf_robot.y(), 2.0f)) *
                    Eigen::Vector2f(-estimated_landmark_position_rf_robot.y(), estimated_landmark_position_rf_robot.x()).transpose() * mr;

                jacobian_measurements.block<1, 3>(meas_idx, 0) = jacobian_meas;
            }

            float measurement_noise = 0.01f;
            Eigen::MatrixXf covariance_measurements = Eigen::MatrixXf::Identity(measurements.size(), measurements.size()) * measurement_noise;
            Eigen::MatrixXf kalman_gain_matrix = covariance_estimate * jacobian_measurements.transpose() * (jacobian_measurements * covariance_estimate * jacobian_measurements.transpose() + covariance_measurements).inverse();

            estimated_pose += kalman_gain_matrix * (gt_bearings - estimated_bearings);
            covariance_estimate = (Eigen::Matrix3f::Identity() - kalman_gain_matrix * jacobian_measurements) * covariance_estimate;
        }
    }
}
