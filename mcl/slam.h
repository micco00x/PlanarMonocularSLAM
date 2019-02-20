#include <set>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "Camera.h"
#include "Landmark.h"
#include "unicycle.h"

namespace mcl {
    namespace slam {
        // estimated_state: extended configuration of the unicycle (x, y, theta, x_l1, y_l1, ...)
        // covariance_estimate: convariance matrix of estimated_state
        // displacement: (x_{t+1}-x_{t}, y_{t+1}-y_{t}, theta_{t+1}-theta{t})
        // control_noise: sigma_u^2
        void predict(Eigen::VectorXf& estimated_state,
                     Eigen::MatrixXf& covariance_estimate,
                     const Eigen::Vector3f& displacement,
                     const float control_noise = 0.001f) {

            Eigen::MatrixXf jacobian_transition = Eigen::MatrixXf::Identity(estimated_state.rows(), estimated_state.rows());
            Eigen::MatrixXf jacobian_controls = Eigen::MatrixXf::Zero(estimated_state.rows(), 3);
            Eigen::Matrix3f covariance_controls;

            jacobian_controls.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();

            covariance_controls << std::pow(displacement[0], 2.0f) + control_noise, 0.0f, 0.0f,
                                   0.0f, std::pow(displacement[1], 2.0f) + control_noise, 0.0f,
                                   0.0f, 0.0f, std::pow(displacement[2], 2.0f) + control_noise;

            // Perform prediciton step updating estimated_state and covariance_estimate:
            mcl::unicycle::transition(estimated_state, displacement);
            covariance_estimate = jacobian_transition * covariance_estimate * jacobian_transition.transpose() +
                                  jacobian_controls * covariance_controls * jacobian_controls.transpose();
        }

        // estimated_state: configuration of the unicycle (x, y, theta)
        // covariance_estimate: convariance matrix of estimated_state
        // landmarks: vector containing all the information about the landmarks
        // measurements: vector containing all the information about the measurements (current step)
        void update(Eigen::Vector3f& gt_pose,
                    Eigen::VectorXf& estimated_state,
                    Eigen::MatrixXf& covariance_estimate,
                    const std::vector<mcl::Landmark>& landmarks,
                    const std::vector<mcl::Measurement>& measurements,
                    const mcl::Camera& camera,
                    std::map<int, int>& id_to_state_map,
                    std::vector<int>& state_to_id_map) {

            // Estimated transform:
            Eigen::Matrix2f estimated_rotation_robot2world;
            estimated_rotation_robot2world << std::cos(estimated_state[2]), -std::sin(estimated_state[2]),
                                              std::sin(estimated_state[2]),  std::cos(estimated_state[2]);
            Eigen::Matrix2f estimated_rotation_world2robot = estimated_rotation_robot2world.transpose();
            Eigen::Matrix3f estimated_rotation_world2robot_3f = Eigen::Matrix3f::Identity();
            estimated_rotation_world2robot_3f.block<2, 2>(0, 0) = estimated_rotation_world2robot;
            Eigen::Matrix3f estimated_derivative_transpose_rotation_world2robot;
            estimated_derivative_transpose_rotation_world2robot << -std::sin(estimated_state[2]),  std::cos(estimated_state[2]), 0.0f,
                                                                   -std::cos(estimated_state[2]), -std::sin(estimated_state[2]), 0.0f,
                                                                                            0.0f,                          0.0f, 0.0f;

            // GT transform:
            Eigen::Matrix4f gt_transform_robot2world;
            gt_transform_robot2world <<  std::cos(gt_pose[2]), -std::sin(gt_pose[2]), 0.0f, gt_pose[0],
                                         std::sin(gt_pose[2]),  std::cos(gt_pose[2]), 0.0f, gt_pose[1],
                                                         0.0f,                  0.0f, 1.0f,       0.0f,
                                                         0.0f,                  0.0f, 0.0f,       1.0f;
            Eigen::Matrix4f gt_transform_world2camera = (gt_transform_robot2world * camera.transform_rf_parent).inverse();

            const Eigen::Matrix3f rotation_robot2camera = camera.transform_rf_parent.block<3, 3>(0, 0).transpose();

            int tot_measurements_known_landmarks = 0;

            std::vector<Eigen::Vector4f> gt_landmark_position_rf_robot_vector;

            // Quickly count number of known landmarks in current set of
            // measurements to avoid resizing multiple times,
            // NOTE: there could be multiple measurements relative to the
            // same landmark, make sure to count only once:
            std::set<int> id_known_landmarks;
            std::set<int> id_unknown_landmarks;
            for (const auto& meas : measurements) {
                if (id_to_state_map.find(meas.gt_landmark_id) != id_to_state_map.end()) {
                    ++tot_measurements_known_landmarks;
                    id_known_landmarks.insert(meas.gt_landmark_id);
                } else {
                    id_unknown_landmarks.insert(meas.gt_landmark_id);
                }
            }
            const int number_of_unknown_landmarks = id_unknown_landmarks.size();

            Eigen::VectorXf measured_uv(2 * tot_measurements_known_landmarks);
            Eigen::VectorXf estimated_uv(2 * tot_measurements_known_landmarks);
            Eigen::MatrixXf jacobian_measurements = Eigen::MatrixXf::Zero(2 * tot_measurements_known_landmarks, estimated_state.rows());

            int k = 0;
            for (const auto& meas : measurements) {
                // Simulating depth sensor:
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
                gt_landmark_position_rf_robot_vector.push_back(gt_landmark_position_rf_robot);

                // Check if measured landmark has already been seen:
                auto state_map_it = id_to_state_map.find(meas.gt_landmark_id);
                if (state_map_it != id_to_state_map.end()) {
                    // Compute bearing from estimate:
                    Eigen::Vector3f estimated_landmark_position = estimated_state.segment(3 + 3 * state_map_it->second, 3);
                    //std::cout << "*** KNOWN LANDMARK FOUND ***" << std::endl;
                    //std::cout << "\t> estimated_landmark_position: " << estimated_landmark_position.transpose() << std::endl;
                    //std::cout << "\t> landmarks[meas.gt_landmark_id].position: " << landmarks[meas.gt_landmark_id].position.transpose() << std::endl;
                    Eigen::Vector3f estimated_landmark_position_rf_robot = estimated_rotation_world2robot_3f *
                        (estimated_landmark_position - Eigen::Vector3f(estimated_state[0], estimated_state[1], 0.0f));
                    Eigen::Vector3f estimated_landmark_position_rf_camera = (camera.transform_rf_parent.inverse() *
                                                                            Eigen::Vector4f(estimated_landmark_position_rf_robot.x(),
                                                                                            estimated_landmark_position_rf_robot.y(),
                                                                                            estimated_landmark_position_rf_robot.z(),
                                                                                            1.0f)).head(3);
                    Eigen::Vector3f Kpcam = camera.matrix * estimated_landmark_position_rf_camera;
                    estimated_uv(2 * k)     = Kpcam.x() / Kpcam.z();
                    estimated_uv(2 * k + 1) = Kpcam.y() / Kpcam.z();

                    // Compute bearing from measurement (using simulated depth sensor):
                    measured_uv(2 * k)     = meas.u;
                    measured_uv(2 * k + 1) = meas.v;

                    //std::cout << "estimated_uv: " << estimated_uv(2 * k) << ", " << estimated_uv(2 * k + 1)
                    //    << " - measured_uv: "<< measured_uv(2 * k) << ", " << measured_uv(2 * k + 1) << std::endl;

                    // Compute partial derivative of the landmark express in
                    // RF robot wrt robot (in rf world):
                    Eigen::Matrix<float, 3, 3> derivative_landmark_wrt_robot_rf_robot;
                    derivative_landmark_wrt_robot_rf_robot.block<3, 2>(0, 0) = -estimated_rotation_world2robot_3f.block<3, 2>(0, 0);
                    derivative_landmark_wrt_robot_rf_robot.block<3, 1>(0, 2) = estimated_derivative_transpose_rotation_world2robot * (estimated_landmark_position - Eigen::Vector3f(estimated_state[0], estimated_state[1], 0.0f));

                    // Compute partial derivative of proj wrt p=KR^T(x_t - t_t), K camera matrix:
                    Eigen::Matrix<float, 2, 3> derivative_proj_wrt_Kpcam;
                    derivative_proj_wrt_Kpcam << 1.0f / Kpcam.z(), 0.0f, -Kpcam.x() / std::pow(Kpcam.z(), 2.0f),
                                                 0.0f, 1.0f / Kpcam.z(), -Kpcam.y() / std::pow(Kpcam.z(), 2.0f);

                    // Compute partial derivative of proj wrt state (both robot and landmark):
                    jacobian_measurements.block<2, 3>(2 * k, 0) = derivative_proj_wrt_Kpcam * camera.matrix * rotation_robot2camera * derivative_landmark_wrt_robot_rf_robot;
                    jacobian_measurements.block<2, 3>(2 * k, 3 + 3 * state_map_it->second) = derivative_proj_wrt_Kpcam * camera.matrix * rotation_robot2camera * estimated_rotation_world2robot_3f;
                    ++k;
                }
            }

            if (tot_measurements_known_landmarks > 0) {
                const float measurement_noise = 12.0f;
                const Eigen::MatrixXf covariance_measurements = measurement_noise * Eigen::MatrixXf::Identity(2 * tot_measurements_known_landmarks, 2 * tot_measurements_known_landmarks);
                Eigen::MatrixXf kalman_gain_matrix = covariance_estimate * jacobian_measurements.transpose() * (jacobian_measurements * covariance_estimate * jacobian_measurements.transpose() + covariance_measurements).inverse();
                estimated_state.noalias() += kalman_gain_matrix * (measured_uv - estimated_uv);
                covariance_estimate = (Eigen::MatrixXf::Identity(estimated_state.rows(), estimated_state.rows()) - kalman_gain_matrix * jacobian_measurements) * covariance_estimate;
            }

            Eigen::Matrix4f estimated_transform_robot2world;
            estimated_transform_robot2world << std::cos(estimated_state[2]), -std::sin(estimated_state[2]), 0.0f, estimated_state[0],
                                               std::sin(estimated_state[2]),  std::cos(estimated_state[2]), 0.0f, estimated_state[1],
                                                                       0.0f,                          0.0f, 1.0f,               0.0f,
                                                                       0.0f,                          0.0f, 0.0f,               1.0f;

            // Update estimate_state and covariance_estimate if new landmarks
            // have been seen:
            const float initial_landmark_noise = 0.2f;
            estimated_state.conservativeResize(estimated_state.rows() + 3 * number_of_unknown_landmarks);
            covariance_estimate.conservativeResize(covariance_estimate.rows() + 3 * number_of_unknown_landmarks,
                                       covariance_estimate.cols() + 3 * number_of_unknown_landmarks);
            covariance_estimate.rightCols(3 * number_of_unknown_landmarks).setZero();
            covariance_estimate.bottomRows(3 * number_of_unknown_landmarks).setZero();
            covariance_estimate.block(covariance_estimate.rows() - 3 * number_of_unknown_landmarks,
                                      covariance_estimate.cols() - 3 * number_of_unknown_landmarks,
                                      3 * number_of_unknown_landmarks,
                                      3 * number_of_unknown_landmarks) = initial_landmark_noise * Eigen::MatrixXf::Identity(3 * number_of_unknown_landmarks, 3 * number_of_unknown_landmarks);
            k = 0;
            for (const auto& meas : measurements) {
                int meas_idx = &meas - &measurements[0];

                if (id_to_state_map.find(meas.gt_landmark_id) == id_to_state_map.end()) {
                    // Update maps:
                    id_to_state_map[meas.gt_landmark_id] = state_to_id_map.size();
                    state_to_id_map.push_back(meas.gt_landmark_id);

                    // Update estimated_state (covariance updated above):
                    Eigen::Vector3f landmark_initial_position = (estimated_transform_robot2world * gt_landmark_position_rf_robot_vector[meas_idx]).head(3);
                    std::cout << "Adding landmark id=" << meas.gt_landmark_id << " with initial diff " << (landmarks[meas.gt_landmark_id].position - landmark_initial_position).transpose() << std::endl;
                    std::cout << "\t>gt: " << landmarks[meas.gt_landmark_id].position.transpose() << std::endl;
                    std::cout << "\t>init: " << landmark_initial_position.transpose() << std::endl;
                    estimated_state.segment(estimated_state.rows()-3*(number_of_unknown_landmarks-k), 3) = landmark_initial_position;
                    ++k;
                }
            }
        }
    }
}
