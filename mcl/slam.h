#include <set>

#include <Eigen/Dense>

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
            Eigen::Matrix2f estimated_derivative_transpose_rotation_world2robot;
            estimated_derivative_transpose_rotation_world2robot << -std::sin(estimated_state[2]),  std::cos(estimated_state[2]),
                                                                   -std::cos(estimated_state[2]), -std::sin(estimated_state[2]);

            // GT transform:
            Eigen::Matrix4f gt_transform_robot2world;
            gt_transform_robot2world <<  std::cos(gt_pose[2]), -std::sin(gt_pose[2]), 0.0f, gt_pose[0],
                                         std::sin(gt_pose[2]),  std::cos(gt_pose[2]), 0.0f, gt_pose[1],
                                                         0.0f,                  0.0f, 1.0f,       0.0f,
                                                         0.0f,                  0.0f, 0.0f,       1.0f;
            Eigen::Matrix4f gt_transform_world2camera = (gt_transform_robot2world * camera.transform_rf_parent).inverse();

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

            Eigen::VectorXf gt_bearings(tot_measurements_known_landmarks);
            Eigen::VectorXf estimated_bearings(tot_measurements_known_landmarks);
            Eigen::MatrixXf jacobian_measurements(tot_measurements_known_landmarks, estimated_state.rows());

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
                    Eigen::Vector2f estimated_landmark_position = estimated_state.segment(3 + 2 * state_map_it->second, 2);
                    Eigen::Vector2f estimated_landmark_position_rf_robot = estimated_rotation_world2robot * (estimated_landmark_position - estimated_state.head(2));
                    float estimated_bearing = std::atan2(estimated_landmark_position_rf_robot.y(), estimated_landmark_position_rf_robot.x());
                    estimated_bearings[k] = estimated_bearing;

                    // Compute bearing from measurement (using simulated depth sensor):
                    float gt_bearing = std::atan2(gt_landmark_position_rf_robot.y(), gt_landmark_position_rf_robot.x());
                    gt_bearings[k] = gt_bearing;

                    // Compute partial derivative of the landmark (rf_robot) wrt the robot:
                    Eigen::Matrix<float, 2, 3> derivative_landmark_wrt_robot_rf_robot;
                    derivative_landmark_wrt_robot_rf_robot.block<2, 2>(0, 0) = -estimated_rotation_world2robot;
                    derivative_landmark_wrt_robot_rf_robot.block<2, 1>(0, 2) = estimated_derivative_transpose_rotation_world2robot * (estimated_landmark_position - estimated_state.head(2));

                    // Compute partial derivative of atan wrt landmark (rf_robot):
                    Eigen::Matrix<float, 1, 2> derivative_atan_wrt_landmark = 1.0f / estimated_landmark_position_rf_robot.squaredNorm() *
                        Eigen::Vector2f(-estimated_landmark_position_rf_robot.y(), estimated_landmark_position_rf_robot.x()).transpose();

                    // Compute partial derivative of atan wrt both robot and landmark (rf_world):
                    Eigen::MatrixXf jacobian_meas = Eigen::MatrixXf::Zero(1, estimated_state.rows());
                    jacobian_meas.block<1, 3>(0, 0) = derivative_atan_wrt_landmark * derivative_landmark_wrt_robot_rf_robot;
                    jacobian_meas.block<1, 2>(0, 3 + 2 * state_map_it->second) = derivative_atan_wrt_landmark * estimated_rotation_world2robot;

                    // Update full jacobian (atan wrt state):
                    jacobian_measurements.block(k, 0, jacobian_meas.rows(), jacobian_meas.cols()) = jacobian_meas;
                    ++k;
                }
            }

            if (tot_measurements_known_landmarks > 0) {
                const float measurement_noise = 0.01f;
                Eigen::MatrixXf covariance_measurements = Eigen::MatrixXf::Identity(tot_measurements_known_landmarks, tot_measurements_known_landmarks) * measurement_noise;
                Eigen::MatrixXf kalman_gain_matrix = covariance_estimate * jacobian_measurements.transpose() * (jacobian_measurements * covariance_estimate * jacobian_measurements.transpose() + covariance_measurements).inverse();
                estimated_state.noalias() += kalman_gain_matrix * (gt_bearings - estimated_bearings);
                covariance_estimate = (Eigen::MatrixXf::Identity(estimated_state.rows(), estimated_state.rows()) - kalman_gain_matrix * jacobian_measurements) * covariance_estimate;
            }

            Eigen::Matrix4f estimated_transform_robot2world;
            estimated_transform_robot2world << std::cos(estimated_state[2]), -std::sin(estimated_state[2]), 0.0f, estimated_state[0],
                                               std::sin(estimated_state[2]),  std::cos(estimated_state[2]), 0.0f, estimated_state[1],
                                                                       0.0f,                          0.0f, 1.0f,               0.0f,
                                                                       0.0f,                          0.0f, 0.0f,               1.0f;

            // Update estimate_state and covariance_estimate if new landmarks
            // have been seen:
            const float initial_landmark_noise = 2.0f;
            estimated_state.conservativeResize(estimated_state.rows() + 2 * number_of_unknown_landmarks);
            covariance_estimate.conservativeResize(covariance_estimate.rows() + 2 * number_of_unknown_landmarks,
                                       covariance_estimate.cols() + 2 * number_of_unknown_landmarks);
            covariance_estimate.rightCols(2 * number_of_unknown_landmarks).setZero();
            covariance_estimate.bottomRows(2 * number_of_unknown_landmarks).setZero();
            covariance_estimate.block(covariance_estimate.rows() - 2 * number_of_unknown_landmarks,
                                      covariance_estimate.cols() - 2 * number_of_unknown_landmarks,
                                      2 * number_of_unknown_landmarks,
                                      2 * number_of_unknown_landmarks) = initial_landmark_noise * Eigen::MatrixXf::Identity(2 * number_of_unknown_landmarks, 2 * number_of_unknown_landmarks);
            k = 0;
            for (const auto& meas : measurements) {
                int meas_idx = &meas - &measurements[0];

                if (id_to_state_map.find(meas.gt_landmark_id) == id_to_state_map.end()) {
                    // Update maps:
                    id_to_state_map[meas.gt_landmark_id] = state_to_id_map.size();
                    state_to_id_map.push_back(meas.gt_landmark_id);

                    // Update estimated_state (covariance updated above):
                    Eigen::Vector2f landmark_initial_position = (estimated_transform_robot2world * gt_landmark_position_rf_robot_vector[meas_idx]).head(2);
                    //std::cout << "Adding landmark id=" << meas.gt_landmark_id << " with initial diff " << (landmarks[meas.gt_landmark_id].position.head(2) - landmark_initial_position).transpose() << std::endl;
                    estimated_state.segment(estimated_state.rows()-2*(number_of_unknown_landmarks-k), 2) = landmark_initial_position;
                    ++k;
                }
            }
        }
    }
}
