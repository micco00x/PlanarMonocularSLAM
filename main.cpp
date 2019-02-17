#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>

#include <Eigen/Dense>

#include "mcl/Camera.h"
#include "mcl/Landmark.h"
#include "mcl/Measurement.h"

#include "mcl/slam.h"
#include "utils.h"

int main() {
    std::string info_string;
    std::string dummy_string;

    mcl::Camera camera;
    std::vector<mcl::Landmark> landmarks;

    std::vector<Eigen::Vector3f> odom_trajectory;
    std::vector<Eigen::Vector3f> gt_trajectory;

    std::vector<std::vector<mcl::Measurement>> measurements;

    // Read data of the camera from file:
    std::ifstream camera_file("dataset/camera.dat");
    getline(camera_file, dummy_string);
    camera_file >> camera.matrix;
    getline(camera_file, dummy_string);
    getline(camera_file, dummy_string);
    camera_file >> camera.transform_rf_parent;
    camera_file >> dummy_string >> camera.lambda_near;
    camera_file >> dummy_string >> camera.lambda_far;
    camera_file >> dummy_string >> camera.width;
    camera_file >> dummy_string >> camera.height;

    // Read data of the world from file:
    std::ifstream world_file("dataset/world.dat");
    while (getline(world_file, dummy_string)) {
        mcl::Landmark landmark;
        std::istringstream ss(dummy_string);
        ss >> landmark;
        landmarks.push_back(landmark);
    }

    // Read data of the measurements from file(s):
    for (int k = 0; k < NUM_MEASUREMENTS; ++k) {
        std::stringstream ss;
        ss << std::setw(5) << std::setfill('0') << k;
        std::string measurement_filename = "dataset/meas-" + ss.str() + ".dat";
        std::ifstream measurement_file(measurement_filename);
        int seq;
        Eigen::Vector3f gt_pose;
        Eigen::Vector3f odom_pose;
        measurement_file >> dummy_string >> seq;
        measurement_file >> dummy_string >> gt_pose;
        measurement_file >> dummy_string >> odom_pose;
        gt_trajectory.push_back(gt_pose);
        odom_trajectory.push_back(odom_pose);
        std::vector<mcl::Measurement> v_meas;
        while (getline(measurement_file, info_string)) {
            mcl::Measurement meas;
            std::istringstream ss(info_string);
            ss >> dummy_string >> meas;
            v_meas.push_back(meas);
        }
        measurements.push_back(v_meas);
    }

    // SLAM:
    std::ofstream slam_file("bin/slam_trajectory.dat");
    std::map<int, int> id_to_state_map;
    std::vector<int> state_to_id_map;
    Eigen::VectorXf unicycle_pose_estimate = odom_trajectory[0];
    Eigen::MatrixXf covariance_estimate = 10.0f * Eigen::Matrix3f::Identity();
    for (int k = 1; k < NUM_MEASUREMENTS; ++k) {
        std::cout << "Iteration " << k << std::endl;
        Eigen::Vector3f prev_odom_pose = odom_trajectory[k-1];
        Eigen::Vector3f curr_odom_pose = odom_trajectory[k];

        mcl::slam::predict(unicycle_pose_estimate, covariance_estimate, curr_odom_pose - prev_odom_pose);
        //std::cout << "\tPREDICT: " << unicycle_pose_estimate.transpose() << std::endl;
        mcl::slam::update(gt_trajectory[k],
                         unicycle_pose_estimate, covariance_estimate,
                         landmarks, measurements[k], camera,
                         id_to_state_map, state_to_id_map);

        std::cout << "\tGround truth: " << gt_trajectory[k].transpose() << std::endl;
        std::cout << "\tUPDATE: " << unicycle_pose_estimate.head(3).transpose() << std::endl;
        std::cout << "\tdiff: " << gt_trajectory[k].transpose() - unicycle_pose_estimate.head(3).transpose() << std::endl;

        slam_file << unicycle_pose_estimate.head(3).transpose() << std::endl;
    }

    std::ofstream slam_landmarks_file("bin/slam_landmarks.dat");
    for (int k = 3; k < unicycle_pose_estimate.rows(); k += 2) {
        slam_landmarks_file << unicycle_pose_estimate.segment(k, 2).transpose() << std::endl;
    }

    return 0;
}
