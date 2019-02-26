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
            // Actually check that the string read is a point:
            if (info_string.find("point") == 0) {
                mcl::Measurement meas;
                std::istringstream iss(info_string);
                iss >> dummy_string >> meas;
                v_meas.push_back(meas);
            }
        }
        measurements.push_back(v_meas);
    }

/*
    // SLAM:
    std::ofstream slam_file("bin/slam_trajectory.dat");
    std::map<int, int> id_to_state_map;
    std::vector<int> state_to_id_map;
    Eigen::VectorXf unicycle_pose_estimate = odom_trajectory[0];
    Eigen::MatrixXf covariance_estimate = Eigen::Matrix3f::Identity();
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
    for (int k = 3; k < unicycle_pose_estimate.rows(); k += 3) {
        slam_landmarks_file << unicycle_pose_estimate.segment(k, 3).transpose() << std::endl;
    }
*/

    // Reading estimated trajectory (previously done with EKF-SLAM):
    /*std::vector<Eigen::Vector3f> estimated_trajectory;
    estimated_trajectory.push_back(odom_trajectory[0]);
    std::ifstream estimated_trajectory_file("bin/slam_trajectory-end.dat");
    while (getline(estimated_trajectory_file, info_string)) {
        Eigen::Vector3f pose;
        std::istringstream ss(info_string);
        ss >> pose[0] >> pose[1] >> pose[2];
        estimated_trajectory.push_back(pose);
    }*/

    std::vector<Eigen::Vector3f> odometry_displacement;
    for (int k = 1; k < NUM_MEASUREMENTS; ++k) {
        odometry_displacement.push_back(odom_trajectory[k] - odom_trajectory[k-1]);
    }

    // TODO: algorithm to initialize the landmarks (reading from world.dat now)
    std::vector<Eigen::Vector3f> estimated_landmarks;
    for (const auto& landmark : landmarks) {
        estimated_landmarks.push_back(landmark.position);
    }

    // Put all measurements in a vector of measurements and build a
    // proj_association between poses idx and landmark id:
    std::vector<mcl::Measurement> full_measurements;
    std::vector<std::pair<int, int>> proj_pose_landmark_association;
    int c_it_idx = 0;
    for (const auto& c_it : measurements) {
        for (const auto& c_meas_it : c_it) {
            full_measurements.push_back(c_meas_it);
            proj_pose_landmark_association.push_back(std::make_pair(c_it_idx, c_meas_it.gt_landmark_id));
        }
        ++c_it_idx;
    }

    int num_iterations = 15;
    float damping = 0.0f;
    float kernel_threshold = 20000.0f; // sqrt(1000)=31.62[px], sqrt(10000)=100.00[px], sqrt(20000)=141.42[px]
    std::cout << "*** Least Squares ***" << std::endl;
    mcl::slam::least_squares(odom_trajectory,
                             estimated_landmarks,
                             full_measurements, // landmark measurements
                             proj_pose_landmark_association,    // landmark data association
                             //odometry_displacement,
                             landmarks,
                             camera,
                             num_iterations,
                             damping,
                             kernel_threshold);

     std::ofstream ls_slam_file("bin/ls_slam_trajectory.dat");
     for (const auto& pose : odom_trajectory) {
         ls_slam_file << pose.transpose() << std::endl;
     }

     std::ofstream ls_slam_landmarks_file("bin/ls_slam_landmarks.dat");
     for (const auto& landmark : estimated_landmarks) {
         ls_slam_landmarks_file << landmark.transpose() << std::endl;
     }

    return 0;
}
