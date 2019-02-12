#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include <Eigen/Dense>

#include "mcl/Camera.h"
#include "mcl/Landmark.h"
#include "mcl/Measurement.h"

#include "utils.h"

int main() {
    std::string info_string;
    std::string dummy_string;

    mcl::Camera camera;
    Eigen::Matrix4f camera_transform_rf_robot;
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
    camera_file >> camera_transform_rf_robot;
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

    return 0;
}
