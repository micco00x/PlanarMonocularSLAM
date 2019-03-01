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

    // Pose-pose relation from odometry:
    std::vector<Eigen::Vector3f> odometry_displacement;
    for (int k = 1; k < NUM_MEASUREMENTS; ++k) {
        odometry_displacement.push_back(odom_trajectory[k] - odom_trajectory[k-1]);
    }

    // Vector containing all homogeneous camera projection matrices P:
    std::vector<Eigen::Matrix<float, 3, 4>> homogeneous_camera_projection_matrices(odom_trajectory.size());
    //for (const auto& odom_pose : odom_trajectory) {
    for (int k = 0; k < odom_trajectory.size(); ++k) {
        const auto& odom_pose = odom_trajectory[k];
        Eigen::Matrix4f T_robot2world   = mcl::planar_v2t(odom_pose);
        Eigen::Matrix4f T_camera2world  = T_robot2world * camera.transform_rf_parent;
        Eigen::Matrix3f R_world2camera  = T_camera2world.block<3, 3>(0, 0).transpose();
        Eigen::Vector3f camera_position = T_camera2world.block<3, 1>(0, 3);
        Eigen::Matrix<float, 3, 4> hom_camera_proj_matrix; // P = KR[I|-C]
        hom_camera_proj_matrix.block<3, 3>(0, 0).setIdentity();
        hom_camera_proj_matrix.block<3, 1>(0, 3) = -camera_position;
        hom_camera_proj_matrix = camera.matrix * R_world2camera * hom_camera_proj_matrix;
        homogeneous_camera_projection_matrices[k] = hom_camera_proj_matrix;
    }

    // Vector containing DLT matrix for each landmark:
    // NOTE: setting matrix A as explained in Zisserman p.312 and in
    // https://www.uio.no/studier/emner/matnat/its/UNIK4690/v16/forelesninger/lecture_7_2-triangulation.pdf.
    // NOTE: using landmarks.size(), let's say we already know the number of landmarks:
    std::vector<Eigen::Matrix<float, Eigen::Dynamic, 4>> dlt_matrices(landmarks.size());
    //for (const auto& c_it : measurements) {
    for (int k = 0; k < measurements.size(); ++k) {
        const auto& meas_v = measurements[k];
        const Eigen::Matrix<float, 1, 4> p_row1 = homogeneous_camera_projection_matrices[k].block<1, 4>(0, 0);
        const Eigen::Matrix<float, 1, 4> p_row2 = homogeneous_camera_projection_matrices[k].block<1, 4>(1, 0);
        const Eigen::Matrix<float, 1, 4> p_row3 = homogeneous_camera_projection_matrices[k].block<1, 4>(2, 0);
        for (const auto& meas : meas_v) {
            int curr_n_rows = dlt_matrices[meas.gt_landmark_id].rows();
            dlt_matrices[meas.gt_landmark_id].conservativeResize(curr_n_rows + 2, Eigen::NoChange);
            dlt_matrices[meas.gt_landmark_id].row(curr_n_rows  ) = meas.v * p_row3 - p_row2;
            dlt_matrices[meas.gt_landmark_id].row(curr_n_rows+1) = meas.u * p_row3 - p_row1;
        }
    }

    // Determining initial guess of the landmarks:
    // NOTE: using algorithm A5.4 p.593 (Zisserman).
    std::vector<Eigen::Vector3f> dlt_landmarks;
    std::map<int, int> dlt_landmarks_id_to_idx; // helper map to give correct position of dlt_landmarks
    std::set<int> discarded_landmark_ids;
    for (int k = 0; k < dlt_matrices.size(); ++k) {
        const auto& dlt_A = dlt_matrices[k];
        if (dlt_A.rows() >= 4) {
            Eigen::JacobiSVD<Eigen::MatrixXf> svd(dlt_A, Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::Vector4f landmark_pos_init_estimate_hom = svd.matrixV().rightCols(1).col(0);
            Eigen::Vector3f landmark_pos_init_estimate = mcl::to_inhomogeneous(landmark_pos_init_estimate_hom);
            //std::cout << "Landmark " << k << ". gt: " << landmarks[k].position.transpose()
            //    << " - estimate: " << landmark_pos_init_estimate.transpose()
            //    << " - error: " << (landmarks[k].position - landmark_pos_init_estimate).transpose()
            //    << std::endl;
            dlt_landmarks_id_to_idx[k] = dlt_landmarks.size();
            dlt_landmarks.push_back(landmark_pos_init_estimate);
        } else {
            //std::cout << "CANNOT DETERMINE LANDMARK id=" << k << std::endl;
            discarded_landmark_ids.insert(k);
        }
    }

    std::ofstream ls_slam_landmarks_dlt_init_file("bin/ls_slam_landmarks_dlt_init.dat");
    for (const auto& landmark : dlt_landmarks) {
        ls_slam_landmarks_dlt_init_file << landmark.transpose() << std::endl;
    }

    // Put all measurements in a vector of measurements and build a
    // proj_association between poses idx and landmark id:
    std::vector<mcl::Measurement> full_measurements;
    std::vector<std::pair<int, int>> proj_pose_landmark_association;
    int c_it_idx = 0;
    for (const auto& c_it : measurements) {
        for (const auto& c_meas_it : c_it) {
            // Add measurement only if the landmark has not been discarded:
            if (discarded_landmark_ids.find(c_meas_it.gt_landmark_id) == discarded_landmark_ids.end()) {
                full_measurements.push_back(c_meas_it);
                proj_pose_landmark_association.push_back(std::make_pair(c_it_idx, dlt_landmarks_id_to_idx[c_meas_it.gt_landmark_id]));
            }
        }
        ++c_it_idx;
    }

    // Pose-pose relation doing 8-point algorithm:
    // TODO: as explained in https://en.wikipedia.org/wiki/Essential_matrix#Determining_R_and_t_from_E
    //       there could be 4 different solutions, only one of these is feasible, you must generate
    //       all of them and choose the best one. You will have a problem of scaling anyway so
    //       you must deal with that (didn't handled here).
    // TODO: maybe better to do normalized 8-point algorithm to find F, then
    //       determine E = K'^T*F*K (K'=K).
/*
    for (int idxi_meas = 0; idxi_meas < measurements.size(); ++idxi_meas) {
        const auto& meas_vi = measurements[idxi_meas];
        for (int idxj_meas = idxi_meas + 1; idxj_meas < measurements.size(); ++idxj_meas) {
            const auto& meas_vj = measurements[idxj_meas];
            // Check that there are at least 8 correspondences between the two
            // images and then solve the minimization problem as explained in
            // https://en.wikipedia.org/wiki/Eight-point_algorithm:
            Eigen::Matrix<float, 9, Eigen::Dynamic> Y;
            int idxi_land = 0;
            int idxj_land = 0;
            int num_correspondences = 0;
            while (idxi_land < meas_vi.size() &&
                   idxj_land < meas_vj.size()) {
                const mcl::Measurement& meas1 = meas_vi[idxi_land];
                const mcl::Measurement& meas2 = meas_vj[idxj_land];
                const int landmark_id_i = meas1.gt_landmark_id;
                const int landmark_id_j = meas2.gt_landmark_id;
                if (landmark_id_i == landmark_id_j) {
                    // Correspondence found, add it to Y:
                    Eigen::Matrix<float, 9, 1> y_elem;
                    y_elem << meas2.u * meas1.u,
                              meas2.u * meas1.v,
                              meas2.u          ,
                              meas2.v * meas1.u,
                              meas2.v * meas1.v,
                              meas2.v          ,
                                        meas1.u,
                                        meas1.v,
                                           1.0f;
                    Y.conservativeResize(Eigen::NoChange, Y.cols()+1);
                    Y.col(num_correspondences) = y_elem;
                    ++num_correspondences;
                    ++idxi_land;
                    ++idxj_land;
                } else if (landmark_id_i < landmark_id_j) {
                    // No correspondence found, idx of first vector is less
                    // than second one, so you can increas idx of first to
                    // find match, more efficient than checking everything.
                    ++idxi_land;
                } else {
                    // Same as before but for the second vector.
                    ++idxj_land;
                }
            }

            // If there are at least 8 correspondences the 8-point algorithm can be used:
            if (num_correspondences >= 8) {
                std::cout << num_correspondences << " CORRESPONDENCES FOUND: " << idxi_meas << " " << idxj_meas << std::endl;
                Eigen::JacobiSVD<Eigen::MatrixXf> Y_svd(Y, Eigen::ComputeThinU | Eigen::ComputeThinV);
                //std::cout << "Y_svd.matrixU():\n" << Y_svd.matrixU() << std::endl;
                Eigen::Matrix<float, 9, 1> e_v = Y_svd.matrixU().rightCols(1).col(0);
                Eigen::Matrix3f E_est;
                E_est << e_v[0], e_v[1], e_v[2],
                         e_v[3], e_v[4], e_v[5],
                         e_v[6], e_v[7], e_v[8];
                // Enforce internal constraint (solve min_{E'}(||E'-E_{est}||_F)
                // as explain in 8-point algorithm step 3):
                std::cout << "\t> enforcing internal constraint" << std::endl;
                Eigen::JacobiSVD<Eigen::MatrixXf> E_est_svd(E_est, Eigen::ComputeThinU | Eigen::ComputeThinV);
                Eigen::Vector3f singular_values = E_est_svd.singularValues();
                singular_values[2] = 0.0f;
                Eigen::Matrix3f essential_matrix = E_est_svd.matrixU() *
                                                   singular_values.asDiagonal() *
                                                   E_est_svd.matrixV().transpose();
                std::cout << "Checking essential matrix is OK:" << std::endl;
                for (const auto& m1 : meas_vi) {
                    for (const auto& m2 : meas_vj) {
                        if (m1.gt_landmark_id == m2.gt_landmark_id) {
                            std::cout << "\tid " << m1.gt_landmark_id << ": "
                                << Eigen::Vector3f(m2.u, m2.v, 1.0f).transpose() *
                                   essential_matrix *
                                   Eigen::Vector3f(m1.u, m1.v, 1.0f)
                                << std::endl;
                        }
                    }
                }
                // Build R and t from essential matrix as explained in
                // https://en.wikipedia.org/wiki/Essential_matrix#Determining_R_and_t_from_E:
                // TODO: you should actually check between the four possibilities
                //       using W, W^T, t, -t that the resulting T is valid (there
                //       is only one among the four).
                std::cout << "\t> building R, t" << std::endl;
                Eigen::JacobiSVD<Eigen::MatrixXf> E_svd(essential_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
                // Make sure U and V have positive determinant:
                Eigen::MatrixXf U = E_svd.matrixU();
                Eigen::MatrixXf V = E_svd.matrixV();
                if (U.determinant() < 0.0f) U *= -1.0f;
                if (V.determinant() < 0.0f) V *= -1.0f;
                std::cout << "E_svd singular values: " << E_svd.singularValues().transpose() << std::endl;
                // Discard relation if first and second singular values are too
                // far away (they should be: s, s, 0).
                if (E_svd.singularValues()[0] - E_svd.singularValues()[1] >= 0.01f) {
                    std::cout << "SINGULAR VALUES TOO FAR AWAY." << std::endl;
                    continue;
                }
                Eigen::Matrix3f W;
                W << 0.0f, -1.0f, 0.0f,
                     1.0f,  0.0f, 0.0f,
                     0.0f,  0.0f, 1.0f;
                W.transposeInPlace();
                Eigen::Matrix3f Z;
                Z <<  0.0f, 1.0f, 0.0f,
                     -1.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f;
                Eigen::Matrix3f skew_t = U * W * E_svd.singularValues().asDiagonal() * U.transpose();
                //Eigen::Matrix3f skew_t = U * Z * U.transpose();
                // Note that here R and t are such that E = R*skew(t), the
                // transform from the 1nd RF to the 2st it T=[R -Rt; 0 1],
                // hence, from 2nd to 1st is T=[R^T t; 0 1].
                Eigen::Vector3f t_camera2camera = mcl::vector_from_skew(skew_t);
                Eigen::Matrix3f R_camera2camera = U * W.transpose() * V.transpose();
                std::cout << "skew_t:\n" << skew_t << std::endl;
                std::cout << "E-R[t]_x:\n" << essential_matrix - R_camera2camera*skew_t << std::endl;
                Eigen::Matrix4f T_camera2camera = mcl::T_from_Rt(R_camera2camera.transpose(), t_camera2camera);
                std::cout << "T_camera2camera:\n" << T_camera2camera << std::endl;
                Eigen::Matrix4f T_robot2robot = camera.transform_rf_parent * T_camera2camera *
                    camera.transform_rf_parent.inverse();
                Eigen::Matrix3f T_robot2robot_gt = mcl::v2t(gt_trajectory[idxi_meas]).inverse() * mcl::v2t(gt_trajectory[idxj_meas]);
                std::cout << "T_robot2robot_gt:\n" << T_robot2robot_gt << std::endl;
                std::cout << "T_robot2robot:\n" << T_robot2robot << std::endl;
            }
        }
    }
*/

    int num_iterations = 15;
    float damping = 0.0f;
    float kernel_threshold_proj = 20000.0f; // sqrt(1000)=31.62[px], sqrt(10000)=100.00[px], sqrt(20000)=141.42[px]
    float kernel_threshold_pose = 0.1f;
    std::cout << "*** Least Squares ***" << std::endl;
    mcl::slam::least_squares(odom_trajectory,
                             dlt_landmarks,
                             full_measurements, // landmark measurements
                             proj_pose_landmark_association,    // landmark data association
                             odometry_displacement,
                             //landmarks,
                             camera,
                             num_iterations,
                             damping,
                             kernel_threshold_proj,
                             kernel_threshold_pose);

     std::ofstream ls_slam_file("bin/ls_slam_trajectory.dat");
     for (const auto& pose : odom_trajectory) {
         ls_slam_file << pose.transpose() << std::endl;
     }

     std::ofstream ls_slam_landmarks_file("bin/ls_slam_landmarks.dat");
     for (const auto& landmark : dlt_landmarks) {
         ls_slam_landmarks_file << landmark.transpose() << std::endl;
     }

    return 0;
}
