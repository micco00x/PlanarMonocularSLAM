#pragma once
#include <iostream>
#include <Eigen/Dense>

#include "mcl/Landmark.h"
#include "mcl/Measurement.h"

const int NUM_MEASUREMENTS = 336;

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
std::istream& operator>>(std::istream& is, Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>& m) {
    for (int r = 0; r < m.rows(); ++r) {
        for (int c = 0; c < m.cols(); ++c) {
            is >> m(r, c);
        }
    }
    return is;
}

std::istream& operator>>(std::istream& is, mcl::Landmark& landmark) {
    is >> landmark.id >> landmark.position >> landmark.appearance;
    return is;
}

std::istream& operator>>(std::istream& is, mcl::Measurement& measurement) {
    is >> measurement.measured_landmark_id >> measurement.gt_landmark_id
        >> measurement.u >> measurement.v >> measurement.appearance;
    return is;
}
