#pragma once
#include "utils.h"

namespace mcl {
    struct Measurement {
        int measured_landmark_id;
        int gt_landmark_id;
        int u, v;
        Eigen::Matrix<float, mcl::APPEARANCE_SIZE, 1> appearance;
    };
}
