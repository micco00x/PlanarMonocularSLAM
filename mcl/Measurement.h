#pragma once

namespace mcl {
    struct Measurement {
        int measured_landmark_id;
        int gt_landmark_id;
        int u, v;
        Eigen::Matrix<float, 10, 1> appearance;
    };
}
