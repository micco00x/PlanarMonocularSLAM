#pragma once
#include <fstream>
#include <Eigen/Dense>

namespace mcl {
    struct Landmark {
        unsigned int id;
        Eigen::Vector3f position;
        //float appearance[10];
        Eigen::Matrix<float, 10, 1> appearance;
    };
}
