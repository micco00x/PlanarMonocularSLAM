#include <Eigen/Dense>

namespace mcl {
    namespace unicycle {
        Eigen::Vector3f transition(const Eigen::Vector3f& configuration,
                                   const Eigen::Vector2f& displacement) {
            return configuration + Eigen::Vector3f(displacement[0] * std::cos(configuration[2]),
                                                   displacement[0] * std::sin(configuration[2]),
                                                   displacement[1]);
        }
    }
}
