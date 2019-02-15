#include <Eigen/Dense>

namespace mcl {
    namespace unicycle {
        void transition(Eigen::VectorXf& configuration,
                        const Eigen::Vector2f& displacement) {
            configuration.head(3) += Eigen::Vector3f(displacement[0] * std::cos(configuration[2]),
                                                     displacement[0] * std::sin(configuration[2]),
                                                     displacement[1]);
        }
    }
}
