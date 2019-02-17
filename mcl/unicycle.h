#include <Eigen/Dense>

namespace mcl {
    namespace unicycle {
        void transition(Eigen::VectorXf& configuration,
                        const Eigen::Vector3f& displacement) {
            configuration.head(3) += displacement;
        }
    }
}
