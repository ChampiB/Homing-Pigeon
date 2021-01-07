//
// Created by tmac3 on 06/01/2021.
//

#ifndef EXPERIMENTS_AI_TS_FUNCTIONS_H
#define EXPERIMENTS_AI_TS_FUNCTIONS_H

#include <Eigen/Dense>

namespace hopi::distributions {
    class Distribution;
}

namespace hopi::math {

    class Functions {
    public:
        static Eigen::MatrixXd softmax(Eigen::MatrixXd &vector); //TODO move corresponding tests
        static double KL(distributions::Distribution *d1, distributions::Distribution *d2); //TODO move corresponding tests
        static double KL_Categorical(distributions::Distribution *d1, distributions::Distribution *d2); //TODO test
        static double KL_Dirichlet(distributions::Distribution *d1, distributions::Distribution *d2); //TODO test
        static double beta(Eigen::MatrixXd x); //TODO test
        static double digamma(double x); //TODO test
    };

}


#endif //EXPERIMENTS_AI_TS_FUNCTIONS_H
