//
// Created by tmac3 on 06/01/2021.
//

#ifndef EXPERIMENTS_AI_TS_FUNCTIONS_H
#define EXPERIMENTS_AI_TS_FUNCTIONS_H

#include <torch/torch.h>

namespace hopi::distributions {
    class Distribution;
}

namespace hopi::math {

    class Ops {
    public:
        // Math function
        static double KL(distributions::Distribution *d1, distributions::Distribution *d2);
        static double KL_Categorical(distributions::Distribution *d1, distributions::Distribution *d2);
        static double KL_Dirichlet(distributions::Distribution *d1, distributions::Distribution *d2);
        static double log_beta(const torch::Tensor &x);
        static double beta(const torch::Tensor &x);
        static double digamma(double x);

        // Matrices creation
        static torch::Tensor oneHot(int size, int index);
        static torch::Tensor uniformColumnWise(const torch::ArrayRef<long> &sizes);

        // Tensor operators
        // TODO multiplication operator
        // TODO average operator
        // TODO apply the above ops to the message passing algorithm
        // TODO use the function torch::apply()
    };

}

#endif //EXPERIMENTS_AI_TS_FUNCTIONS_H
