//
// Created by Theophile Champion on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_DISTRIBUTION_H
#define HOMING_PIGEON_2_DISTRIBUTION_H

#include <vector>
#include <torch/torch.h>
#include "DistributionType.h"

namespace hopi::distributions {

    /**
     * Interface representing a general probability distribution.
     */
    class Distribution {
    public:
        /**
         * Getter.
         * @return the distribution's type
         */
        [[nodiscard]] virtual DistributionType type() const = 0;

        /**
         * Getter.
         * @return the logarithm of the distribution's parameters
         */
        [[nodiscard]] virtual torch::Tensor logParams() const = 0;

        /**
         * Getter.
         * @return the distribution's parameters
         */
        [[nodiscard]] virtual torch::Tensor params() const = 0;

        /**
         * Update the distribution's parameters.
         * @param param the new parameters
         */
        virtual void updateParams(const torch::Tensor &param) = 0;

        /**
         * Compute the entropy of the distribution.
         * @return the entropy
         */
        virtual double entropy() = 0;
    };

}

#endif //HOMING_PIGEON_2_DISTRIBUTION_H
