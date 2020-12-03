//
// Created by Theophile Champion on 28/11/2020.
//

#ifndef HOMING_PIGEON_TRANSITION_H
#define HOMING_PIGEON_TRANSITION_H

#include "Distribution.h"
#include <memory>
#include <torch/torch.h>

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::distributions {

    /**
     * Class representing a Transition distribution.
     */
    class Transition : public Distribution {
    public:
        //
        // Factory
        //

        /**
         * Create a Transition distribution.
         * @param p the parameters of the distribution
         * @return the created distribution
         */
        static std::unique_ptr<Transition> create(const torch::Tensor &p);

    public:
        //
        // Constructor
        //

        /**
         * Construct a Transition distribution.
         * @param param the parameters of the distribution
         */
        explicit Transition(const torch::Tensor &param);

        //
        // Implementation of the methods of the Distribution class
        //

        /**
         * Getter.
         * @return the distribution's type
         */
        [[nodiscard]] DistributionType type() const override;

        /**
         * Getter.
         * @return the logarithm of the distribution's parameters
         */
        [[nodiscard]] torch::Tensor logParams() const override;

        /**
         * Getter.
         * @return the distribution's parameters
         */
        [[nodiscard]] torch::Tensor params() const override;

        /**
         * Update the distribution's parameters.
         * @param param the new parameters
         */
        void updateParams(const torch::Tensor &param) override;

        /**
         * Compute the entropy of the distribution.
         * @return the entropy
         */
        double entropy() override;

    private:
        torch::Tensor param;
    };

}

#endif //HOMING_PIGEON_TRANSITION_H
