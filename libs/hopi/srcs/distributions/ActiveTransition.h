//
// Created by Theophile Champion on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_ACTIVETRANSITION_H
#define HOMING_PIGEON_2_ACTIVETRANSITION_H

#include "Distribution.h"
#include <memory>
#include <torch/torch.h>

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::distributions {

    /**
     * Class representing a ActiveTransition distribution.
     */
    class ActiveTransition : public Distribution {
    public:
        //
        // Factory
        //

        /**
         * Create a ActiveTransition distribution.
         * @param param the parameters of the distribution
         * @return the created distribution
         */
        static std::unique_ptr<ActiveTransition> create(const torch::Tensor &param);

    public:
        //
        // Constructor
        //
        /**
         * Construct a ActiveTransition distribution.
         * @param param the parameters of the distribution
         */
        explicit ActiveTransition(const torch::Tensor &param);

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
        void updateParams(const torch::Tensor &p) override;

        /**
         * Compute the entropy of the distribution.
         * @return the entropy
         */
        double entropy() override;

    private:
        torch::Tensor param;
    };

}

#endif //HOMING_PIGEON_2_ACTIVETRANSITION_H
