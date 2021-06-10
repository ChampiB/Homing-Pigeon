//
// Created by Theophile Champion on 06/01/2021.
//

#ifndef HOMING_PIGEON_DIRICHLET_H
#define HOMING_PIGEON_DIRICHLET_H

#include "Distribution.h"
#include <memory>
#include <torch/torch.h>

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::distributions {

    /**
     * Class representing a Dirichlet distribution.
     */
    class Dirichlet : public Distribution {
    public:
        //
        // Factories
        //

        /**
         * Create a Dirichlet distribution.
         * @param p the parameters of the distribution
         * @return the created distribution
         */
        static std::unique_ptr<Dirichlet> create(const torch::Tensor &p);

        /**
         * Create a Dirichlet distribution.
         * @param p the parameters of the distribution
         * @return the created distribution
         */
        static std::unique_ptr<Dirichlet> create(const std::shared_ptr<torch::Tensor> &p);

    public:
        //
        // Constructors
        //

        /**
         * Construct a Dirichlet distribution.
         * @param p the parameters of the distribution
         */
        explicit Dirichlet(const torch::Tensor &param);

        /**
         * Construct a Dirichlet distribution.
         * @param p the parameters of the distribution
         */
        explicit Dirichlet(const std::shared_ptr<torch::Tensor> &param);

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

    public:
        /**
         * Compute the expectation of the logarithm of x
         * @param p the parameter of the Dirichlet distribution
         * @return the expectation of the logarithm of x
         */
        static torch::Tensor expectedLog(const torch::Tensor &p);

    private:
        /**
         * Compute the entropy of a 1D Dirichlet whose parameters are given by "p".
         * @param p the parameters of a 1D Dirichlet
         * @return the entropy of the Dirichlet
         */
        static double entropy(const torch::Tensor &&p);

    private:
        std::shared_ptr<torch::Tensor> param;
    };

}

#endif //HOMING_PIGEON_DIRICHLET_H
