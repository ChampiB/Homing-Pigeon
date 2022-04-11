//
// Created by Theophile Champion on 28/11/2020.
//

#ifndef HOMING_PIGEON_ENVIRONMENT_H
#define HOMING_PIGEON_ENVIRONMENT_H

#include <torch/torch.h>
#include "EnvType.h"

namespace hopi::environments {

    /**
     * Interface representing a general environment.
     */
    class Environment {
    public:
        /**
         * Reset the environment to its initial state.
         * @return the initial observation
         */
        virtual torch::Tensor reset() = 0;

        /**
         * Execute an action in the environment.
         * @param action the action to be executed
         * @return the observation made after executing the action
         */
        virtual torch::Tensor execute(int action) = 0;

        /**
         * Display the environment.
         */
        virtual void print() = 0;

        /**
         * Getter.
         * @return the number of actions available to the agent
         */
        [[nodiscard]] virtual int actions() const = 0;

        /**
         * Getter.
         * @return the number of states in the environment
         */
        [[nodiscard]] virtual int states() const = 0;

        /**
         * Getter.
         * @return the number of observations in the environment
         */
        [[nodiscard]] virtual int observations() const = 0;

        /**
         * Getter.
         * @return the true likelihood mapping
         */
        [[nodiscard]] virtual torch::Tensor A() const = 0;

        /**
         * Getter.
         * @return the true transition mapping
         */
        [[nodiscard]] virtual torch::Tensor B() const = 0;

        /**
         * Getter.
         * @return the true initial hidden states
         */
        [[nodiscard]] virtual torch::Tensor D() const = 0;

        /**
         * Getter.
         * @param advanced should the prior preferences be advanced?
         * @return the prior preferences over observations
         */
        [[nodiscard]] virtual torch::Tensor pref_states(bool advanced) const = 0;

        /**
         * Getter.
         * @return the prior preferences over observations
         */
        [[nodiscard]] virtual torch::Tensor pref_obs() const = 0;

        /**
         * Getter.
         * @return environment's type.
         */
        [[nodiscard]] virtual EnvType type() const = 0;

        /**
         * Getter.
         * @return true if the agent solved the environment false otherwise.
         */
        [[nodiscard]] virtual bool solved() const = 0;
    };

}

#endif //HOMING_PIGEON_ENVIRONMENT_H
