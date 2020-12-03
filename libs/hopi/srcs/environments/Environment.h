//
// Created by Theophile Champion on 28/11/2020.
//

#ifndef HOMING_PIGEON_ENVIRONMENT_H
#define HOMING_PIGEON_ENVIRONMENT_H

#include <torch/torch.h>

namespace hopi::environments {

    /**
     * Interface representing a general environment.
     */
    class Environment {
    public:
        /**
         * Execute an action in the environment.
         * @param action the action to be executed
         * @return the observation made after executing the action
         */
        virtual int execute(int action) = 0;

        /**
         * Display the environment.
         */
        virtual void print() const = 0;

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
        virtual torch::Tensor A() = 0;

        /**
         * Getter.
         * @return the true transition mapping
         */
        virtual torch::Tensor B() = 0;

        /**
         * Getter.
         * @return the true initial hidden states
         */
        virtual torch::Tensor D() = 0;
    };

}

#endif //HOMING_PIGEON_ENVIRONMENT_H
