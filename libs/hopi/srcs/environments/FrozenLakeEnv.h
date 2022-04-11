//
// Created by Theophile Champion on 28/11/2020.
//

#ifndef HOMING_PIGEON_FROZEN_LAKE_ENV_H
#define HOMING_PIGEON_FROZEN_LAKE_ENV_H

#include "Environment.h"
#include <string>
#include <memory>
#include <torch/torch.h>

namespace hopi::environments {

    /**
     * Class representing a environment where an agent evolve in a lake.
     */
    class FrozenLakeEnv : public Environment {
    public:
        //
        // Factory
        //

        /**
         * Create a lake environment.
         * @param file the name of the file from which the environment should be loaded
         * @return the environment
         */
        static std::unique_ptr<FrozenLakeEnv> create(const std::string &file);

        //
        // Constructor
        //
        /**
         * Construct a lake environment.
         * @param file the name of the file from which the environment should be loaded
         */
        explicit FrozenLakeEnv(const std::string &file);

    public:
        /**
         * The actions available to the agent in the frozen lake environment.
         */
        enum Action: int {
            UP    = 0,
            DOWN  = 1,
            LEFT  = 2,
            RIGHT = 3,
        };

    public:
        //
        // Implementation of the methods of the Environment class
        //

        /**
         * Reset the environment to its initial state.
         */
        torch::Tensor reset() override;

        /**
         * Execute an action in the environment.
         * @param action the action to be executed
         * @return the observation made after executing the action
         */
        torch::Tensor execute(int action) override;

        /**
         * Display the environment.
         */
        void print() override;

        /**
         * Getter.
         * @return the number of actions available to the agent
         */
        [[nodiscard]] int actions() const override;

        /**
         * Getter.
         * @return the number of states in the environment
         */
        [[nodiscard]] int states() const override;

        /**
         * Getter.
         * @return the number of observations in the environment
         */
        [[nodiscard]] int observations() const override;

        /**
         * Getter.
         * @return the true likelihood mapping
         */
        [[nodiscard]] torch::Tensor A() const override;

        /**
         * Getter.
         * @return the true transition mapping
         */
        [[nodiscard]] torch::Tensor B() const override;

        /**
         * Getter.
         * @return the true initial hidden states
         */
        [[nodiscard]] torch::Tensor D() const override;

        /**
         * Getter.
         * @param advanced should the prior preferences be advanced?
         * @return the prior preferences over observations
         */
        [[nodiscard]] torch::Tensor pref_states(bool advanced) const override;

        /**
         * Getter.
         * @return the prior preferences over observations
         */
        [[nodiscard]] torch::Tensor pref_obs() const override;

        /**
         * Getter.
         * @return true if the agent solved the environment false otherwise.
         */
        [[nodiscard]] bool solved() const override;

        /**
         * Getter.
         * @return environment's type.
         */
        [[nodiscard]] EnvType type() const override;

    public:
        /**
         * Getter.
         * @return the agent position
         */
        [[nodiscard]] std::pair<int,int> agentPosition() const;

        /**
         * Getter.
         * @return the exit position
         */
        [[nodiscard]] std::pair<int,int> exitPosition() const;

        /**
         * Getter.
         * @return the agent' score
         */
        [[nodiscard]] double agentScore() const;

        /**
         * Getter.
         * @param row the row index
         * @param col the column index
         * @return the value stored at position (row, col) in the lake, i.e., zero for an empty cell and one for a wall
         */
        double operator()(int row, int col);

        /**
         * Compute the Manhattan distance between the agent and the exit.
         * @param agent the agent position
         * @param exit the exit position
         * @return the Manhattan distance
         */
        static int manhattan_distance(const std::pair<int,int>& agent, const std::pair<int,int> &exit);

    private:

        /**
         * Simulate the execution of an action in the environment but does not modify the environment state.
         * @param action the action to perform
         * @param pos the current position of the agent
         * @return the new position of the agent
         */
        [[nodiscard]] std::pair<int,int> execute(int action, const std::pair<int,int> &pos) const;

        /**
         * Associate to each empty cell a state index, e.g., for the following lake
         * SFFF                            1  2  3  4
         * FHFH     we get the mapping     5  6  7  8
         * FFFH    ------------------->    9  10 11 12
         * HFFG                            13 14 15 16
         */
        void loadStatesIndexes();

    private:
        std::pair<int,int> agent_initial_pos;
        std::pair<int,int> agent_pos;
        std::pair<int,int> exit_pos;
        torch::Tensor lake;
        torch::Tensor states_idx;
        int nb_states;
        double agent_score;
    };

}

#endif //HOMING_PIGEON_FROZEN_LAKE_ENV_H
