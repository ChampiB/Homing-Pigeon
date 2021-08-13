//
// Created by Theophile Champion on 30/07/2021.
//

#ifndef HOMING_PIGEON_GRAPH_ENV_H
#define HOMING_PIGEON_GRAPH_ENV_H

#include "Environment.h"

namespace hopi::environments {

    class GraphEnv : public Environment {
    public:
        /**
         * The two type of observation that can be made by the agent.
         */
        enum ObsType {
            GOOD = 0,
            BAD  = 1
        };

    public:
        /**
         * Create a new graph environment.
         * @param nGood the number of good paths
         * @param nBad the number of bad paths
         * @param sizeGoodPaths the size of each good path
         * @return the created graph environment
         */
        static std::unique_ptr<GraphEnv> create(int nGood, int nBad, const std::vector<int> &sizeGoodPaths);

        /**
         * Construct the graph environment.
         * @param nGood the number of good path after the initial state
         * @param nBad the number of bad path after the initial state
         * @param sizeGoodPaths the size of each good paths
         */
        GraphEnv(int nGood, int nBad, std::vector<int> sizeGoodPaths);

        /**
         * Reset the environment to its initial state.
         * @return the initial observation
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
        void print() const override;

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
         * @return true if the agent solved the environment false otherwise.
         */
        [[nodiscard]] bool solved() const override;

        /**
         * Getter.
         * @return environment's type.
         */
        [[nodiscard]] EnvType type() const override;

        /**
         * Make the function "print" verbose.
         */
        void verbose();

        /**
         * Getter.
         * @return the index of the goal
         */
        [[nodiscard]] int goalState() const;

        /**
         * Getter.
         * @return the state in which the agent is located
         */
        [[nodiscard]] int agentState() const;

    private:
        [[nodiscard]] std::vector<std::string> getPathsStates() const;
        [[nodiscard]] std::vector<std::string> getPathsName(std::vector<std::string> &paths_states) const;
        void printHelp() const;
        [[nodiscard]] int execute(int action, int agent_state) const;
        static std::vector<std::vector<int>> initialisePaths(const std::vector<int> &sizeGoodPaths);

    private:
        int agent_state;
        int n_states;
        int n_good;
        int n_bad;
        int breadth;
        std::vector<std::vector<int>> paths;
        bool _verbose;
        int longest_path_size;
    };

}

#endif //HOMING_PIGEON_GRAPH_ENV_H
