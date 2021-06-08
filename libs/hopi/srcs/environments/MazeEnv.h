//
// Created by Theophile Champion on 28/11/2020.
//

#ifndef HOMING_PIGEON_MAZE_ENV_H
#define HOMING_PIGEON_MAZE_ENV_H

#include "Environment.h"
#include <string>
#include <memory>
#include <torch/torch.h>

namespace hopi::environments {

    /**
     * Class representing a environment where an agent evolve in a maze.
     */
    class MazeEnv : public Environment {
    public:
        //
        // Factory
        //

        /**
         * Create a maze environment.
         * @param file the name of the file from which the environment should be loaded
         * @return the environment
         */
        static std::unique_ptr<MazeEnv> create(const std::string &file);

        //
        // Constructor
        //
        /**
         * Construct a maze environment.
         * @param file the name of the file from which the environment should be loaded
         */
        explicit MazeEnv(const std::string &file);

    public:
        //
        // Possible action available to the agent
        //
        enum Action: int {
            UP    = 0,
            DOWN  = 1,
            LEFT  = 2,
            RIGHT = 3,
            IDLE  = 4
        };

    public:
        //
        // Implementation of the methods of the Distribution class
        //

        /**
         * Execute an action in the environment.
         * @param action the action to be executed
         * @return the observation made after executing the action
         */
        int execute(int action) override;

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
        torch::Tensor A() override;

        /**
         * Getter.
         * @return the true transition mapping
         */
        torch::Tensor B() override;

        /**
         * Getter.
         * @return the true initial hidden states
         */
        torch::Tensor D() override;

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
         * @param row the row index
         * @param col the column index
         * @return the value stored at position (row, col) in the maze, i.e., zero for an empty cell and one for a wall
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
         * Compute the Manhattan distance between the agent and the maze's exit.
         * @param agent the agent position
         * @return the Manhattan distance
         */
        [[nodiscard]] int manhattan_distance(const std::pair<int,int> &pos) const;

        /**
         * Simulate the execution of an action in the environment but does not modify the environment state.
         * @param action the action to perform
         * @param pos the current position of the agent
         * @return the new position of the agent
         */
        [[nodiscard]] std::pair<int,int> execute(int action, const std::pair<int,int> &pos) const;

        /**
         * Associate to each empty cell a state index, e.g., for the following maze
         * WWWW                            WWWW
         * W  W     we get the mapping     W12W
         * W WW    ------------------->    W3WW
         * W  W                            W45W
         * WWWW                            WWWW
         * .
         */
        void loadStatesIndexes();

    private:
        std::pair<int,int> agent_pos;
        std::pair<int,int> exit_pos;
        torch::Tensor maze;
        torch::Tensor states_idx;
        int nb_states;
    };

}

#endif //HOMING_PIGEON_MAZE_ENV_H
