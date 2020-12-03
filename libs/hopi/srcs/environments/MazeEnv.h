//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_MAZEENV_H
#define HOMING_PIGEON_2_MAZEENV_H

#include "Environment.h"
#include <string>
#include <memory>
#include <Eigen/Dense>

namespace hopi::environments {

    class MazeEnv : public Environment {
    public:
        enum Action: int {
            UP    = 0,
            DOWN  = 1,
            LEFT  = 2,
            RIGHT = 3,
            IDLE  = 4
        };

    public:
        explicit MazeEnv(const std::string& file);
        int execute(int action) override;
        void print() const override;
        [[nodiscard]] int actions() const override;
        [[nodiscard]] int states() const override;
        [[nodiscard]] int observations() const override;
        Eigen::MatrixXd A() override;
        std::vector<Eigen::MatrixXd> B() override;
        Eigen::MatrixXd D() override;

    public:
        [[nodiscard]] std::pair<int,int> agentPosition() const;
        [[nodiscard]] std::pair<int,int> exitPosition() const;
        double& operator()(int row, int col);
        [[nodiscard]] int manhattan_distance(const std::pair<int,int>& pos) const;
        static int manhattan_distance(const std::pair<int,int>& agent, const std::pair<int,int>& exit);

    private:
        [[nodiscard]] std::pair<int,int> execute(int action, const std::pair<int,int>& pos) const;
        void loadStatesIndexes();

    private:
        std::pair<int,int> agent_pos;
        std::pair<int,int> exit_pos;
        Eigen::MatrixXd maze;
        Eigen::MatrixXi states_idx;
        int nb_states;
    };

}

#endif //HOMING_PIGEON_2_MAZEENV_H
