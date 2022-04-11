//
// Created by Theophile Champion on 28/11/2020.
//

#include "FrozenLakeEnv.h"
#include "api/API.h"
#include "math/Ops.h"
#include <iostream>

using namespace torch;
using namespace hopi::api;
using namespace hopi::math;

namespace hopi::environments {

    std::unique_ptr<FrozenLakeEnv> FrozenLakeEnv::create(const std::string &file) {
        return std::make_unique<FrozenLakeEnv>(file);
    }

    FrozenLakeEnv::FrozenLakeEnv(const std::string &file) {
        std::pair<int,int> lake_size;
        std::ifstream input(file);
        std::string line;

        if (!input.is_open())
            throw std::runtime_error("In FrozenLake::FrozenLake(...): Could not open lake's file.");

        agent_pos = {-1,-1};
        exit_pos = {-1,-1};

        input >> lake_size.first >> lake_size.second;
        getline(input, line); // return "\n"

        nb_states = lake_size.first * lake_size.second;
        lake = API::empty({lake_size.first, lake_size.second});
        for (int i = 0; i < lake_size.first; ++i) {
            getline(input, line);
            for (int j = 0; j < line.length() && j < lake_size.second; ++j) {
                switch (line[j]) {
                    case 'H':
                        lake[i][j] = 1;
                        break;
                    case 'F':
                        lake[i][j] = 0;
                        break;
                    case 'G':
                        lake[i][j] = 0;
                        exit_pos.first = i;
                        exit_pos.second = j;
                        break;
                    case 'S':
                        lake[i][j] = 0;
                        agent_pos.first = i;
                        agent_pos.second = j;
                        break;
                    default:
                        throw std::runtime_error("Invalid file format: '" + file + "'");
                }
            }
            for (int j = (int) line.length(); j < lake_size.second; ++j) {
                lake[i][j] = 1;
            }
        }
        agent_initial_pos = agent_pos;
        agent_score = 0;

        loadStatesIndexes();
    }

    torch::Tensor FrozenLakeEnv::reset() {
        agent_pos = agent_initial_pos;
        agent_score = 0;
        return Ops::one_hot(observations(), states_idx[agent_pos.first][agent_pos.second].item<int>());
    }

    torch::Tensor FrozenLakeEnv::execute(int action) {
        // Execute the agent's action.
        agent_pos = execute(action, agent_pos);

        // If the agent fell in a hole, its score goes down.
        if (lake[agent_pos.first][agent_pos.second].item<double>() == 1)
            agent_score -= 1;

        // If the agent reaches the frisbee, its score goes up.
        if (agent_pos.first == exit_pos.first && agent_pos.second == exit_pos.second)
            agent_score += 10;

        return Ops::one_hot(observations(), states_idx[agent_pos.first][agent_pos.second].item<int>());
    }

    void FrozenLakeEnv::print() {
        std::cout << "Current score: " << agent_score << std::endl;
        for (int i = 0; i < lake.size(0); ++i) {
            for (int j = 0; j < lake.size(1); ++j) {
                if (agent_pos.first == i && agent_pos.second == j)
                    std::cout << "A";
                else if (exit_pos.first == i && exit_pos.second == j)
                    std::cout << "G";
                else if (lake[i][j].item<double>() == 0)
                    std::cout << ".";
                else
                    std::cout << "H";
            }
            std::cout << std::endl;
        }
        std::cout << "A = agent position" << std::endl;
        std::cout << "G = goal position" << std::endl;
        std::cout << "H = hole" << std::endl;
    }

    int FrozenLakeEnv::actions() const {
        return 4;
    }

    int FrozenLakeEnv::observations() const {
        return nb_states;
    }

    int FrozenLakeEnv::states() const {
        return nb_states;
    }

    std::pair<int, int> FrozenLakeEnv::agentPosition() const {
        return agent_pos;
    }

    std::pair<int, int> FrozenLakeEnv::exitPosition() const {
        return exit_pos;
    }

    double FrozenLakeEnv::operator()(int row, int col) {
        return lake[row][col].item<double>();
    }

    void FrozenLakeEnv::loadStatesIndexes() {
        int state_id = 0;

        states_idx = API::full(lake.sizes(), -1).to(kInt);
        for (int j = 0; j < lake.size(0); ++j) {
            for (int i = 0; i < lake.size(1); ++i) {
                states_idx[j][i] = state_id;
                ++state_id;
            }
        }
    }

    Tensor FrozenLakeEnv::A() const {
        double noise = 0.01;
        double epsilon = noise / (observations() - 1);
        Tensor res = API::full({observations(), observations()}, epsilon);

        for (int i = 0; i < observations(); ++i) {
            res[i][i] = 1 - noise;
        }

        return res;
    }

    Tensor FrozenLakeEnv::B() const {
        Tensor B = API::full({states(), states(), actions()}, 0.1 / (states() - 1));

        for (int i = 0; i < lake.size(1); ++i) {
            for (int j = 0; j < lake.size(0); ++j) {
                if (lake[j][i].item<double>() == 0) {
                    auto current_pos = std::make_pair(j, i);
                    for (int k = 0; k < actions(); ++k) {
                        auto dest_pos = execute(k, current_pos);
                        B[states_idx[dest_pos.first][dest_pos.second]][states_idx[j][i]][k] = 0.9;
                    }
                }
            }
        }
        return B;
    }

    Tensor FrozenLakeEnv::D() const {
        Tensor D = API::full({states()}, 0.1 / (states() - 1));

        D[states_idx[agent_pos.first][agent_pos.second]] = 0.9;
        return D;
    }

    torch::Tensor FrozenLakeEnv::pref_states(bool advanced) const {
        if (advanced)
            throw std::invalid_argument("Advanced prior are not supported in graph environment.");
        return Ops::uniform({states()});
    }

    torch::Tensor FrozenLakeEnv::pref_obs() const {
        double penalty = -10;
        Tensor C = API::zeros({states()});

        for (int i = 0; i < lake.size(1); ++i) {
            for (int j = 0; j < lake.size(0); ++j) {
                C[states_idx[j][i]] = lake[j][i] * penalty \
                 - manhattan_distance({j,i}, exit_pos);
            }
        }
        return C;
    }

    int FrozenLakeEnv::manhattan_distance(const std::pair<int,int>& agent, const std::pair<int,int>& exit) {
        return std::abs(agent.first - exit.first) + std::abs(agent.second - exit.second);
    }

    std::pair<int, int> FrozenLakeEnv::execute(int action, const std::pair<int, int> &pos) const {
        std::pair res = pos;
        auto maze_a = lake.accessor<double,2>();

        switch (action) {
            case UP:
                if (res.first - 1 >= 0)
                    res.first -= 1;
                break;
            case DOWN:
                if (res.first + 1 < lake.size(0))
                    res.first += 1;
                break;
            case LEFT:
                if (res.second - 1 >= 0)
                    res.second -= 1;
                break;
            case RIGHT:
                if (res.second + 1 < lake.size(1))
                    res.second += 1;
                break;
            default:
                assert(false && "FrozenLakeEnv::execute, unsupported action.");
        }
        return res;
    }

    bool FrozenLakeEnv::solved() const {
        auto agent = agentPosition();
        auto exit = exitPosition();
        return agent.first == exit.first && agent.second == exit.second;
    }

    EnvType FrozenLakeEnv::type() const {
        return FROZEN_LAKE;
    }

    double FrozenLakeEnv::agentScore() const {
        return agent_score;
    }

}
