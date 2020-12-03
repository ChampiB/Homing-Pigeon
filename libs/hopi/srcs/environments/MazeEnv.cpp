//
// Created by tmac3 on 28/11/2020.
//

#include "MazeEnv.h"
#include <fstream>
#include <iostream>

using namespace Eigen;

namespace hopi::environments {

    MazeEnv::MazeEnv(const std::string& file) {
        std::pair<int,int> maze_size;
        std::ifstream input(file);
        std::string line;

        agent_pos = std::make_pair(-1,-1);
        exit_pos = std::make_pair(-1,-1);
        nb_states = 0;

        input >> maze_size.first >> maze_size.second;
        getline(input, line); // return "\n"
        maze = MatrixXd(maze_size.first, maze_size.second);
        for (int i = 0; i < maze_size.first; ++i) {
            getline(input, line);
            for (int j = 0; j < line.length() && j < maze_size.second; ++j) {
                switch (line[j]) {
                    case 'W':
                        maze(i,j) = 1;
                        break;
                    case '.':
                        nb_states += 1;
                        maze(i,j) = 0;
                        break;
                    case 'E':
                        nb_states += 1;
                        maze(i,j) = 0;
                        exit_pos.first = i;
                        exit_pos.second = j;
                        break;
                    case 'S':
                        nb_states += 1;
                        maze(i,j) = 0;
                        agent_pos.first = i;
                        agent_pos.second = j;
                        break;
                    default:
                        throw std::runtime_error("Invalid file format: '" + file + "'");
                }
            }
            for (int j = line.length(); j < maze_size.second; ++j) {
                maze(i,j) = 1;
            }
        }
    }

    int MazeEnv::execute(int action) {
        agent_pos = execute(action, agent_pos);
        return manhattan_distance(agent_pos);
    }

    void MazeEnv::print() const {
        for (int i = 0; i < maze.rows(); ++i) {
            for (int j = 0; j < maze.cols(); ++j) {
                if (agent_pos.first == i && agent_pos.second == j)
                    std::cout << "A";
                else if (exit_pos.first == i && exit_pos.second == j)
                    std::cout << "E";
                else if (maze(i,j) == 0)
                    std::cout << " ";
                else
                    std::cout << "W";
            }
            std::cout << std::endl;
        }
        std::cout << "A = agent position" << std::endl;
        std::cout << "E = exit position" << std::endl;
        std::cout << "W = wall" << std::endl;
    }

    int MazeEnv::manhattan_distance(const std::pair<int,int>& pos) const {
        return manhattan_distance(pos, exit_pos);
    }

    int MazeEnv::manhattan_distance(const std::pair<int,int>& agent, const std::pair<int,int>& exit) {
        return std::abs(agent.first - exit.first) + std::abs(agent.second - exit.second);
    }

    int MazeEnv::actions() const {
        return 5;
    }

    int MazeEnv::observations() const {
        return maze.rows() + maze.cols() - 5;
    }

    int MazeEnv::states() const {
        return nb_states;
    }

    std::pair<int, int> MazeEnv::agentPosition() const {
        return agent_pos;
    }

    std::pair<int, int> MazeEnv::exitPosition() const {
        return exit_pos;
    }

    double &MazeEnv::operator()(int row, int col) {
        return maze(row, col);
    }

    void MazeEnv::loadStatesIndexes() {
        int state_id = 0;

        states_idx = MatrixXi::Constant(maze.rows(), maze.cols(), -1);
        for (int j = 0; j < maze.rows(); ++j) {
            for (int i = 0; i < maze.cols(); ++i) {
                if (maze(j, i) == 0) {
                    states_idx(j, i) = state_id;
                    ++state_id;
                }
            }
        }
    }

    MatrixXd MazeEnv::A() {
        if (states_idx.size() == 0) {
            loadStatesIndexes();
        }
        MatrixXd A = MatrixXd::Constant(observations(), states(), 0.1 / (observations() - 1));

        for (int i = 0; i < maze.cols(); ++i) {
            for (int j = 0; j < maze.rows(); ++j) {
                if (maze(j, i) == 0) {
                    A(manhattan_distance(std::make_pair(j, i)), states_idx(j, i)) = 0.9;
                }
            }
        }
        return A;
    }

    std::vector<MatrixXd> MazeEnv::B() {
        if (states_idx.size() == 0) {
            loadStatesIndexes();
        }
        std::vector<MatrixXd> B;

        for (int i = 0; i < actions(); ++i) {
            B.emplace_back(MatrixXd::Constant(states(), states(), 0.1 / (states() - 1)));
        }
        for (int i = 0; i < maze.cols(); ++i) {
            for (int j = 0; j < maze.rows(); ++j) {
                if (maze(j, i) == 0) {
                    auto current_pos = std::make_pair(j, i);
                    for (int k = 0; k < actions(); ++k) {
                        auto dest_pos = execute(k, current_pos);
                        B[k](states_idx(dest_pos.first, dest_pos.second), states_idx(j, i)) = 0.9;
                    }
                }
            }
        }
        return B;
    }

    MatrixXd MazeEnv::D() {
        if (states_idx.size() == 0) {
            loadStatesIndexes();
        }
        MatrixXd D = MatrixXd::Constant(states(), 1, 0.1 / (states() - 1));

        D(states_idx(agent_pos.first, agent_pos.second), 0) = 0.9;
        return D;
    }

    std::pair<int, int> MazeEnv::execute(int action, const std::pair<int, int> &pos) const {
        std::pair<int, int> res = pos;

        switch (action) {
            case UP:
                if (res.first - 1 >= 0 && maze(res.first - 1, res.second) == 0)
                    res.first -= 1;
                break;
            case DOWN:
                if (res.first + 1 < maze.rows() && maze(res.first + 1, res.second) == 0)
                    res.first += 1;
                break;
            case LEFT:
                if (res.second - 1 >= 0 && maze(res.first, res.second - 1) == 0)
                    res.second -= 1;
                break;
            case RIGHT:
                if (res.second + 1 < maze.cols() && maze(res.first, res.second + 1) == 0)
                    res.second += 1;
                break;
            case IDLE:
                break;
            default:
                throw std::runtime_error("Invalid action.");
        }
        return res;
    }

}