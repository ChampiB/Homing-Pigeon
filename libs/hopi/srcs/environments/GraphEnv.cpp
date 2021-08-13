//
// Created by Theophile Champion on 30/07/2021.
//

#include "GraphEnv.h"
#include "math/Ops.h"
#include "api/API.h"
#include <numeric>
#include <torch/torch.h>

using namespace hopi::math;
using namespace hopi::api;
using namespace torch;

namespace hopi::environments {

    std::unique_ptr<GraphEnv> GraphEnv::create(int nGood, int nBad, const std::vector<int> &sizeGoodPaths) {
        return std::make_unique<GraphEnv>(nGood, nBad, sizeGoodPaths);
    }

    GraphEnv::GraphEnv(int nGood, int nBad, std::vector<int> sizeGoodPaths) {
        _verbose = false;
        agent_state = 0;
        n_states = 2 + std::accumulate(sizeGoodPaths.begin(), sizeGoodPaths.end(), 0);
        breadth = nGood + nBad;
        n_good = nGood;
        n_bad = nBad;
        paths = initialisePaths(sizeGoodPaths);
        // Compute the size of the longest path
        longest_path_size = 0;
        for (auto &path : paths) {
            if (path.size() > longest_path_size)
                longest_path_size = (int)path.size();
        }
    }

    std::vector<std::vector<int>> GraphEnv::initialisePaths(const std::vector<int> &sizeGoodPaths) {
        std::vector<std::vector<int>> pathsStates;
        int state_id = 2;

        for (int size_good_path : sizeGoodPaths) {
            std::vector<int> pathState;
            for (int j = 0; j < size_good_path; ++j) {
                pathState.push_back(state_id);
                ++state_id;
            }
            pathsStates.push_back(pathState);
        }
        return pathsStates;
    }

    torch::Tensor GraphEnv::reset() {
        agent_state = 0;
        return Ops::one_hot(observations(), ObsType::GOOD);
    }

    torch::Tensor GraphEnv::execute(int action) {
        if (action < 0 or action >= actions())
            throw std::runtime_error("GraphEnv: invalid action.");

        agent_state = execute(action, agent_state);
        return Ops::one_hot(observations(), (agent_state == 1) ? ObsType::BAD : ObsType::GOOD);
    }

    std::vector<std::string> GraphEnv::getPathsStates() const {
        std::vector<std::string> paths_states;

        for (auto &path : paths) {
            std::string path_state = " ";
            for (int s : path) {
                path_state.append(std::to_string(s) + " ");
            }
            paths_states.push_back(path_state);
        }
        return paths_states;
    }

    std::vector<std::string> GraphEnv::getPathsName(std::vector<std::string> &paths_states) const {
        std::vector<std::string> paths_name;

        for (int i = 0; i < paths.size(); ++i) {
            std::string path_name = " Path_" + std::to_string(i) + " ";
            long diff = (int)path_name.length() - (int)paths_states[i].length();

            if (diff < 0)
                path_name.append(std::string(-diff, ' '));
            else if (diff > 0)
                paths_states[i].append(std::string(diff, ' '));
            paths_name.push_back(path_name);
        }
        return paths_name;
    }

    void GraphEnv::printHelp() const {
        std::cout << "The agent starts in state 0, i.e. the initial state." << std::endl;
        std::cout << "State 1 is an absorbing state that should be avoided by the agent." << std::endl;
        std::cout << "At each time step the agent must pick from a set of " << std::to_string(actions()) + " actions." << std::endl;
        std::cout << "In the initial states: "  << std::endl;
        std::cout << " - " << std::to_string(n_bad) << " actions lead to the bad state" << std::endl;
        std::cout << " - " << std::to_string(n_good) << " actions lead to good paths" << std::endl;
        std::cout << "When engaged on a path, there is no way back and only one action keep the agent on that path." << std::endl;
        std::cout << "The " + std::to_string(breadth - 1) + " other actions lead the agent to the bad state." << std::endl;
        std::cout << "Ideally the agent would pick the longest of the good path at the beginning at stick to it."  << std::endl;
        std::cout << "Finally, the last state of the longest of the good paths is the (absorbing) goal state."  << std::endl;
        std::cout << std::endl;
    }

    void GraphEnv::print() const {
        // Create the strings containing the states of each good path.
        std::vector<std::string> paths_states = getPathsStates();

        // Create the string path name, i.e. Path_N where N is the path index.
        std::vector<std::string> paths_name = getPathsName(paths_states);

        // Create the separator line.
        std::string sep_line = "#";
        for (auto &name : paths_name) {
            sep_line.append(std::string(name.length(), '='));
            sep_line.append("#");
        }

        // Describe the environment precisely.
        if (_verbose)
            printHelp();

        // Display the table indication which state the agent is in.
        std::cout << "                 " << sep_line << std::endl;
        std::cout << "                 |";
        for (auto &name : paths_name) {
            std::cout << name << "|";
        }
        std::cout << std::endl;
        std::cout << "#========#=======" << sep_line << std::endl;
        std::cout << "| States | 0 | 1 |";
        for (auto &states : paths_states) {
            std::cout << states << "|";
        }
        std::cout << std::endl;
        std::cout << "#========#=======" << sep_line << std::endl;
        std::cout << std::endl;
        std::cout << "Agent is in state: " << std::to_string(agent_state) << std::endl;
        std::cout << std::endl;
    }

    int GraphEnv::actions() const {
        return breadth;
    }

    int GraphEnv::states() const {
        return n_states;
    }

    int GraphEnv::observations() const {
        return 2;
    }

    torch::Tensor GraphEnv::A() const {
        Tensor A = API::full({observations(), states()}, 0.1 / (observations() - 1));

        for (int s = 0; s < states(); ++s) {
            int obs = (s == 1) ? ObsType::BAD : ObsType::GOOD;
            A[obs][s] = 0.9;
        }
        return A;
    }

    torch::Tensor GraphEnv::B() const {
        Tensor B = API::full({states(), states(), actions()}, 0.1 / (states() - 1));

        for (int s = 0; s < states(); ++s) {
            for (int a = 0; a < actions(); ++a) {
                B[execute(a, s)][s][a] = 0.9;
            }
        }
        return B;
    }

    torch::Tensor GraphEnv::D() const {
        Tensor D = API::full({states()}, 0.1 / (states() - 1));

        D[0] = 0.9;
        return D;
    }

    void GraphEnv::verbose() {
        _verbose = true;
    }

    int GraphEnv::goalState() const {
        for (auto &path : paths) {
            if (path.size() == longest_path_size)
                return path[longest_path_size - 1];
        }
        assert(false && "Impossible to retrieve goal state in GraphEnv::goalState().");
        return -1;
    }

    int GraphEnv::agentState() const {
        return agent_state;
    }

    int GraphEnv::execute(int action, int state) const {
        // If initial state and good action selected.
        if (state == 0 && action < n_good) {
            // Go to the first state of the path corresponding to the "action".
            return paths[action][0];
        }
        // If agent is in initial state and selected a bad action or the agent is in the bad state already.
        if (state == 0 or state == 1) {
            // Go to bad state.
            return 1;
        }
        // If already engaged in a path and select bad action.
        if (action > 0) {
            // Go to bad state.
            return 1;
        }
        // If already engaged in a path and select good action.
        for (auto &path : paths) {
            for (int i = 0; i < path.size(); ++i) {
                if (state == path[i]) {
                    if (i + 1 != path.size())
                        return path[i + 1];
                    else
                        return (longest_path_size == path.size()) ? path[i] : 1;
                }
            }
        }
        std::string msg = "Invalid action or state, when calling GraphEnv::execute("\
                        + std::to_string(action) + ", " + std::to_string(state) + ")";
        throw std::runtime_error(msg);
    }

    bool GraphEnv::solved() const {
        return agentState() == goalState() or agentState() == 1;
    }

    EnvType GraphEnv::type() const {
        return GRAPH;
    }

}
