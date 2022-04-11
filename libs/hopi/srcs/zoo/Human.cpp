//
// Created by Theophile Champion on 03/08/2021.
//

#include "Human.h"
#include "algorithms/planning/MCTSConfig.h"
#include "algorithms/planning/MCTSNodeData.h"
#include "environments/Environment.h"

using namespace hopi::environments;
using namespace hopi::algorithms::planning;
using namespace hopi::graphs;
using namespace hopi::nodes;
using namespace torch;

namespace hopi::zoo {

    std::shared_ptr<Human> Human::create() {
        return std::make_shared<Human>();
    }

    Human::Human() = default;

    void Human::step(const std::shared_ptr<Environment> &env, const EvaluationType &type) {
        std::cout << "What action do you want to play? ";
        std::string action_str;
        std::getline(std::cin, action_str);
        int action = std::stoi(action_str);
        env->execute(action);
    }

}
