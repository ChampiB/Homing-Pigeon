//
// Created by Theophile Champion on 03/08/2021.
//

#ifndef EXPERIMENTS_AI_TS_HUMAN_H
#define EXPERIMENTS_AI_TS_HUMAN_H

#include <memory>
#include <torch/torch.h>
#include "algorithms/planning/EvaluationType.h"

namespace hopi::environments {
    class Environment;
}
namespace hopi::graphs {
    class FactorGraph;
}
namespace hopi::algorithms::planning {
    class MCTSConfig;
    class MCTS;
    class MCTSv2;
}
namespace hopi::nodes {
    class VarNode;
}

namespace hopi::zoo {

    /**
     * Class allowing a human to play.
     */
    class Human {
    public:
        /**
         * Create a agent which allows a human to play.
         * @return the human agent.
         */
        static std::shared_ptr<Human> create();

        /**
         * Create a agent which allows a human to play.
         */
        Human();

        /**
         * Execute on step of the action perception cycle in the environment.
         * @param env the environment to act in.
         * @param type the type of evaluation to use during planning.
         */
        static void step(
                const std::shared_ptr<environments::Environment> &env,
                const algorithms::planning::EvaluationType &type
        );
    };

}

#endif //EXPERIMENTS_AI_TS_HUMAN_H
