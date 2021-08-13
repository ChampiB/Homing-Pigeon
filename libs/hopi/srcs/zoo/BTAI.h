//
// Created by Theophile Champion on 03/08/2021.
//

#ifndef EXPERIMENTS_AI_TS_BTAI_V2_H
#define EXPERIMENTS_AI_TS_BTAI_V2_H

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
}
namespace hopi::nodes {
    class VarNode;
}

namespace hopi::zoo {

    class BTAI {
    public:
        /**
         * Create a Partially Observable Markov Decision Process.
         * @param env the environment.
         * @param config the configuration of the tree search.
         * @param obs the initial observation.
         * @return the BTAI agent.
         */
        static std::shared_ptr<BTAI> create(
            const environments::Environment *env,
            const std::shared_ptr<algorithms::planning::MCTSConfig> &config,
            const torch::Tensor &obs
        );

        /**
         * Create a Partially Observable Markov Decision Process.
         * @param env the environment.
         * @param config the configuration of the tree search.
         * @param obs the initial observation.
         */
        BTAI(
            const environments::Environment *env,
            const std::shared_ptr<algorithms::planning::MCTSConfig> &config,
            const torch::Tensor &obs
        );

        /**
         * Execute on step of the action perception cycle in the environment.
         * @param env the environment to act in.
         * @param type the type of evaluation to use during planning.
         */
        void step(
                const std::shared_ptr<environments::Environment> &env,
                const algorithms::planning::EvaluationType &type
        );

    private:
        torch::Tensor _a;
        torch::Tensor _b;
        torch::Tensor _c;
        torch::Tensor _d;

        std::unique_ptr<algorithms::planning::MCTS> _mcts;
        std::shared_ptr<graphs::FactorGraph> _fg;
    };

}

#endif //EXPERIMENTS_AI_TS_BTAI_V2_H
