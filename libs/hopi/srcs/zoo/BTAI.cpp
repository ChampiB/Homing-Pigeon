//
// Created by Theophile Champion on 03/08/2021.
//

#include "BTAI.h"
#include "algorithms/planning/MCTSConfig.h"
#include "algorithms/planning/MCTS.h"
#include "algorithms/planning/MCTSNodeData.h"
#include "algorithms/inference/VMP.h"
#include "distributions/Categorical.h"
#include "graphs/FactorGraph.h"
#include "nodes/VarNode.h"
#include "environments/Environment.h"
#include "api/API.h"

using namespace hopi::environments;
using namespace hopi::algorithms::planning;
using namespace hopi::algorithms::inference;
using namespace hopi::distributions;
using namespace hopi::api;
using namespace hopi::graphs;
using namespace hopi::nodes;
using namespace torch;

namespace hopi::zoo {

    std::shared_ptr<BTAI> BTAI::create(
            const Environment *env,
            const std::shared_ptr<MCTSConfig> &config,
            const Tensor &obs
    ) {
        return std::make_shared<BTAI>(env, config, obs);
    }

    BTAI::BTAI(
            const Environment *env,
            const std::shared_ptr<MCTSConfig> &config,
            const Tensor &obs
    ) {
        // Retrieve current factor graph.
        _fg = FactorGraph::current();

        // Retrieve model's parameters.
        _a = env->A();
        _b = env->B();
        _d = env->D();

        // Compute posterior beliefs over initial state
        VarNode *s = API::Categorical(_d);
        VarNode *o = API::Transition(s, _a);
        o->setType(VarNodeType::OBSERVED);
        o->setPosterior(Categorical::create(obs));

        // Set root of the tree.
        _fg->setTreeRoot(s);

        // Create the MCTS algorithm.
        _mcts = MCTS::create(config);
    }

    BTAI::~BTAI() {
        this->_mcts = nullptr;
        this->_fg = nullptr;
    }

    void BTAI::step(const std::shared_ptr<Environment> &env, const EvaluationType &type) {
        VMP::inference(_fg->getNodes());
        for (int j = 0; j < _mcts->config()->nbPlanningSteps(); ++j) {
            auto selectedNode = _mcts->selectNode(_fg->treeRoot(), env->actions());
            auto expandedNodes = _mcts->expansion(selectedNode, _a, _b);
            VMP::inference(expandedNodes);
            _mcts->evaluation(expandedNodes, _a, type);
            _mcts->propagation(expandedNodes);
        }

        int action = _mcts->selectAction(_fg->treeRoot());
        auto obs = env->execute(action);
        _fg->integrate(action, obs, _a, _b);
    }

}
