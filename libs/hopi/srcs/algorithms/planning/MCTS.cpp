//
// Created by Theophile Champion on 03/08/2021.
//

#include "MCTS.h"
#include <torch/torch.h>
#include "nodes/VarNode.h"
#include "nodes/FactorNode.h"
#include "distributions/Distribution.h"
#include "algorithms/planning/MCTSNodeData.h"
#include "algorithms/inference/VMP.h"
#include "api/API.h"
#include "math/Ops.h"
#include "MCTSConfig.h"
#include "EvaluationType.h"

using namespace torch;
using namespace hopi::api;
using namespace hopi::math;
using namespace hopi::distributions;
using namespace hopi::nodes;

namespace hopi::algorithms::planning {

    std::unique_ptr<MCTS> MCTS::create(const std::shared_ptr<MCTSConfig> &config) {
        return std::make_unique<MCTS>(config);
    }

    MCTS::MCTS(const std::shared_ptr<MCTSConfig> &config) {
        _config = config;
    }

    VarNode *MCTS::selectNode(VarNode *root, int nbActions) const {
        auto compUCT = [this](FactorNode *n1, FactorNode *n2) {
            return this->compareUCT(n1->child(), n2->child());
        };
        VarNode *curr = root;

        while (curr->nChildrenHiddenStates() == nbActions) {
            std::vector<FactorNode*> copy;
            std::copy_if(
                    curr->firstChild(), curr->lastChild(), std::back_inserter(copy),
                    [](FactorNode *n){return n->child()->data()->action != -1;}
            );
            curr = (*std::max_element(copy.begin(), copy.end(), compUCT))->child();
        }
        return curr;
    }

    std::vector<VarNode*> MCTS::expansion(VarNode *node, const torch::Tensor &a, const torch::Tensor &b) {
        std::vector<VarNode*> expandedNodes;

        for (int action = 0; action < b.size(2); ++action) {
            // Create future hidden states
            auto b_action = squeeze(torch::narrow(b, 2, action, 1));
            VarNode *s = API::Transition(node, b_action);
            s->data()->action = action;
            s->data()->cost = 0;
            s->data()->visits = 1;
            VarNode *o = API::Transition(s, a);
            // Add state and observation to list of expanded nodes
            expandedNodes.push_back(s);
            expandedNodes.push_back(o);
        }
        return expandedNodes;
    }

    void MCTS::evaluation(const std::vector<VarNode*> &nodes, const torch::Tensor &a, const EvaluationType &type) {
        static std::map<EvaluationType, EvaluationFunction> eFunctions = {
                {EFE, &MCTS::efe},
                {DOUBLE_KL, &MCTS::doubleKL}
        };

        if (eFunctions.find(type) == eFunctions.end())
            throw std::runtime_error("In MCTS::evaluation, unsupported evaluation type.");
        for (int i = 0; i < nodes.size(); i += 2) {
            auto sBeliefs = nodes[i]->posterior()->params();
            auto oBeliefs = nodes[i + 1]->posterior()->params();
            nodes[i]->data()->cost = eFunctions[type](sBeliefs, oBeliefs, a, _config);
        }
    }

    void MCTS::propagation(const std::vector<VarNode*> &nodes) {
        std::vector<VarNode*> copy;
        std::copy_if (
                nodes.begin(), nodes.end(), std::back_inserter(copy),
                [](VarNode *n){return n->data()->action != -1;}
        );
        auto bestChild = *std::min_element(copy.begin(), copy.end(), &MCTS::compareCost);
        double cost = bestChild->data()->cost;
        auto current = parent(bestChild);

        while (current != nullptr) {
            current->data()->cost += cost;
            current->data()->visits += 1;
            current = parent(current);
        }
    }

    bool MCTS::compareCost(VarNode *n1, VarNode *n2) {
        return n1->data()->cost < n2->data()->cost;
    }

    bool MCTS::compareUCT(VarNode *n1, VarNode *n2) const {
        return uct(n1) < uct(n2);
    }

    double MCTS::uct(VarNode *node) const {
        int n = parent(node)->data()->visits;
        int n_i = node->data()->visits;

        return - node->data()->cost / n_i + _config->explorationConstant() * std::sqrt(std::log(n) / n_i);
    }

    double MCTS::efe(
            const Tensor &sBeliefs,
            const Tensor &oBeliefs,
            const Tensor &a,
            const std::shared_ptr<MCTSConfig> &conf
    ) {
        auto c = conf->obsPreferences();
        auto risk = torch::matmul(oBeliefs, oBeliefs.log() - c.log()).item<double>();
        auto ambiguity = - torch::matmul(torch::diag(torch::matmul(a.log().t(), a)), sBeliefs).item<double>();

        return risk + ambiguity;
    }

    double MCTS::doubleKL(
            const Tensor &sBeliefs,
            const Tensor &oBeliefs,
            const Tensor &a,
            const std::shared_ptr<MCTSConfig> &conf
    ) {
        auto cObs = conf->obsPreferences();
        auto riskObs = torch::inner(oBeliefs, oBeliefs.log() - cObs.log()).item<double>();
        auto cState = conf->statesPreferences();
        auto riskState = torch::inner(sBeliefs, sBeliefs.log() - cState.log()).item<double>();

        return riskObs + riskState;
    }

    int MCTS::selectAction(VarNode *root) const {
        Tensor w = API::empty({root->nChildrenHiddenStates()});

        for (int i = 0; i < root->nChildren(); ++i) {
            auto child = root->child(i);
            int action = child->data()->action;
            if (action == -1)
                continue;
            w[action] = - _config->actionPrecision() * child->data()->cost / child->data()->visits;
        }
        w = softmax(w, 0);
        return Ops::randomInt(w);
    }

    std::shared_ptr<MCTSConfig> MCTS::config() const {
        return _config;
    }

    VarNode *MCTS::parent(VarNode *node) {
        return node->parent()->parent(0);
    }

}
