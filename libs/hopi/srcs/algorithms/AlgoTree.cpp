//
// Created by Theophile Champion on 28/11/2020.
//

#include "AlgoTree.h"
#include <numeric>
#include <random>
#include "distributions/Categorical.h"
#include "distributions/Dirichlet.h"
#include "api/API.h"
#include "nodes/VarNode.h"
#include "nodes/FactorNode.h"
#include "graphs/FactorGraph.h"
#include "math/Ops.h"
#include <torch/torch.h>
#include <limits>
#include <chrono>
#include <iostream>
#include <utility>

using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::math;
using namespace hopi::api;
using namespace hopi::distributions;
using namespace torch;
using namespace torch::indexing;
using namespace std::chrono;

namespace hopi::algorithms {

    /**
     * Factories
     */

    std::unique_ptr<AlgoTree> AlgoTree::create(int n_acts, const Tensor &state_pref, const Tensor &obs_pref) {
        return std::make_unique<AlgoTree>(n_acts, state_pref, obs_pref);
    }

    std::unique_ptr<AlgoTree> AlgoTree::create(int n_acts, const Tensor &&state_pref, const Tensor &&obs_pref) {
        return std::make_unique<AlgoTree>(n_acts, state_pref, obs_pref);
    }

    std::unique_ptr<AlgoTree> AlgoTree::create(AlgoTreeConfig &config) {
        return std::make_unique<AlgoTree>(config);
    }

    /**
     * Constructors
     */

    AlgoTree::AlgoTree(int n_acts, const Tensor &state_pref, const Tensor &obs_pref) :
            AlgoTree(AlgoTreeConfig(n_acts, state_pref, obs_pref)) {}

    AlgoTree::AlgoTree(int n_acts, const Tensor &&state_pref, const Tensor &&obs_pref) :
            AlgoTree(AlgoTreeConfig(n_acts, state_pref, obs_pref)) {}

    AlgoTree::AlgoTree(AlgoTreeConfig &conf) : AlgoTree(AlgoTreeConfig(conf)) {}

    AlgoTree::AlgoTree(AlgoTreeConfig &&conf) : config(conf) {
        gen = std::default_random_engine(high_resolution_clock::now().time_since_epoch().count());
        nodeSelectionImpl = nodeSelectionFn(config.node_selection_type);
        evaluationImpl = evaluationFn(config.evaluation_type);
        tree_root = nullptr;

    }

    /**
     * Step 1: Select the next node to be expanded
     */

    VarNode *AlgoTree::nodeSelection(const std::shared_ptr<FactorGraph> &fg) {
        if (us.empty()) {
            // If there are no unexplored states,
            //    then add the root node as an unexplored states...
            tree_root = fg->treeRoot();
            us.emplace_back(tree_root, (*tree_root->firstChild())->child());
            tree_root->setG(std::numeric_limits<double>::min());
            return fg->treeRoot();
        } else if (us[0].first->g() == std::numeric_limits<double>::min()) {
            // Else if the first unexplored state is the root node (i.e. g == -inf),
            //    then return the root node...
            return us[0].first;
        } else {
            // Otherwise uses the node selection procedure requested by the user,
            //    i.e. MIN, SAMPLING, SOFTMAX_SAMPLING
            return (this->*nodeSelectionImpl)();
        }
    }

    /**
     * Step 2: Expansion of the selected n
     */

    void AlgoTree::expansion(VarNode *n, VarNode *A, VarNode *B) {
        // Gather A and B parameters from the variable nodes
        auto A_param = Dirichlet::expectedLog(A->posterior()->params()).permute({1,0}).exp();
        auto B_param = Dirichlet::expectedLog(B->posterior()->params()).permute({2,0,1}).exp();
        // Call the expansion taking tensors of parameters as arguments
        expansion(n, A_param, B_param);
    }

    void AlgoTree::expansion(VarNode *n, const Tensor &A, const Tensor &B) {
        // Select an unexplored action to expand the tree
        std::vector<int> ua = unexploredActions(n);
        std::uniform_int_distribution<int> rand_int(0, (int)ua.size() - 1);
        int action = ua[rand_int(gen)];

        // Generative model expansion
        auto B_action = squeeze(B.index_select(2, at::tensor(action)));
        VarNode *s = API::Transition(n, B_action);
        VarNode *o = API::Transition(s, A);
        s->setAction(action);
        s->setBiased(Categorical::create(config.state_pref));
        o->setBiased(Categorical::create(config.obs_pref));
        last_expansion = {s, o};

        // Update list of unexplored_states
        if (ua.size() == 1) {
            us.erase(std::find_if(us.begin(), us.end(), [n](const VarNodePair& p) {
                return p.first == n;
            }));
        }
        if (config.max_tree_depth == -1 || distanceFromRoot(s) < config.max_tree_depth) {
            us.emplace_back(s, o);
        }
    }

    /**
     * Optional step: Return the last expanded nodes (i.e., the last state and observation that have been expanded) so
     * that local inference can be performed on them.
     */
    std::vector<VarNode*> AlgoTree::lastExpandedNodes() const {
        std::vector<VarNode*> vars;

        vars.push_back(last_expansion.first);
        vars.push_back(last_expansion.second);
        return vars;
    }

    /**
     * Step 3: Evaluation of the newly expanded node
     */

    void AlgoTree::evaluation() {
        double g = (*evaluationImpl)(last_expansion.first, last_expansion.second);
        last_expansion.first->setG(g);
    }

    /**
     * Step 4: Back-propagation of the information in the tree
     */

    void AlgoTree::propagation(VarNode *node, VarNode *root) const {
        // If type == NO_BP then do nothing
        // If type == DOWNWARD_BP then G_child = G_parent + G_child
        if (config.back_propagation_type == DOWNWARD_BP) {
            node->setG(node->g() + node->parent()->parent(0)->g());
        }
        while (node != root) {
            node->incrementN(); // N_ancestors++
            auto parent = node->parent()->parent(0);
            // If type == UPWARD_BP then G_parent = G_parent + G_child
            if (config.back_propagation_type == UPWARD_BP) {
                parent->setG(parent->g() + node->g());
            }
            node = parent;
        }
        root->incrementN(); // N_ancestors++
    }

    /**
     * Step 5: Selection of the best action to perform
     */

    int AlgoTree::actionSelection(VarNode *root) {
        double best_g      = std::numeric_limits<double>::max();
        int    best_n      = std::numeric_limits<int>::min();
        int    best_action = -1;
        VarNode *current;

        for (auto it = root->firstChild(); it != root->lastChild(); ++it) {
            current = (*it)->child();
            if (current->type() == OBSERVED) { // Do not consider the observed child, i.e. P(o|s)
                continue;
            }
            if (current->n() > best_n || (current->n() == best_n && current->g() < best_g) ) {
                best_action = current->action();
                best_n = current->n();
                best_g = current->g();
            }
        }
        return best_action;
    }

    /**
     * Different types of node selection
     */

    VarNode *AlgoTree::nodeSelectionMin() {
        return std::min_element(us.begin(), us.end(), AlgoTree::CompareCost)->first;
    }

    VarNode *AlgoTree::nodeSelectionSampling() {
        std::vector<double> weight;
        for (auto & s : us) {
            weight.push_back(-s.first->g());
        }
        std::discrete_distribution<int> rand_int(weight.begin(), weight.end());
        return us[rand_int(gen)].first;
    }

    VarNode *AlgoTree::nodeSelectionSoftmaxSampling() {
        Tensor w = API::empty({ (long) us.size() });
        for (int i = 0; i < us.size(); ++i) {
            w[i] = -us[i].first->g();
        }
        w = softmax(w, 0);
        auto r_ptr = w.data_ptr<double>();
        std::vector<double> weight{r_ptr, r_ptr + w.size(0)};
        std::discrete_distribution<int> rand_int(weight.begin(), weight.end());
        return us[rand_int(gen)].first;
    }

    /**
     * Different types of evaluation
     */

    double AlgoTree::doubleKL(nodes::VarNode *s, nodes::VarNode *o) {
        return Ops::kl(s->posterior(), s->biased()) + Ops::kl(o->posterior(), o->biased());
    }

    double AlgoTree::efe(nodes::VarNode *s, nodes::VarNode *o) {
        auto ambiguity = Ops::average(-o->prior()->logParams(), o->prior()->params(), {0,1}, {1});

        ambiguity = Ops::average(ambiguity, s->posterior()->params(), {0});
        return Ops::kl(o->posterior(), o->biased()) + ambiguity.item<double>();
    }

    /**
     * Auxiliary functions
     */

    bool AlgoTree::CompareCost(std::pair<VarNode*,VarNode*> a1, std::pair<VarNode*,VarNode*> a2) {
        return a1.first->g() < a2.first->g();
    }

    std::vector<int> AlgoTree::unexploredActions(VarNode *n) const {
        std::vector<int> ua(config.n_actions);
        std::iota(ua.begin(), ua.end(), 0);

        for (auto it = n->firstChild(); it != n->lastChild(); ++it) {
            auto i = std::find(ua.begin(), ua.end(), (*it)->child()->action());
            if (i != ua.end()) {
                ua.erase(i);
            }
        }
        if (ua.empty()) {
            throw std::runtime_error("AlgoTree::unexploredActions, no more unexplored actions.");
        }
        return ua;
    }

    int AlgoTree::distanceFromRoot(nodes::VarNode *n) {
        int d = 0;

        while (n != tree_root) {
            n = n->parent()->parent(0);
            ++d;
        }
        return d;
    }

    /**
     * Configuration functions
     */

    NodeSelectionFn AlgoTree::nodeSelectionFn(NodeSelectionType type) {
        static std::map<NodeSelectionType, NodeSelectionFn> map {
                {MIN,              &AlgoTree::nodeSelectionMin},
                {SAMPLING,         &AlgoTree::nodeSelectionSampling},
                {SOFTMAX_SAMPLING, &AlgoTree::nodeSelectionSoftmaxSampling},
        };

        assert(map.find(type) != map.end() && "AlgoTree::nodeSelectionFn, unsupported node selection type.");
        return map[type];
    }

    EvaluationFn AlgoTree::evaluationFn(EvaluationType type) {
        static std::map<EvaluationType, EvaluationFn> map {
                {DOUBLE_KL, &AlgoTree::doubleKL},
                {EFE,       &AlgoTree::efe},
        };

        assert(map.find(type) != map.end() && "AlgoTree::evaluationFn, unsupported evaluation type.");
        return map[type];
    }

}
