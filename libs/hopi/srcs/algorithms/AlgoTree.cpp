//
// Created by tmac3 on 28/11/2020.
//

#include "AlgoTree.h"
#include <numeric>
#include <random>
#include "distributions/Transition.h"
#include "distributions/Categorical.h"
#include "distributions/Dirichlet.h"
#include "nodes/VarNode.h"
#include "nodes/FactorNode.h"
#include "graphs/FactorGraph.h"
#include "math/Functions.h"
#include <Eigen/Dense>
#include <limits>
#include <chrono>
#include <iostream>

using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::math;
using namespace hopi::distributions;
using namespace Eigen;

using namespace std::chrono;

namespace hopi::algorithms {

    AlgoTree::AlgoTree(int na, MatrixXd sp, MatrixXd op, int max_tree_depth) {
        state_pref = std::move(sp);
        obs_pref = std::move(op);
        n_actions = na;
        gen = std::default_random_engine(high_resolution_clock::now().time_since_epoch().count());
        tree_root = nullptr;
        _mtd = max_tree_depth;
    }

    /**
     * Step 1: Select the next node to be expanded
     */

    VarNode *AlgoTree::nodeSelection(const std::shared_ptr<FactorGraph>& fg, NodeSelectionType type) {
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
            switch (type) {
                case MIN:
                    return nodeSelectionMin(fg);
                case SAMPLING:
                    return nodeSelectionSampling(fg);
                case SOFTMAX_SAMPLING:
                    return nodeSelectionSoftmaxSampling(fg);
                default:
                    throw std::runtime_error("Unsupported node selection type.");
            }
        }
    }

    VarNode *AlgoTree::nodeSelectionMin(const std::shared_ptr<FactorGraph>& fg) {
        return std::min_element(us.begin(), us.end(), AlgoTree::CompareQuality)->first;
    }

    VarNode *AlgoTree::nodeSelectionSampling(const std::shared_ptr<FactorGraph>& fg) {
        std::vector<double> weight;
        for (auto & s : us) {
            weight.push_back(-s.first->g());
        }
        std::discrete_distribution<int> rand_int(weight.begin(), weight.end());
        return us[rand_int(gen)].first;
    }

    VarNode *AlgoTree::nodeSelectionSoftmaxSampling(const std::shared_ptr<FactorGraph> &fg) {
        MatrixXd w(us.size(), 1);
        for (int i = 0; i < us.size(); ++i) {
            w(i, 0) = -us[i].first->g();
        }
        w = Functions::softmax(w);
        std::vector<double> weight;
        for (int i = 0; i < us.size(); ++i) {
            weight.push_back(w(i, 0));
        }
        std::discrete_distribution<int> rand_int(weight.begin(), weight.end());
        return us[rand_int(gen)].first;
    }

    /**
     * Step 2: Expansion of the selected node
     */

    void AlgoTree::expansion(VarNode *node, VarNode *A, VarNode *B) {
        // Gather A and B parameters from the variable nodes
        auto A_param = Dirichlet::expectedLog(A->posterior()->params())[0];
        auto B_param = Dirichlet::expectedLog(B->posterior()->params());
        A_param = A_param.array().exp();
        for (auto & i : B_param) {
            i = i.array().exp();
        }
        // Call the expansion taking matrices of parameters as arguments
        expansion(node, A_param, B_param);
    }

    void AlgoTree::expansion(VarNode *node, MatrixXd& A, std::vector<MatrixXd>& B) {
        // Select an unexplored action to expand the tree
        std::vector<int> ua = unexploredActions(node);
        if (ua.empty()) {
            throw std::runtime_error("No more unexplored action: this node cannot be expanded.");
        }
        std::uniform_int_distribution<int> rand_int(0, ua.size() - 1);
        int action = ua[rand_int(gen)];
        VarNode *s = Transition::create(node, B[action]);
        VarNode *o = Transition::create(s, A);

        // Generative model expansion
        s->setAction(action);
        s->setBiased(std::make_unique<Categorical>(state_pref));
        o->setBiased(std::make_unique<Categorical>(obs_pref));
        last_expansion = std::make_pair(s, o);

        // Update list of unexplored_states
        if (ua.size() == 1) {
            us.erase(std::find_if(us.begin(), us.end(), [node](const std::pair<VarNode*,VarNode*>& p) {
                return p.first == node;
            }));
        }
        if (_mtd == -1 || distance_from_root(s) < _mtd) {
            us.emplace_back(s, o);
        }
    }

    /**
     * Step 3: Evaluation of the newly expanded node
     */

    void AlgoTree::evaluation(EvaluationType type) {
        switch (type) {
            case EvaluationType::SUM:
                evaluationSum(last_expansion.first, last_expansion.second);
                break;
            case EvaluationType::AVERAGE:
                evaluationAverage(last_expansion.first, last_expansion.second);
                break;
            case EvaluationType::KL:
                evaluationKL(last_expansion.first, last_expansion.second);
                break;
            default:
                throw std::runtime_error("Unsupported evaluation type.");
        }
    }

    /**
     * Step 4: Back-propagation of the information in the tree
     */

    void AlgoTree::backpropagation(VarNode *node, VarNode *root, BackPropagationType type) {
        // If type == NO_BP then do nothing
        // If type == DOWNWARD_BP then G_child = G_parent + G_child
        if (type == DOWNWARD_BP) {
            node->setG(node->g() + node->parent()->parent(0)->g());
        }
        while (node != root) {
            node->incrementN(); // N_ancestors++
            auto parent = node->parent()->parent(0);
            // If type == UPWARD_BP then G_parent = G_parent + G_child
            if (type == UPWARD_BP) {
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
     * Auxiliary functions
     */

    bool AlgoTree::CompareQuality(std::pair<VarNode*,VarNode*> a1, std::pair<VarNode*,VarNode*> a2) {
        return a1.first->g() < a2.first->g();
    }

    std::vector<int> AlgoTree::unexploredActions(VarNode *node) const {
        std::vector<int> ua(n_actions);
        std::iota(ua.begin(), ua.end(), 0);

        for (auto it = node->firstChild(); it != node->lastChild(); ++it) {
            auto i = std::find(ua.begin(), ua.end(), (*it)->child()->action());
            if (i != ua.end()) {
                ua.erase(i);
            }
        }
        return ua;
    }

    std::vector<VarNode*> AlgoTree::lastExpandedNodes() const {
        std::vector<VarNode*> vars;

        vars.push_back(last_expansion.first);
        vars.push_back(last_expansion.second);
        return vars;
    }

    void AlgoTree::evaluationSum(nodes::VarNode *s, nodes::VarNode *o) {
        double parent_G = s->parent()->parent(0)->g();

        if (parent_G == std::numeric_limits<double>::min()) {
            parent_G = 0;
        }
        s->setG(parent_G \
                 + Functions::KL(s->posterior(), s->biased()) \
                 + Functions::KL(o->posterior(), o->biased()) );
    }

    void AlgoTree::evaluationAverage(nodes::VarNode *s, nodes::VarNode *o) {
        int d = distance_from_root(s);
        double parent_G = s->parent()->parent(0)->g();

        if (parent_G == std::numeric_limits<double>::min()) {
            parent_G = 0;
        }
        double g = Functions::KL(s->posterior(), s->biased()) \
                 + Functions::KL(o->posterior(), o->biased());
        s->setG(((d * parent_G) + g) / (double)(d + 1));
    }

    void AlgoTree::evaluationKL(nodes::VarNode *s, nodes::VarNode *o) {
        s->setG(Functions::KL(s->posterior(), s->biased()) \
                 + Functions::KL(o->posterior(), o->biased()) );
    }

    int AlgoTree::distance_from_root(nodes::VarNode *n) {
        int d = 0;

        while (n != tree_root) {
            n = n->parent()->parent(0);
            ++d;
        }
        return d;
    }

}
