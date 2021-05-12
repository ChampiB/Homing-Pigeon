//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_ALGOTREE_H
#define HOMING_PIGEON_2_ALGOTREE_H

#include <memory>
#include <random>
#include <torch/torch.h>
#include <map>
#include "AlgoTreeConfig.h"

namespace hopi::graphs {
    class FactorGraph;
}
namespace hopi::nodes {
    class VarNode;
}
namespace hopi::distributions {
    class Distribution;
}

namespace hopi::algorithms {

    // Definition of aliases
    using VarNodePair = std::pair<nodes::VarNode*, nodes::VarNode*>;
    using NodeSelectionFn = nodes::VarNode *(AlgoTree::*)();
    using EvaluationFn = double (*)(nodes::VarNode *, nodes::VarNode *);

    class AlgoTree {
    public:
        // Factories
        static std::unique_ptr<AlgoTree> create(int n_acts, const torch::Tensor &&state_pref, const torch::Tensor &&obs_pref);
        static std::unique_ptr<AlgoTree> create(int n_acts, const torch::Tensor &state_pref,  const torch::Tensor &obs_pref);
        static std::unique_ptr<AlgoTree> create(AlgoTreeConfig &config);

    public:
        // Constructors
        explicit AlgoTree(int n_acts, const torch::Tensor &&state_pref, const torch::Tensor &&obs_pref);
        explicit AlgoTree(int n_acts, const torch::Tensor &state_pref,  const torch::Tensor &obs_pref);
        explicit AlgoTree(AlgoTreeConfig &config);
        explicit AlgoTree(AlgoTreeConfig &&config);

    public:
        // Core functions
        nodes::VarNode *nodeSelection(const std::shared_ptr<graphs::FactorGraph>& fg);
        void expansion(nodes::VarNode *p, const torch::Tensor &A, const torch::Tensor &B);
        void expansion(nodes::VarNode *p, nodes::VarNode *A, nodes::VarNode *B);
        void evaluation();
        void backpropagation(nodes::VarNode *node, nodes::VarNode *root) const;
        static int actionSelection(nodes::VarNode *root);

    public:
        // Auxiliary functions
        std::vector<int> unexploredActions(nodes::VarNode *node) const; // TODO make private
        [[nodiscard]] std::vector<nodes::VarNode*> lastExpandedNodes() const; // TODO make private
        static bool CompareQuality(VarNodePair a1, VarNodePair a2); // TODO make private
        int distanceFromRoot(nodes::VarNode *n); // TODO make private

    private:
        std::default_random_engine gen; // Random number generator
        std::vector<VarNodePair> us;    // Unexplored states: state-observation pairs
        VarNodePair last_expansion;     // State-observation pair
        nodes::VarNode *tree_root;
        AlgoTreeConfig config;

    private:
        // The implementation of node selection, evaluation and quality to use.
        NodeSelectionFn nodeSelectionImpl;
        EvaluationFn evaluationImpl;

        // Configuration functions
        static NodeSelectionFn nodeSelectionFn(NodeSelectionType type);
        static EvaluationFn    evaluationFn(EvaluationType type);

        // Different kind of evaluation
        static double doubleKL(nodes::VarNode *s, nodes::VarNode *o);
        static double efe(nodes::VarNode *s, nodes::VarNode *o); // TODO test

        // Different kind of node selection
        nodes::VarNode *nodeSelectionMin();
        nodes::VarNode *nodeSelectionSampling();
        nodes::VarNode *nodeSelectionSoftmaxSampling();

    };

}

#endif //HOMING_PIGEON_2_ALGOTREE_H
