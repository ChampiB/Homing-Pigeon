//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_ALGOTREE_H
#define HOMING_PIGEON_2_ALGOTREE_H

#include <memory>
#include <random>
#include <Eigen/Dense>
#include <map>
#include "EvaluationType.h"
#include "BackPropagationType.h"
#include "NodeSelectionType.h"

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

    class AlgoTree {
    public:
        AlgoTree(int n_actions, Eigen::MatrixXd state_pref, Eigen::MatrixXd obs_pref, int max_tree_depth = -1);
        nodes::VarNode *nodeSelection(
                const std::shared_ptr<graphs::FactorGraph>& fg,
                NodeSelectionType type = NodeSelectionType::SOFTMAX_SAMPLING
        );
        void expansion(nodes::VarNode *p, Eigen::MatrixXd& A, std::vector<Eigen::MatrixXd>& B);
        void expansion(nodes::VarNode *p, nodes::VarNode *A, nodes::VarNode *B);
        void evaluation(EvaluationType type = EvaluationType::KL);
        static void backpropagation(nodes::VarNode *node, nodes::VarNode *root, BackPropagationType type = UPWARD_BP);
        static int actionSelection(nodes::VarNode *root);

    public:
        std::vector<int> unexploredActions(nodes::VarNode *node) const;
        [[nodiscard]] std::vector<nodes::VarNode*> lastExpandedNodes() const;
        static bool CompareQuality(
                std::pair<nodes::VarNode*,nodes::VarNode*> a1,
                std::pair<nodes::VarNode*,nodes::VarNode*> a2
        );
        int distance_from_root(nodes::VarNode *n);

    private:
        static void evaluationSum(nodes::VarNode *s, nodes::VarNode *o);
        void evaluationAverage(nodes::VarNode *s, nodes::VarNode *o);
        static void evaluationKL(nodes::VarNode *s, nodes::VarNode *o);

        nodes::VarNode *nodeSelectionMin(const std::shared_ptr<graphs::FactorGraph>& fg);
        nodes::VarNode *nodeSelectionSampling(const std::shared_ptr<graphs::FactorGraph>& fg);
        nodes::VarNode *nodeSelectionSoftmaxSampling(const std::shared_ptr<graphs::FactorGraph>& fg);

    private:
        std::default_random_engine gen; // Random number generator

    private:
        std::vector<std::pair<nodes::VarNode*,nodes::VarNode*>> us; // unexplored states: pair state and observation
        std::pair<nodes::VarNode*,nodes::VarNode*> last_expansion;
        Eigen::MatrixXd state_pref;
        Eigen::MatrixXd obs_pref;
        int n_actions;
        nodes::VarNode *tree_root;
        int _mtd; // max tree depth
    };

}

#endif //HOMING_PIGEON_2_ALGOTREE_H
