//
// Created by tmac3 on 28/04/2021.
//

#include "AlgoTreeConfig.h"

using namespace Eigen;

namespace hopi::algorithms {

    AlgoTreeConfig::AlgoTreeConfig(int n_acts, MatrixXd &sp, MatrixXd &op) : AlgoTreeConfig(n_acts) {
        state_pref = sp.replicate(1, 1);
        obs_pref   = op.replicate(1, 1);
    }

    AlgoTreeConfig::AlgoTreeConfig(int n_acts, MatrixXd &&sp, MatrixXd &&op) : AlgoTreeConfig(n_acts) {
        state_pref = sp;
        obs_pref   = op;
    }

    AlgoTreeConfig::AlgoTreeConfig(int n_acts, MatrixXd &&sp, MatrixXd &op) : AlgoTreeConfig(n_acts) {
        state_pref = sp;
        obs_pref   = op.replicate(1, 1);
    }

    AlgoTreeConfig::AlgoTreeConfig(int n_acts, MatrixXd &sp, MatrixXd &&op) : AlgoTreeConfig(n_acts) {
        state_pref = sp.replicate(1, 1);
        obs_pref   = op;
    }

    AlgoTreeConfig::AlgoTreeConfig(int n_acts) {
        n_actions             = n_acts;
        max_tree_depth        = -1;
        node_selection_type   = NodeSelectionType::SOFTMAX_SAMPLING;
        evaluation_type       = EvaluationType::DOUBLE_KL;
        back_propagation_type = BackPropagationType::UPWARD_BP;
    }

    AlgoTreeConfig::AlgoTreeConfig(const AlgoTreeConfig &rhs) {
        state_pref             = rhs.state_pref;
        obs_pref               = rhs.obs_pref;
        n_actions              = rhs.n_actions;
        max_tree_depth         = rhs.max_tree_depth;
        node_selection_type    = rhs.node_selection_type;
        evaluation_type        = rhs.evaluation_type;
        back_propagation_type  = rhs.back_propagation_type;
    }

}
