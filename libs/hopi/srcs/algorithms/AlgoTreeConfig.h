//
// Created by Theophile Champion on 28/04/2021.
//

#ifndef HOMINGPIGEON_ALGOTREECONFIG_H
#define HOMINGPIGEON_ALGOTREECONFIG_H

#include <torch/torch.h>
#include "BackPropagationType.h"
#include "NodeSelectionType.h"
#include "EvaluationType.h"

namespace hopi::algorithms {
    class AlgoTree;
}
namespace hopi::nodes {
    class VarNode;
}
namespace hopi::graphs {
    class FactorGraph;
}

namespace hopi::algorithms {

    /**
     * This class stores the configuration of a tree search algorithm.
     */
    class AlgoTreeConfig {

    public:
        /**
         * Construct a configuration for a tree search algorithm.
         * @param n_acts the number of actions available to the agent
         * @param state_pref the agent's prior preferences over hidden states
         * @param obs_pref the agent's prior preferences over observations
         */
        AlgoTreeConfig(int n_acts, const torch::Tensor &state_pref, const torch::Tensor &obs_pref);

        /**
         * Construct a configuration for a tree search algorithm.
         * @param n_acts the number of actions available to the agent
         * @param state_pref the agent's prior preferences over hidden states
         * @param obs_pref the agent's prior preferences over observations
         */
        AlgoTreeConfig(int n_acts, const torch::Tensor &&state_pref, const torch::Tensor &&obs_pref);

        /**
         * Construct a configuration for a tree search algorithm.
         * @param n_acts the number of actions available to the agent
         * @param state_pref the agent's prior preferences over hidden states
         * @param obs_pref the agent's prior preferences over observations
         */
        AlgoTreeConfig(int n_acts, const torch::Tensor &&state_pref, const torch::Tensor &obs_pref);

        /**
         * Construct a configuration for a tree search algorithm.
         * @param n_acts the number of actions available to the agent
         * @param state_pref the agent's prior preferences over hidden states
         * @param obs_pref the agent's prior preferences over observations
         */
        AlgoTreeConfig(int n_acts, const torch::Tensor &sp, const torch::Tensor &&op);

        /**
         * Copy constructor.
         * @param rhs the other configuration from which the copy is performed
         */
        AlgoTreeConfig(const AlgoTreeConfig &rhs);

    public:
        int                 n_actions;
        torch::Tensor       state_pref;
        torch::Tensor       obs_pref;
        int                 max_tree_depth;
        NodeSelectionType   node_selection_type;
        EvaluationType      evaluation_type;
        BackPropagationType back_propagation_type;

    private:
        /**
         * Construct a default configuration for the tree search algorithm. This constructor should not be called by
         * the final user.
         * @param n_acts the number of actions available to the agent
         */
        explicit AlgoTreeConfig(int n_acts);
    };

}

#endif //HOMINGPIGEON_ALGOTREECONFIG_H
