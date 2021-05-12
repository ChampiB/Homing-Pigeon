//
// Created by tmac3 on 28/04/2021.
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

    class AlgoTreeConfig {

    public:
        AlgoTreeConfig(int n_acts, const torch::Tensor &state_pref,  const torch::Tensor &obs_pref);
        AlgoTreeConfig(int n_acts, const torch::Tensor &&state_pref, const torch::Tensor &&obs_pref);
        AlgoTreeConfig(int n_acts, const torch::Tensor &&state_pref, const torch::Tensor &obs_pref);
        AlgoTreeConfig(int n_acts, const torch::Tensor &sp,          const torch::Tensor &&op);
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
        explicit AlgoTreeConfig(int n_acts);
    };

}

#endif //HOMINGPIGEON_ALGOTREECONFIG_H
