//
// Created by Theophile Champion on 28/11/2020.
//

#include "VMP.h"
#include <distributions/Categorical.h>
#include "nodes/FactorNode.h"
#include "nodes/VarNode.h"
#include "iterators/HiddenVarIter.h"
#include "iterators/AdjacentFactorsIter.h"

using namespace hopi::graphs;
using namespace hopi::iterators;
using namespace hopi::distributions;
using namespace hopi::nodes;
using namespace torch;

namespace hopi::algorithms::inference {

    void VMP::inference(const std::vector<VarNode*>& vars, double epsilon, int max_iter) {
        double VFE = std::numeric_limits<double>::max();
        int iter = 0;

        while (true) {
            // Perform inference
            HiddenVarIter hiddenIt(vars);

            while (*hiddenIt != nullptr) {
                inference(*hiddenIt);
                ++hiddenIt;
            }

            // Check if the variational free energy have converged
            double new_VFE = vfe(vars);

            if (VFE - new_VFE < epsilon) {
                break;
            }
            VFE = new_VFE;

            // Check if the maximum number of iterations have been reached
            if (iter >= max_iter) {
                break;
            }
            ++iter;
        }
    }

    void VMP::inference(VarNode *var) {
        Tensor post_param;
        AdjacentFactorsIter factorIt(var);

        while (*factorIt != nullptr) {
            if (post_param.numel() == 0) {
                post_param = (*factorIt)->message(var);
            } else {
                post_param += (*factorIt)->message(var);
            }
            ++factorIt;
        }
        var->posterior()->updateParams(post_param);
    }

    double VMP::vfe(const std::vector<nodes::VarNode *> &vars) {
        double VFE = 0;

        for (auto v : vars) {
            VFE += v->parent()->vfe();
        }
        return VFE;
    }

}
