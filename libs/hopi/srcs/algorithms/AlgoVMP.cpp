//
// Created by tmac3 on 28/11/2020.
//

#include "AlgoVMP.h"
#include <Eigen/Dense>
#include <distributions/Categorical.h>
#include "graphs/FactorGraph.h"
#include "nodes/FactorNode.h"
#include "nodes/VarNode.h"
#include "iterators/HiddenVarIter.h"
#include "iterators/AdjacentFactorsIter.h"

using namespace hopi::graphs;
using namespace hopi::iterators;
using namespace hopi::distributions;
using namespace hopi::nodes;
using namespace Eigen;

namespace hopi::algorithms {

    void AlgoVMP::inference(const std::vector<VarNode*>& vars, double epsilon, int max_iter) {
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

    void AlgoVMP::inference(VarNode *var) {
        std::vector<MatrixXd> post_param;
        AdjacentFactorsIter factorIt(var);
        while (*factorIt != nullptr) {
            if (post_param.empty()) {
                // posterior parameters = message
                post_param = (*factorIt)->message(var);
            } else {
                // posterior parameters += message
                auto msg = (*factorIt)->message(var);
                for (int i = 0; i < post_param.size(); ++i) {
                    post_param[i] = post_param[i] + msg[i];
                }
            }
            ++factorIt;
        }
        var->posterior()->updateParams(post_param);
    }

    double AlgoVMP::vfe(const std::vector<nodes::VarNode *> &vars) {
        double VFE = 0;

        for (auto v : vars) {
            VFE += v->parent()->vfe();
        }
        return VFE;
    }

}
