//
// Created by tmac3 on 28/11/2020.
//

#include "AlgoVMP.h"
#include <Eigen/Dense>
#include <distributions/Categorical.h>
#include <iostream>
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

    void AlgoVMP::inference(const std::vector<VarNode*>& vars, double epsilon) {
        double VFE = std::numeric_limits<double>::max();

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
        }
    }

    void AlgoVMP::inference(VarNode *var) {
        MatrixXd post_param;
        AdjacentFactorsIter factorIt(var);
        while (*factorIt != nullptr) {
            if (post_param.rows() == 0 && post_param.cols() == 0) {
                post_param = (*factorIt)->message(var);
            } else {
                post_param = post_param + (*factorIt)->message(var);
            }
            ++factorIt;
        }
        var->setPosterior(std::make_unique<Categorical>(AlgoVMP::softmax(post_param)));
    }

    MatrixXd AlgoVMP::softmax(MatrixXd &vector) {
        MatrixXd res = vector.array() - vector.maxCoeff();
        res = res.array().exp();
        double sum = res.sum();

        if (sum == 0) {
            return MatrixXd::Constant(vector.rows(), 1, 1.0 / res.size());
        }
        return res / sum;
    }

    double AlgoVMP::vfe(const std::vector<nodes::VarNode *> &vars) {
        double VFE = 0;

        for (auto v : vars) {
            if (v->type() == HIDDEN) {
                auto p  = v->posterior()->probability()[0];
                auto lp = v->posterior()->logProbability()[0];
                VFE += (p.transpose() * lp)(0, 0);
            }
            VFE -= v->parent()->vfe();
        }
        return VFE;
    }

}
