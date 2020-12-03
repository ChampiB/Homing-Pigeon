//
// Created by tmac3 on 28/11/2020.
//

#include "Transition.h"
#include "nodes/VarNode.h"
#include "nodes/TransitionNode.h"
#include "graphs/FactorGraph.h"
#include "Categorical.h"
#include <Eigen/Dense>
#include <utility>

using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace Eigen;

namespace hopi::distributions {

    VarNode *Transition::create(VarNode *s, const Eigen::MatrixXd& param) {
        std::shared_ptr<FactorGraph> fg = FactorGraph::current();
        VarNode *var = fg->addNode(std::make_unique<VarNode>(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(std::make_unique<TransitionNode>(s, var));

        var->setParent(factor);
        var->setPrior(std::make_unique<Transition>(param));
        var->setPosterior(std::make_unique<Categorical>(
            Eigen::MatrixXd::Constant(param.rows(), 1, 1.0 / param.rows())
        ));

        s->addChild(factor);
        return var;
    }

    Transition::Transition(Eigen::MatrixXd p) {
        param = std::move(p);
    }

    [[nodiscard]] DistributionType Transition::type() const {
        return DistributionType::TRANSITION;
    }

    std::vector<MatrixXd> Transition::logProbability() const {
        MatrixXd copy = param;
        std::vector<MatrixXd> res{copy.array().log()};
        return res;
    }

    std::vector<MatrixXd> Transition::probability() const {
        MatrixXd copy = param;
        std::vector<MatrixXd> res{copy.array()};
        return res;
    }

}