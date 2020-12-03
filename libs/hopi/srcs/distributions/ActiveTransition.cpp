//
// Created by tmac3 on 28/11/2020.
//

#include "nodes/ActiveTransitionNode.h"
#include "ActiveTransition.h"
#include "nodes/VarNode.h"
#include "graphs/FactorGraph.h"
#include "Categorical.h"

using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace Eigen;

namespace hopi::distributions {

    VarNode *ActiveTransition::create(VarNode *s, VarNode *a, const std::vector<MatrixXd> &p) {
        std::shared_ptr<FactorGraph> fg = FactorGraph::current();
        VarNode *var = fg->addNode(std::make_unique<VarNode>(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(std::make_unique<ActiveTransitionNode>(s, a, var));

        var->setParent(factor);
        var->setPrior(std::make_unique<ActiveTransition>(p));
        var->setPosterior(std::make_unique<Categorical>(
                MatrixXd::Constant(p[0].rows(), 1, 1.0 / p[0].rows())
        ));

        s->addChild(factor);
        a->addChild(factor);
        return var;
    }

    ActiveTransition::ActiveTransition(std::vector<MatrixXd> p) {
        param = std::move(p);
    }

    [[nodiscard]] DistributionType ActiveTransition::type() const {
        return DistributionType::ACTIVE_TRANSITION;
    }

    std::vector<MatrixXd> ActiveTransition::logProbability() const {
        std::vector<MatrixXd> res(param.size());

        for (int i = 0; i < param.size(); ++i) {
            res[i] = param[i];
            res[i] = res[i].array().log();
        }
        return res;
    }

    std::vector<MatrixXd> ActiveTransition::probability() const {
        std::vector<MatrixXd> res(param.size());

        for (int i = 0; i < param.size(); ++i) {
            res[i] = param[i];
        }
        return res;
    }

}
