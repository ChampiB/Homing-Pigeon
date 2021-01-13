//
// Created by tmac3 on 28/11/2020.
//

#include "nodes/ActiveTransitionNode.h"
#include "ActiveTransition.h"
#include "nodes/VarNode.h"
#include "math/Functions.h"
#include "graphs/FactorGraph.h"
#include "Categorical.h"

using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::math;
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

    VarNode *ActiveTransition::create(VarNode *s, VarNode *a, VarNode *param) {
        std::shared_ptr<FactorGraph> fg = FactorGraph::current();
        VarNode *var = fg->addNode(std::make_unique<VarNode>(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(std::make_unique<ActiveTransitionNode>(s, a, var, param));

        var->setParent(factor);
        auto dim = param->prior()->params()[0].rows();
        var->setPosterior(std::make_unique<Categorical>(
            MatrixXd::Constant(dim, 1, 1.0 / dim)
        ));

        s->addChild(factor);
        a->addChild(factor);
        param->addChild(factor);
        return var;
    }

    ActiveTransition::ActiveTransition(std::vector<MatrixXd> p) {
        param = std::move(p);
    }

    [[nodiscard]] DistributionType ActiveTransition::type() const {
        return DistributionType::ACTIVE_TRANSITION;
    }

    std::vector<MatrixXd> ActiveTransition::logParams() const {
        std::vector<MatrixXd> res(param.size());

        for (int i = 0; i < param.size(); ++i) {
            res[i] = param[i];
            res[i] = res[i].array().log();
        }
        return res;
    }

    std::vector<MatrixXd> ActiveTransition::params() const {
        std::vector<MatrixXd> res(param.size());

        for (int i = 0; i < param.size(); ++i) {
            res[i] = param[i];
        }
        return res;
    }

    void ActiveTransition::setParams(std::vector<Eigen::MatrixXd> &p) {
        if (p.size() != param.size()) {
            throw std::runtime_error("ActiveTransition::setParams argument size must match parameter size.");
        }
        for (int i = 0; i < p.size(); ++i) {
            param[i] = Functions::softmax(p[i]);
        }
    }

    double ActiveTransition::entropy() {
        throw std::runtime_error("Unsupported: ActiveTransition::entropy()");
    }

}
