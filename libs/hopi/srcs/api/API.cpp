//
// Created by tmac3 on 10/05/2021.
//

#include "API.h"
#include "graphs/FactorGraph.h"
#include "nodes/VarNode.h"
#include "nodes/CategoricalNode.h"
#include "nodes/TransitionNode.h"
#include "distributions/Categorical.h"
#include "distributions/Transition.h"
#include "nodes/ActiveTransitionNode.h"
#include "distributions/ActiveTransition.h"
#include "distributions/Dirichlet.h"
#include "nodes/DirichletNode.h"
#include "math/Functions.h"
#include <memory>

using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::distributions;
using namespace hopi::math;
using namespace Eigen;

namespace hopi::api {

    RV *API::Categorical(const MatrixXd &param) {
        auto fg = FactorGraph::current();
        VarNode *var = fg->addNode(VarNode::create(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(CategoricalNode::create(var));

        var->setParent(factor);
        var->setPrior(Categorical::create(param));
        var->setPosterior(Categorical::create(MatrixXd::Constant(param.rows(), 1, 1.0 / (double)param.rows())));
        return var;
    }

    RV *API::Categorical(RV *param) {
        auto fg = FactorGraph::current();
        VarNode *var = fg->addNode(VarNode::create(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(CategoricalNode::create(var, param));

        var->setParent(factor);
        auto dim = param->prior()->params()[0].rows();
        var->setPosterior(Categorical::create(MatrixXd::Constant(dim, 1, 1.0 / (double)dim)));

        param->addChild(factor);
        return var;
    }

    RV *API::Transition(RV *s, const MatrixXd &param) {
        auto fg = FactorGraph::current();
        VarNode *var = fg->addNode(VarNode::create(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(TransitionNode::create(s, var));

        var->setParent(factor);
        var->setPrior(Transition::create(param));
        var->setPosterior(Categorical::create(Functions::uniformColumnWise((int)param.rows(), 1)));

        s->addChild(factor);
        return var;
    }

    RV *API::Transition(RV *s, RV *param) {
        auto fg = FactorGraph::current();
        VarNode *var = fg->addNode(VarNode::create(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(TransitionNode::create(s, var, param));

        var->setParent(factor);
        auto dim = param->prior()->params()[0].rows();
        var->setPosterior(Categorical::create(MatrixXd::Constant(dim, 1, 1.0 / (double)dim)));

        s->addChild(factor);
        param->addChild(factor);
        return var;
    }

    RV *API::ActiveTransition(RV *s, RV *a, const std::vector<MatrixXd> &param) {
        auto fg = FactorGraph::current();
        VarNode *var = fg->addNode(VarNode::create(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(ActiveTransitionNode::create(s, a, var));

        var->setParent(factor);
        var->setPrior(ActiveTransition::create(param));
        var->setPosterior(Categorical::create(MatrixXd::Constant(param[0].rows(), 1, 1.0 / (double)param[0].rows())));

        s->addChild(factor);
        a->addChild(factor);
        return var;
    }

    RV *API::ActiveTransition(RV *s, RV *a, RV *param) {
        auto fg = FactorGraph::current();
        VarNode *var = fg->addNode(VarNode::create(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(ActiveTransitionNode::create(s, a, var, param));

        var->setParent(factor);
        auto dim = param->prior()->params()[0].rows();
        var->setPosterior(Categorical::create(MatrixXd::Constant(dim, 1, 1.0 / (double)dim)));

        s->addChild(factor);
        a->addChild(factor);
        param->addChild(factor);
        return var;
    }

    RV *API::Dirichlet(const MatrixXd &param) {
        std::vector<MatrixXd> p = { param };
        return Dirichlet(p);
    }

    RV *API::Dirichlet(const std::vector<MatrixXd> &param) {
        auto fg = FactorGraph::current();
        VarNode *var = fg->addNode(VarNode::create(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(DirichletNode::create(var));
        std::vector<MatrixXd> param_copy;

        for (const auto & i : param) {
            param_copy.push_back(i);
        }
        var->setParent(factor);
        var->setPrior(Dirichlet::create(param));
        var->setPosterior(Dirichlet::create(param_copy));
        return var;
    }

}
