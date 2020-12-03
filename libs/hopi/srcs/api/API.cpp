//
// Created by Theophile Champion on 10/05/2021.
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
#include "math/Ops.h"

using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::distributions;
using namespace hopi::math;
using namespace torch;

namespace hopi::api {

    RV *API::Categorical(const Tensor &param) {
        auto fg = FactorGraph::current();
        VarNode *var = fg->addNode(VarNode::create(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(CategoricalNode::create(var));

        var->setParent(factor);
        var->setPrior(Categorical::create(param));
        var->setPosterior(Categorical::create(Ops::uniform({param.size(0)})));
        return var;
    }

    RV *API::Categorical(RV *param) {
        auto fg = FactorGraph::current();
        VarNode *var = fg->addNode(VarNode::create(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(CategoricalNode::create(var, param));

        var->setParent(factor);
        auto dim = param->prior()->params().size(0);
        var->setPosterior(Categorical::create(Ops::uniform({dim})));

        param->addChild(factor);
        return var;
    }

    RV *API::Transition(RV *s, const Tensor &param) {
        auto fg = FactorGraph::current();
        VarNode *var = fg->addNode(VarNode::create(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(TransitionNode::create(s, var));

        var->setParent(factor);
        var->setPrior(Transition::create(param));
        var->setPosterior(Categorical::create(Ops::uniform({param.size(0)})));

        s->addChild(factor);
        return var;
    }

    RV *API::Transition(RV *s, RV *param) {
        auto fg = FactorGraph::current();
        VarNode *var = fg->addNode(VarNode::create(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(TransitionNode::create(s, var, param));

        var->setParent(factor);
        auto dim = param->prior()->params().size(0);
        var->setPosterior(Categorical::create(Ops::uniform({dim})));

        s->addChild(factor);
        param->addChild(factor);
        return var;
    }

    RV *API::ActiveTransition(RV *s, RV *a, const Tensor &param) {
        auto fg = FactorGraph::current();
        VarNode *var = fg->addNode(VarNode::create(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(ActiveTransitionNode::create(s, a, var));

        var->setParent(factor);
        var->setPrior(ActiveTransition::create(param));
        var->setPosterior(Categorical::create(Ops::uniform({param.size(0)})));

        s->addChild(factor);
        a->addChild(factor);
        return var;
    }

    RV *API::ActiveTransition(RV *s, RV *a, RV *param) {
        auto fg = FactorGraph::current();
        VarNode *var = fg->addNode(VarNode::create(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(ActiveTransitionNode::create(s, a, var, param));

        var->setParent(factor);
        auto dim = param->prior()->params().size(0);
        var->setPosterior(Categorical::create(Ops::uniform({dim})));

        s->addChild(factor);
        a->addChild(factor);
        param->addChild(factor);
        return var;
    }

    RV *API::Dirichlet(const Tensor &param) {
        auto fg = FactorGraph::current();
        VarNode *var = fg->addNode(VarNode::create(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(DirichletNode::create(var));

        var->setParent(factor);
        var->setPrior(Dirichlet::create(param));
        var->setPosterior(Dirichlet::create(param));
        return var;
    }

    static ScalarType dType = kDouble;

    void API::setDataType(const ScalarType &type) {
        dType = type;
    }

    ScalarType API::dataType() {
        return dType;
    }

    Tensor API::zeros(IntArrayRef sizes) {
        return torch::zeros(sizes).to(dataType());
    }

    Tensor API::tensor(const torch::detail::TensorDataContainer &&data) {
        return torch::tensor(data).to(dataType());
    }

    Tensor API::empty(IntArrayRef sizes) {
        return torch::empty(sizes).to(dataType());
    }

    torch::Tensor API::range(at::Scalar from, at::Scalar to) {
        return torch::arange(from, to).to(dataType());
    }

    torch::Tensor API::ones(at::IntArrayRef sizes) {
        return torch::ones(sizes).to(dataType());
    }

}
