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

    RV *API::Categorical(const std::shared_ptr<Tensor> &param) {
        return API::Categorical(Categorical::create(param));
    }

    RV *API::Categorical(const Tensor &param) {
        return API::Categorical(Categorical::create(param));
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

    RV *API::Transition(RV *s, const std::shared_ptr<Tensor> &param) {
        return API::Transition(s, Transition::create(param));
    }

    RV *API::Transition(RV *s, const Tensor &param) {
        return API::Transition(s, Transition::create(param));
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
        return API::ActiveTransition(s, a, ActiveTransition::create(param));
    }

    RV *API::ActiveTransition(RV *s, RV *a, const std::shared_ptr<Tensor> &param) {
        return API::ActiveTransition(s, a, ActiveTransition::create(param));
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
        return API::Dirichlet(Dirichlet::create(param), Dirichlet::create(param));
    }

    RV *API::Dirichlet(const std::shared_ptr<torch::Tensor> &param) {
        return API::Dirichlet(Dirichlet::create(param), Dirichlet::create(*param));
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

    std::shared_ptr<torch::Tensor> API::toPtr(const Tensor &tensor) {
        return std::make_shared<torch::Tensor>(tensor);
    }

    std::shared_ptr<torch::Tensor> API::toPtr(const Tensor &&tensor) {
        return std::make_shared<torch::Tensor>(tensor);
    }

    RV *API::Categorical(std::unique_ptr<Distribution> prior) {
        auto fg = FactorGraph::current();
        VarNode *var = fg->addNode(VarNode::create(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(CategoricalNode::create(var));
        long nb_params = prior->params().size(0);

        var->setParent(factor);
        var->setPrior(std::move(prior));
        var->setPosterior(Categorical::create(Ops::uniform({nb_params})));
        return var;
    }

    RV *API::Transition(RV *s, std::unique_ptr<Distribution> prior) {
        auto fg = FactorGraph::current();
        VarNode *var = fg->addNode(VarNode::create(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(TransitionNode::create(s, var));
        long nb_params = prior->params().size(0);

        var->setParent(factor);
        var->setPrior(std::move(prior));
        var->setPosterior(Categorical::create(Ops::uniform({nb_params})));

        s->addChild(factor);
        return var;
    }

    RV *API::ActiveTransition(RV *s, RV *a, std::unique_ptr<Distribution> prior) {
        auto fg = FactorGraph::current();
        VarNode *var = fg->addNode(VarNode::create(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(ActiveTransitionNode::create(s, a, var));
        long nb_params = prior->params().size(0);

        var->setParent(factor);
        var->setPrior(std::move(prior));
        var->setPosterior(Categorical::create(Ops::uniform({nb_params})));

        s->addChild(factor);
        a->addChild(factor);
        return var;
    }

    RV *API::Dirichlet(std::unique_ptr<distributions::Distribution> prior,
                       std::unique_ptr<distributions::Distribution> posterior) {
        auto fg = FactorGraph::current();
        VarNode *var = fg->addNode(VarNode::create(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(DirichletNode::create(var));

        var->setParent(factor);
        var->setPrior(std::move(prior));
        var->setPosterior(std::move(posterior));
        return var;
    }

    std::vector<double> API::toStdVector(const Tensor &w) {
        std::vector<double> weight(w.size(0));

        for (int i = 0; i < w.size(0); ++i) {
            weight[i] = w[i].item<double>();
        }
        return weight;
    }

}
