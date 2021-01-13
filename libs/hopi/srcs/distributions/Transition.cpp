//
// Created by tmac3 on 28/11/2020.
//

#include "Transition.h"
#include "nodes/VarNode.h"
#include "nodes/TransitionNode.h"
#include "math/Functions.h"
#include "graphs/FactorGraph.h"
#include "Categorical.h"
#include <Eigen/Dense>

using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::math;
using namespace Eigen;

namespace hopi::distributions {

    VarNode *Transition::create(VarNode *s, const Eigen::MatrixXd& param) {
        std::shared_ptr<FactorGraph> fg = FactorGraph::current();
        VarNode *var = fg->addNode(std::make_unique<VarNode>(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(std::make_unique<TransitionNode>(s, var));

        var->setParent(factor);
        var->setPrior(std::make_unique<Transition>(param));
        var->setPosterior(std::make_unique<Categorical>(
            MatrixXd::Constant(param.rows(), 1, 1.0 / param.rows())
        ));

        s->addChild(factor);
        return var;
    }

    VarNode *Transition::create(VarNode *s, VarNode *param) {
        std::shared_ptr<FactorGraph> fg = FactorGraph::current();
        VarNode *var = fg->addNode(std::make_unique<VarNode>(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(std::make_unique<TransitionNode>(s, var, param));

        var->setParent(factor);
        auto dim = param->prior()->params()[0].rows();
        var->setPosterior(std::make_unique<Categorical>(
            MatrixXd::Constant(dim, 1, 1.0 / dim)
        ));

        s->addChild(factor);
        param->addChild(factor);
        return var;
    }

    Transition::Transition(Eigen::MatrixXd p) {
        param = std::move(p);
    }

    [[nodiscard]] DistributionType Transition::type() const {
        return DistributionType::TRANSITION;
    }

    std::vector<MatrixXd> Transition::logParams() const {
        MatrixXd copy = param;
        std::vector<MatrixXd> res{copy.array().log()};
        return res;
    }

    std::vector<MatrixXd> Transition::params() const {
        MatrixXd copy = param;
        std::vector<MatrixXd> res{copy.array()};
        return res;
    }

    void Transition::setParams(std::vector<Eigen::MatrixXd> &p) {
        if (p.size() != 1) {
            throw std::runtime_error("Transition::setParams argument size must be equal to one.");
        }
        param = Functions::softmax(p[0]);
    }

    double Transition::entropy() {
        throw std::runtime_error("Unsupported: Transition::entropy()");
    }

}