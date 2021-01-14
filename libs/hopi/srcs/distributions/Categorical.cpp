//
// Created by tmac3 on 28/11/2020.
//

#include "Categorical.h"
#include "nodes/VarNode.h"
#include "nodes/CategoricalNode.h"
#include "math/Functions.h"
#include "graphs/FactorGraph.h"
#include <Eigen/Dense>

using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::math;
using namespace Eigen;

namespace hopi::distributions {

    VarNode *Categorical::create(const Eigen::MatrixXd& param) {
        std::shared_ptr<FactorGraph> fg = FactorGraph::current();
        VarNode *var = fg->addNode(std::make_unique<VarNode>(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(std::make_unique<CategoricalNode>(var));

        var->setParent(factor);
        var->setPrior(std::make_unique<Categorical>(param));
        var->setPosterior(std::make_unique<Categorical>(
            MatrixXd::Constant(param.rows(), 1, 1.0 / param.rows())
        ));
        return var;
    }

    VarNode *Categorical::create(VarNode *param) {
        std::shared_ptr<FactorGraph> fg = FactorGraph::current();
        VarNode *var = fg->addNode(std::make_unique<VarNode>(VarNodeType::HIDDEN));
        FactorNode *factor = fg->addFactor(std::make_unique<CategoricalNode>(var, param));

        var->setParent(factor);
        auto dim = param->prior()->params()[0].rows();
        var->setPosterior(std::make_unique<Categorical>(
            MatrixXd::Constant(dim, 1, 1.0 / dim)
        ));

        param->addChild(factor);
        return var;
    }

    Categorical::Categorical(Eigen::MatrixXd p) {
        param = std::move(p);
    }

    DistributionType Categorical::type() const {
        return DistributionType::CATEGORICAL;
    }

    int Categorical::cardinality() const {
        return param.rows();
    }

    double Categorical::p(int id) const{
        return param(id);
    }

    std::vector<MatrixXd> Categorical::logParams() const {
        MatrixXd copy = param;
        std::vector<MatrixXd> res {copy.array().log()};
        return res;
    }

    std::vector<MatrixXd> Categorical::params() const {
        MatrixXd copy = param;
        std::vector<MatrixXd> res {copy.array()};
        return res;
    }

    void Categorical::setParams(std::vector<Eigen::MatrixXd> &p) {
        if (p.size() != 1) {
            throw std::runtime_error("Categorical::setParams argument size must be equal to one.");
        }
        param = Functions::softmax(p[0]);
    }

    double Categorical::entropy() {
        double e = 0;
        auto p   = params()[0];
        auto lp  = logParams()[0];

        for (int i = 0; i < p.rows(); ++i) {
            if (p(i,0) != 0) {
                e -= p(i,0) * lp(i,0);
            }
        }
        return e;
    }

}
