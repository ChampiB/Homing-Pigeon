//
// Created by tmac3 on 28/11/2020.
//

#include "Categorical.h"
#include "nodes/VarNode.h"
#include "math/Functions.h"
#include <Eigen/Dense>

using namespace hopi::nodes;
using namespace hopi::math;
using namespace Eigen;

namespace hopi::distributions {

    std::unique_ptr<Categorical> Categorical::create(const Eigen::MatrixXd &param) {
        return std::make_unique<Categorical>(param);
    }

    std::unique_ptr<Categorical> Categorical::create(const Eigen::MatrixXd &&param) {
        return std::make_unique<Categorical>(param);
    }

    Categorical::Categorical(const Eigen::MatrixXd &p) {
        param = p;
    }

    Categorical::Categorical(const Eigen::MatrixXd &&p) {
        param = p;
    }

    DistributionType Categorical::type() const {
        return DistributionType::CATEGORICAL;
    }

    int Categorical::cardinality() const {
        return (int)param.size();
    }

    double Categorical::p(int id) const{
        return param(id);
    }

    std::vector<MatrixXd> Categorical::logParams() const {
        MatrixXd copy = param;
        return {copy.array().log()};
    }

    std::vector<MatrixXd> Categorical::params() const {
        MatrixXd copy = param;
        return {copy};
    }

    void Categorical::updateParams(std::vector<Eigen::MatrixXd> &p) {
        if (p.size() != 1) {
            throw std::runtime_error("Categorical::updateParams argument size must be equal to one.");
        }
        param = Functions::softmax(p[0]);
    }

    double Categorical::entropy() {
        double e = 0;
        auto p   = params()[0];
        auto lp  = logParams()[0];

        for (int i = 0; i < p.size(); ++i) {
            if (p(i,0) != 0) {
                e -= p(i,0) * lp(i,0);
            }
        }
        return e;
    }

}
