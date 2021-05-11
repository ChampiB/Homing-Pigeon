//
// Created by tmac3 on 28/11/2020.
//

#include "Transition.h"
#include "nodes/VarNode.h"
#include "math/Functions.h"
#include <Eigen/Dense>

using namespace hopi::nodes;
using namespace hopi::math;
using namespace Eigen;

namespace hopi::distributions {

    std::unique_ptr<Transition> Transition::create(const MatrixXd &p) {
        return std::make_unique<Transition>(p);
    }

    Transition::Transition(const Eigen::MatrixXd &p) {
        param = p;
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

    void Transition::updateParams(std::vector<Eigen::MatrixXd> &p) {
        if (p.size() != 1) {
            throw std::runtime_error("Transition::updateParams argument size must be equal to one.");
        }
        param = Functions::softmax(p[0]);
    }

    double Transition::entropy() {
        throw std::runtime_error("Unsupported: Transition::entropy()");
    }

}