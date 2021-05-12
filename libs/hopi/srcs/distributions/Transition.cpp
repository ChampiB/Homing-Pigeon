//
// Created by tmac3 on 28/11/2020.
//

#include "Transition.h"
#include "nodes/VarNode.h"
#include "math/Ops.h"

using namespace hopi::nodes;
using namespace hopi::math;
using namespace torch;

namespace hopi::distributions {

    std::unique_ptr<Transition> Transition::create(const Tensor &p) {
        return std::make_unique<Transition>(p);
    }

    Transition::Transition(const Tensor &p) {
        param = p;
    }

    [[nodiscard]] DistributionType Transition::type() const {
        return DistributionType::TRANSITION;
    }

    Tensor Transition::logParams() const {
        return params().log();
    }

    Tensor Transition::params() const {
        return param.detach().clone();
    }

    void Transition::updateParams(const Tensor &p) {
        if (p.dim() != 1) {
            throw std::runtime_error("Transition::updateParams argument size must be equal to one.");
        }
        param = torch::softmax(p, 0);
    }

    double Transition::entropy() {
        throw std::runtime_error("Unsupported: Transition::entropy()");
    }

}