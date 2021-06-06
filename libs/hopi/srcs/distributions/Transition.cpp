//
// Created by Theophile Champion on 28/11/2020.
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
        assert(false && "Transition::updateParams, unsupported.");
    }

    double Transition::entropy() {
        assert(false && "Transition::entropy, unsupported.");
    }

}