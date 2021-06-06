//
// Created by Theophile Champion on 28/11/2020.
//

#include "ActiveTransition.h"
#include "nodes/VarNode.h"
#include "math/Ops.h"

using namespace hopi::nodes;
using namespace hopi::math;
using namespace torch;

namespace hopi::distributions {

    std::unique_ptr<ActiveTransition> ActiveTransition::create(const Tensor &p) {
        return std::make_unique<ActiveTransition>(p);
    }

    ActiveTransition::ActiveTransition(const Tensor &p) {
        param = p;
    }

    [[nodiscard]] DistributionType ActiveTransition::type() const {
        return DistributionType::ACTIVE_TRANSITION;
    }

    Tensor ActiveTransition::logParams() const {
        return params().log();
    }

    Tensor ActiveTransition::params() const {
        return param.detach().clone();
    }

    void ActiveTransition::updateParams(const Tensor &p) {
        assert(false && "ActiveTransition::updateParams, unsupported.");
    }

    double ActiveTransition::entropy() {
        assert(false && "ActiveTransition::entropy, unsupported.");
    }

}
