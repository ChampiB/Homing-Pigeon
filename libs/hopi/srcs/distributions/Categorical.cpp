//
// Created by Theophile Champion on 28/11/2020.
//

#include "Categorical.h"
#include "nodes/VarNode.h"
#include "math/Ops.h"

using namespace hopi::nodes;
using namespace hopi::math;
using namespace torch;

namespace hopi::distributions {

    std::unique_ptr<Categorical> Categorical::create(const Tensor &param) {
        return std::make_unique<Categorical>(param);
    }

    std::unique_ptr<Categorical> Categorical::create(const Tensor &&param) {
        return std::make_unique<Categorical>(param);
    }

    Categorical::Categorical(const Tensor &p) {
        param = p;
    }

    Categorical::Categorical(const Tensor &&p) {
        param = p;
    }

    DistributionType Categorical::type() const {
        return DistributionType::CATEGORICAL;
    }

    Tensor Categorical::logParams() const {
        return params().log();
    }

    Tensor Categorical::params() const {
        return param.detach().clone();
    }

    void Categorical::updateParams(const Tensor &p) {
        assert(p.dim() == 1 && "Categorical::updateParams, input must have dimension one.");
        param = softmax(p, 0);
    }

    double Categorical::entropy() {
        Tensor p = params();
        Tensor indexes = where(p != 0, true, false);

        return -1 * (p * logParams()).index({indexes}).sum().item<double>();
    }

}
