//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_TRANSITION_H
#define HOMING_PIGEON_2_TRANSITION_H

#include "Distribution.h"
#include <memory>
#include <torch/torch.h>

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::distributions {

    class Transition : public Distribution {
    public:
        static std::unique_ptr<Transition> create(const torch::Tensor &p);

    public:
        explicit Transition(const torch::Tensor &param);
        [[nodiscard]] DistributionType type() const override;
        [[nodiscard]] torch::Tensor logParams() const override;
        [[nodiscard]] torch::Tensor params() const override;
        void updateParams(const torch::Tensor &param) override;
        double entropy() override;

    private:
        torch::Tensor param;
    };

}

#endif //HOMING_PIGEON_2_TRANSITION_H
