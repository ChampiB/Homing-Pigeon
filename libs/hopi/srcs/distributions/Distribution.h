//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_DISTRIBUTION_H
#define HOMING_PIGEON_2_DISTRIBUTION_H

#include <vector>
#include <torch/torch.h>
#include "DistributionType.h"

namespace hopi::distributions {

    class Distribution {
    public:
        [[nodiscard]] virtual DistributionType type() const = 0;
        [[nodiscard]] virtual torch::Tensor logParams() const = 0;
        [[nodiscard]] virtual torch::Tensor params() const = 0;
        virtual void updateParams(const torch::Tensor &param) = 0;
        virtual double entropy() = 0;
    };

}

#endif //HOMING_PIGEON_2_DISTRIBUTION_H
