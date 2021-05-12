//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_CATEGORICAL_H
#define HOMING_PIGEON_2_CATEGORICAL_H

#include "Distribution.h"
#include <memory>
#include <torch/torch.h>

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::distributions {

    class Categorical : public Distribution {
    public:
        static std::unique_ptr<Categorical> create(const torch::Tensor &param);
        static std::unique_ptr<Categorical> create(const torch::Tensor &&param);

    public:
        explicit Categorical(const torch::Tensor &param);
        explicit Categorical(const torch::Tensor &&param);
        [[nodiscard]] DistributionType type() const override;
        [[nodiscard]] torch::Tensor p(int id) const; // Probability of X = id.
        [[nodiscard]] torch::Tensor logParams() const override;
        [[nodiscard]] torch::Tensor params() const override;
        void updateParams(const torch::Tensor &param) override;
        double entropy() override;

    private:
        torch::Tensor param;
    };

}

#endif //HOMING_PIGEON_2_CATEGORICAL_H
