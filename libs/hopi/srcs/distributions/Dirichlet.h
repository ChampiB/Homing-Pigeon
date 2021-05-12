//
// Created by tmac3 on 06/01/2021.
//

#ifndef EXPERIMENTS_AI_TS_DIRICHLET_H
#define EXPERIMENTS_AI_TS_DIRICHLET_H

#include "Distribution.h"
#include <memory>
#include <torch/torch.h>

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::distributions {

    class Dirichlet : public Distribution {
    public:
        static std::unique_ptr<Dirichlet> create(const torch::Tensor &p);

    public:
        explicit Dirichlet(const torch::Tensor &param);
        [[nodiscard]] DistributionType type() const override;
        [[nodiscard]] torch::Tensor logParams() const override;
        [[nodiscard]] torch::Tensor params() const override;
        void updateParams(const torch::Tensor &param) override;
        double entropy() override;
        static double entropy(const torch::Tensor &&p);
        static torch::Tensor expectedLog(const torch::Tensor &p);
        void increaseParam(int matrixId, int rowId, int colId);

    private:
        torch::Tensor param;
    };

}

#endif //EXPERIMENTS_AI_TS_DIRICHLET_H
