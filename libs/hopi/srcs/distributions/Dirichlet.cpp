//
// Created by tmac3 on 06/01/2021.
//

#include "Dirichlet.h"
#include "nodes/VarNode.h"
#include "math/Ops.h"

using namespace hopi::nodes;
using namespace hopi::math;
using namespace torch;
using namespace torch::indexing;

namespace hopi::distributions {

    std::unique_ptr<Dirichlet> Dirichlet::create(const Tensor &p) {
        return std::make_unique<Dirichlet>(p);
    }

    Dirichlet::Dirichlet(const Tensor &p) {
        param = p;
    }

    DistributionType Dirichlet::type() const {
        return DistributionType::DIRICHLET;
    }

    [[nodiscard]] Tensor Dirichlet::logParams() const {
        return params().log();
    }

    [[nodiscard]] Tensor Dirichlet::params() const {
        return param.detach().clone();
    }

    void Dirichlet::updateParams(const Tensor &p) {
        param = p;
    }

    double Dirichlet::entropy(const Tensor &&p) {
        if (p.dim() != 1) {
            throw std::runtime_error("Unsupported call to Dirichlet::entropy (input) tensor's dimension must be one.");
        }
        double e = 0;
        auto s = p.sum().item<double>();
        auto acc = 0.0;

        for (int k = 0; k < p.size(0); ++k) {
            acc += (p[k].item<double>() - 1) * Ops::digamma(p[k].item<double>());
        }
        e += Ops::log_beta(p) \
          +  (s - (double) p.size(0)) * Ops::digamma(s) \
          -  acc;
        return e;
    }

    double Dirichlet::entropy() {
        Tensor e = torch::zeros({1});

        for (int i = 0; i < param.size(0); ++i) {
            for (int j = 0; j < param.size(2); ++j) {
                e += entropy(param.index({i, None, j}));
            }
        }
        return e.item<double>();
    }

    Tensor Dirichlet::expectedLog(const Tensor &p) {
        Tensor m = p;

        for (int i = 0; i < p.size(0); ++i) {
            for (int k = 0; k < p.size(2); ++k) {
                auto sum = m.index({i, None, k}).sum().item<double>();
                for (int j = 0; j < p.size(1); ++j) {
                    m[i][j][k] = Ops::digamma(m[i][j][k].item<double>()) - Ops::digamma(sum);
                }
            }
        }
        return m;
    }

    void Dirichlet::increaseParam(int matrixId, int rowId, int colId) {
        param[matrixId][rowId][colId] += 1;
    }

}
