//
// Created by Theophile Champion on 06/01/2021.
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
        assert(param.dim() == p.dim() && "Dirichlet::updateParams, inputs must have the same dimensions.");
        assert(param.sizes() == p.sizes() && "Dirichlet::updateParams, inputs must have the same sizes.");
        param = p;
    }

    double Dirichlet::entropy(const Tensor &&p) {
        assert(p.dim() == 1 && "Dirichlet::entropy, input must have dimension one.");
        auto p_a = p.accessor<double,1>();
        double sum = 0;
        double acc = 0;

        for (int k = 0; k < p_a.size(0); ++k) {
            acc += (p_a[k] - 1) * Ops::digamma(p_a[k]);
            sum += p_a[k];
        }
        return Ops::log_beta(p) + (sum - (double) p.size(0)) * Ops::digamma(sum) - acc;
    }

    double Dirichlet::entropy() {
        double e = 0;

        Ops::unsqueeze(3 - param.dim(), {&param});
        for (int i = 0; i < param.size(0); ++i) {
            for (int j = 0; j < param.size(1); ++j) {
                e += entropy(param.index({i, j, Ellipsis}));
            }
        }
        param = torch::squeeze(param);
        return e;
    }

    Tensor Dirichlet::expectedLog(const Tensor &p) {
        // Make sure input are valid
        assert(p.dim() <= 3 && "Dirichlet::expectedLog does not support Dirichlet of dimension superior to three");

        // Compute the expected log
        Tensor m = p.detach().clone();

        Ops::unsqueeze(3 - m.dim(), {&m});
        for (int i = 0; i < m.size(0); ++i) {
            for (int j = 0; j < m.size(1); ++j) {
                auto sum = m.index({i,j,Ellipsis}).sum().item<double>();
                for (int k = 0; k < m.size(2); ++k) {
                    m[i][j][k] = Ops::digamma(m[i][j][k].item<double>()) - Ops::digamma(sum);
                }
            }
        }
        return torch::squeeze(m);
    }

}
