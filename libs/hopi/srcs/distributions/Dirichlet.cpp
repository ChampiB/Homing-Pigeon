//
// Created by tmac3 on 06/01/2021.
//

#include "Dirichlet.h"
#include "nodes/VarNode.h"
#include "math/Functions.h"

using namespace hopi::nodes;
using namespace hopi::math;
using namespace Eigen;

namespace hopi::distributions {

    std::unique_ptr<Dirichlet> Dirichlet::create(const std::vector<Eigen::MatrixXd> &p) {
        return std::make_unique<Dirichlet>(p);
    }

    Dirichlet::Dirichlet(const std::vector<Eigen::MatrixXd> &p) {
        param = p;
    }

    DistributionType Dirichlet::type() const {
        return DistributionType::DIRICHLET;
    }

    [[nodiscard]] std::vector<Eigen::MatrixXd> Dirichlet::logParams() const {
        std::vector<Eigen::MatrixXd> res;

        for (const auto & i : param) {
            res.emplace_back(i.array().log());
        }
        return res;
    }

    [[nodiscard]] std::vector<Eigen::MatrixXd> Dirichlet::params() const {
        std::vector<Eigen::MatrixXd> res;

        for (const auto & i : param) {
            res.emplace_back(i);
        }
        return res;
    }

    void Dirichlet::updateParams(std::vector<Eigen::MatrixXd> &p) {
        for (int i = 0; i < param.size(); ++i) {
            param[i] = p[i];
        }
    }

    double Dirichlet::entropy(MatrixXd p) {
        double e = 0;
        auto s = p.sum();
        auto acc = 0;

        for (int k = 0; k < p.rows(); ++k) {
            acc += (p(k) - 1) * Functions::digamma(p(k));
        }
        e += Functions::log_beta(p) \
          +  (s - p.rows()) * Functions::digamma(s) \
          -  acc;
        return e;
    }

    double Dirichlet::entropy() {
        double e = 0;

        for (auto & i : param) {
            for (int j = 0; j < param[0].cols(); ++j) {
                e += entropy(i.block(0, j, param[0].rows(), 1));
            }
        }
        return e;
    }

    std::vector<MatrixXd> Dirichlet::expectedLog(std::vector<MatrixXd> p) {
        std::vector<MatrixXd> m = p;

        for (int i = 0; i < p.size(); ++i) {
            for (int k = 0; k < p[i].cols(); ++k) {
                MatrixXd col = m[i].col(k);
                for (int j = 0; j < p[i].rows(); ++j) {
                    m[i](j,k) = Functions::digamma(m[i](j, k)) - Functions::digamma(col.sum());
                }
            }
        }
        return m;
    }

    void Dirichlet::increaseParam(int matrixId, int rowId, int colId) {
        ++param[matrixId](rowId, colId);
    }

}
