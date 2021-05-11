//
// Created by tmac3 on 28/11/2020.
//

#include "ActiveTransition.h"
#include "nodes/VarNode.h"
#include "math/Functions.h"

using namespace hopi::nodes;
using namespace hopi::math;
using namespace Eigen;

namespace hopi::distributions {

    std::unique_ptr<ActiveTransition> ActiveTransition::create(const std::vector<MatrixXd> &p) {
        return std::make_unique<ActiveTransition>(p);
    }

    ActiveTransition::ActiveTransition(const std::vector<MatrixXd> &p) {
        param = p;
    }

    [[nodiscard]] DistributionType ActiveTransition::type() const {
        return DistributionType::ACTIVE_TRANSITION;
    }

    std::vector<MatrixXd> ActiveTransition::logParams() const {
        std::vector<MatrixXd> res(param.size());

        for (int i = 0; i < param.size(); ++i) {
            res[i] = param[i];
            res[i] = res[i].array().log();
        }
        return res;
    }

    std::vector<MatrixXd> ActiveTransition::params() const {
        std::vector<MatrixXd> res(param.size());

        for (int i = 0; i < param.size(); ++i) {
            res[i] = param[i];
        }
        return res;
    }

    void ActiveTransition::updateParams(std::vector<Eigen::MatrixXd> &p) {
        if (p.size() != param.size()) {
            throw std::runtime_error("ActiveTransition::updateParams argument size must match parameter size.");
        }
        for (int i = 0; i < p.size(); ++i) {
            param[i] = Functions::softmax(p[i]);
        }
    }

    double ActiveTransition::entropy() {
        throw std::runtime_error("Unsupported: ActiveTransition::entropy()");
    }

}
