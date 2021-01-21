//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_DISTRIBUTION_H
#define HOMING_PIGEON_2_DISTRIBUTION_H

#include <vector>
#include <Eigen/Dense>
#include "DistributionType.h"

namespace hopi::distributions {

    class Distribution {
    public:
        [[nodiscard]] virtual DistributionType type() const = 0;
        [[nodiscard]] virtual std::vector<Eigen::MatrixXd> logParams() const = 0;
        [[nodiscard]] virtual std::vector<Eigen::MatrixXd> params() const = 0;
        virtual void updateParams(std::vector<Eigen::MatrixXd> &param) = 0;
        virtual double entropy() = 0;
    };

}

#endif //HOMING_PIGEON_2_DISTRIBUTION_H
