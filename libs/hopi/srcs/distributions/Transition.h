//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_TRANSITION_H
#define HOMING_PIGEON_2_TRANSITION_H

#include "Distribution.h"
#include <memory>
#include <Eigen/Dense>

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::distributions {

    class Transition : public Distribution {
    public:
        static nodes::VarNode *create(nodes::VarNode *s, const Eigen::MatrixXd& param);

    public:
        explicit Transition(Eigen::MatrixXd param);
        [[nodiscard]] DistributionType type() const override;
        [[nodiscard]] std::vector<Eigen::MatrixXd> logParams() const override;
        [[nodiscard]] std::vector<Eigen::MatrixXd> params() const override;
        void setParams(std::vector<Eigen::MatrixXd> &param) override; //TODO test
        double entropy() override; //TODO test

    private:
        Eigen::MatrixXd param;
    };

}

#endif //HOMING_PIGEON_2_TRANSITION_H
