//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_ACTIVETRANSITION_H
#define HOMING_PIGEON_2_ACTIVETRANSITION_H

#include "Distribution.h"
#include <vector>
#include <Eigen/Dense>

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::distributions {

    class ActiveTransition : public Distribution {
    public:
        static nodes::VarNode *create(
            nodes::VarNode *s, nodes::VarNode *a, const std::vector<Eigen::MatrixXd>& param
        );
        static nodes::VarNode *create(
            nodes::VarNode *s, nodes::VarNode *a, nodes::VarNode *param
        );

    public:
        explicit ActiveTransition(std::vector<Eigen::MatrixXd> param);
        [[nodiscard]] DistributionType type() const override;
        [[nodiscard]] std::vector<Eigen::MatrixXd> logParams() const override;
        [[nodiscard]] std::vector<Eigen::MatrixXd> params() const override;
        void updateParams(std::vector<Eigen::MatrixXd> &p) override;
        double entropy() override;

    private:
        std::vector<Eigen::MatrixXd> param;
    };

}

#endif //HOMING_PIGEON_2_ACTIVETRANSITION_H
