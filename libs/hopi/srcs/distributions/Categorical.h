//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_CATEGORICAL_H
#define HOMING_PIGEON_2_CATEGORICAL_H

#include "Distribution.h"
#include <memory>
#include <Eigen/Dense>

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::distributions {

    class Categorical : public Distribution {
    public:
        static nodes::VarNode *create(const Eigen::MatrixXd& param);

    public:
        explicit Categorical(Eigen::MatrixXd param);
        [[nodiscard]] DistributionType type() const override;
        [[nodiscard]] int cardinality() const;
        [[nodiscard]] double p(int id) const; // Probability of X = id.
        [[nodiscard]] std::vector<Eigen::MatrixXd> logProbability() const override;
        [[nodiscard]] std::vector<Eigen::MatrixXd> probability() const override;

    private:
        Eigen::MatrixXd param;
    };

}

#endif //HOMING_PIGEON_2_CATEGORICAL_H
