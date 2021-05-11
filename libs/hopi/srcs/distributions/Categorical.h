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
        static std::unique_ptr<Categorical> create(const Eigen::MatrixXd &param);
        static std::unique_ptr<Categorical> create(const Eigen::MatrixXd &&param);

    public:
        explicit Categorical(const Eigen::MatrixXd &param);
        explicit Categorical(const Eigen::MatrixXd &&param);
        [[nodiscard]] DistributionType type() const override;
        [[nodiscard]] int cardinality() const;
        [[nodiscard]] double p(int id) const; // Probability of X = id.
        [[nodiscard]] std::vector<Eigen::MatrixXd> logParams() const override;
        [[nodiscard]] std::vector<Eigen::MatrixXd> params() const override;
        void updateParams(std::vector<Eigen::MatrixXd> &param) override;
        double entropy() override;

    private:
        Eigen::MatrixXd param;
    };

}

#endif //HOMING_PIGEON_2_CATEGORICAL_H
