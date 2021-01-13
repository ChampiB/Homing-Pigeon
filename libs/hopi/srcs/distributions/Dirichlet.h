//
// Created by tmac3 on 06/01/2021.
//

#ifndef EXPERIMENTS_AI_TS_DIRICHLET_H
#define EXPERIMENTS_AI_TS_DIRICHLET_H

#include "Distribution.h"

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::distributions {

    class Dirichlet : public Distribution {
    public:
        static nodes::VarNode *create(const std::vector<Eigen::MatrixXd>& param); //TODO test
        static nodes::VarNode *create(const Eigen::MatrixXd& param); //TODO test

    public:
        explicit Dirichlet(const std::vector<Eigen::MatrixXd> &param); //TODO test
        [[nodiscard]] DistributionType type() const override; //TODO test
        [[nodiscard]] std::vector<Eigen::MatrixXd> logParams() const override; //TODO test
        [[nodiscard]] std::vector<Eigen::MatrixXd> params() const override; //TODO test
        void setParams(std::vector<Eigen::MatrixXd> &param) override; //TODO test
        double entropy() override; //TODO test
        static double entropy(Eigen::MatrixXd p); //TODO test
        static std::vector<Eigen::MatrixXd> expectedLog(std::vector<Eigen::MatrixXd> p); //TODO test

    private:
        std::vector<Eigen::MatrixXd> param;
    };

}

#endif //EXPERIMENTS_AI_TS_DIRICHLET_H
