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
        static nodes::VarNode *create(const std::vector<Eigen::MatrixXd>& param);
        static nodes::VarNode *create(const Eigen::MatrixXd& param);

    public:
        explicit Dirichlet(const std::vector<Eigen::MatrixXd> &param);
        [[nodiscard]] DistributionType type() const override;
        [[nodiscard]] std::vector<Eigen::MatrixXd> logParams() const override;
        [[nodiscard]] std::vector<Eigen::MatrixXd> params() const override;
        void updateParams(std::vector<Eigen::MatrixXd> &param) override;
        double entropy() override;
        static double entropy(Eigen::MatrixXd p);
        static std::vector<Eigen::MatrixXd> expectedLog(std::vector<Eigen::MatrixXd> p);
        void increaseParam(int matrixId, int rowId, int colId);

    private:
        std::vector<Eigen::MatrixXd> param;
    };

}

#endif //EXPERIMENTS_AI_TS_DIRICHLET_H
