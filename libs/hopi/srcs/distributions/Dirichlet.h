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
        Dirichlet(const std::vector<Eigen::MatrixXd> &p, const std::vector<int> &f);
        [[nodiscard]] DistributionType type() const override;
        [[nodiscard]] std::vector<Eigen::MatrixXd> logParams() const override;
        [[nodiscard]] std::vector<Eigen::MatrixXd> params() const override;
        void updateParams(std::vector<Eigen::MatrixXd> &param) override;
        double entropy() override;
        static double entropy(Eigen::MatrixXd p);
        static std::vector<Eigen::MatrixXd> expectedLog(std::vector<Eigen::MatrixXd> p);
        void setFilters(const std::vector<int> &f); // TODO TEST
        void increaseParam(int matrixId, int rowId, int colId);

    private:
        std::vector<Eigen::MatrixXd> param;
        std::vector<int> filters; // a vector of size 3 defining how many matrices, rows and columns should be updated.
    };

}

#endif //EXPERIMENTS_AI_TS_DIRICHLET_H
