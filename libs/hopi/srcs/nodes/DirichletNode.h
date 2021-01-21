//
// Created by tmac3 on 06/01/2021.
//

#ifndef EXPERIMENTS_AI_TS_DIRICHLETNODE_H
#define EXPERIMENTS_AI_TS_DIRICHLETNODE_H

#include "FactorNode.h"

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::nodes {

    class DirichletNode : public FactorNode {
    public:
        explicit DirichletNode(VarNode *node);
        VarNode *parent(int index) override;
        VarNode *child() override;
        std::vector<Eigen::MatrixXd> message(VarNode *to) override;
        double vfe() override;
        static double energy(Eigen::MatrixXd prior, Eigen::MatrixXd post);

    private:
        std::vector<Eigen::MatrixXd> childMessage();

    private:
        VarNode *childNode;
    };

}


#endif //EXPERIMENTS_AI_TS_DIRICHLETNODE_H
