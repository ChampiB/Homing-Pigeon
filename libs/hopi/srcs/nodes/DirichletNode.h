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
        explicit DirichletNode(VarNode *node); //TODO test
        VarNode *parent(int index) override; //TODO test
        VarNode *child() override; //TODO test
        std::vector<Eigen::MatrixXd> message(VarNode *to) override; //TODO test
        double vfe() override; //TODO test
        static double energy(Eigen::MatrixXd prior, Eigen::MatrixXd post); //TODO test

    private:
        std::vector<Eigen::MatrixXd> childMessage(); //TODO test

    private:
        VarNode *childNode;
    };

}


#endif //EXPERIMENTS_AI_TS_DIRICHLETNODE_H
