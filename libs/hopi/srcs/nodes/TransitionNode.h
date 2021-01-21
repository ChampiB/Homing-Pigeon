//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_TRANSITIONNODE_H
#define HOMING_PIGEON_2_TRANSITIONNODE_H

#include "FactorNode.h"
#include <Eigen/Dense>

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::nodes {

    class TransitionNode : public FactorNode {
    public:
        TransitionNode(VarNode *from, VarNode *to, VarNode *a);
        TransitionNode(VarNode *from, VarNode *to);
        VarNode *parent(int index) override;
        VarNode *child() override;
        std::vector<Eigen::MatrixXd> message(VarNode *to) override;
        double vfe() override;

    private:
        std::vector<Eigen::MatrixXd> toMessage();
        std::vector<Eigen::MatrixXd> fromMessage();
        std::vector<Eigen::MatrixXd> aMessage();
        Eigen::MatrixXd getLogA();

    private:
        VarNode *from;
        VarNode *to;
        VarNode *A;
    };

}

#endif //HOMING_PIGEON_2_TRANSITIONNODE_H
