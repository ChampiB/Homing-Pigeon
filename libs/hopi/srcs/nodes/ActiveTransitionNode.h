//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_ACTIVETRANSITIONNODE_H
#define HOMING_PIGEON_2_ACTIVETRANSITIONNODE_H

#include "FactorNode.h"
#include <Eigen/Dense>

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::nodes {

    class ActiveTransitionNode : public FactorNode {
    public:
        ActiveTransitionNode(VarNode *from, VarNode *action, VarNode *to, VarNode *B);
        ActiveTransitionNode(VarNode *from, VarNode *action, VarNode *to);
        VarNode *parent(int index) override;
        VarNode *child() override;
        std::vector<Eigen::MatrixXd> message(VarNode *to) override;
        double vfe() override;

    private:
        std::vector<Eigen::MatrixXd> toMessage();
        std::vector<Eigen::MatrixXd> fromMessage();
        std::vector<Eigen::MatrixXd> actionMessage();
        std::vector<Eigen::MatrixXd> bMessage();
        std::vector<Eigen::MatrixXd> getLogB();

    private:
        VarNode *from;
        VarNode *action;
        VarNode *to;
        VarNode *B;
    };

}

#endif //HOMING_PIGEON_2_ACTIVETRANSITIONNODE_H
