//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_CATEGORICALNODE_H
#define HOMING_PIGEON_2_CATEGORICALNODE_H

#include "FactorNode.h"
#include <memory>
#include <Eigen/Dense>

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::nodes {

    class CategoricalNode : public FactorNode {
    public:
        CategoricalNode(VarNode *node, VarNode *d);
        explicit CategoricalNode(VarNode *node);
        VarNode *parent(int index) override;
        VarNode *child() override;
        std::vector<Eigen::MatrixXd> message(VarNode *to) override; //TODO test
        double vfe() override;

    private:
        std::vector<Eigen::MatrixXd> childMessage(); //TODO test
        std::vector<Eigen::MatrixXd> dMessage(); //TODO test
        Eigen::MatrixXd getLogD(); //TODO test

    private:
        VarNode *childNode;
        VarNode *D;
    };

}

#endif //HOMING_PIGEON_2_CATEGORICALNODE_H
