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
        explicit CategoricalNode(VarNode *node);
        VarNode *parent(int index) override;
        VarNode *child() override;
        Eigen::MatrixXd message(VarNode *to) override;
        double vfe() override;

    private:
        Eigen::MatrixXd childMessage();

    private:
        VarNode *childNode;
    };

}

#endif //HOMING_PIGEON_2_CATEGORICALNODE_H
