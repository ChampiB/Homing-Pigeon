//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_CATEGORICALNODE_H
#define HOMING_PIGEON_2_CATEGORICALNODE_H

#include "FactorNode.h"
#include "api/API.h"
#include <memory>
#include <Eigen/Dense>

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::nodes {

    class CategoricalNode : public FactorNode {
    public:
        static std::unique_ptr<CategoricalNode> create(RV *node, RV *d);
        static std::unique_ptr<CategoricalNode> create(RV *node);

    public:
        CategoricalNode(VarNode *node, VarNode *d);
        explicit CategoricalNode(VarNode *node);
        VarNode *parent(int index) override;
        VarNode *child() override;
        std::vector<Eigen::MatrixXd> message(VarNode *to) override;
        double vfe() override;

    private:
        std::vector<Eigen::MatrixXd> childMessage();
        std::vector<Eigen::MatrixXd> dMessage();
        Eigen::MatrixXd getLogD();

    private:
        VarNode *childNode;
        VarNode *D;
    };

}

#endif //HOMING_PIGEON_2_CATEGORICALNODE_H
