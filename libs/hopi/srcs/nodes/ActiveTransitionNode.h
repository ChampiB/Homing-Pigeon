//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_ACTIVETRANSITIONNODE_H
#define HOMING_PIGEON_2_ACTIVETRANSITIONNODE_H

#include "FactorNode.h"
#include <torch/torch.h>
#include <api/Aliases.h>
#include <memory>

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::nodes {

    class ActiveTransitionNode : public FactorNode {
    public:
        static std::unique_ptr<ActiveTransitionNode> create(RV *from, RV *action, RV *to, RV *B);
        static std::unique_ptr<ActiveTransitionNode> create(RV *from, RV *action, RV *to);

    public:
        ActiveTransitionNode(VarNode *from, VarNode *action, VarNode *to, VarNode *B);
        ActiveTransitionNode(VarNode *from, VarNode *action, VarNode *to);
        VarNode *parent(int index) override;
        VarNode *child() override;
        torch::Tensor message(VarNode *to) override;
        double vfe() override;

    private:
        torch::Tensor toMessage();
        torch::Tensor fromMessage();
        torch::Tensor actionMessage();
        torch::Tensor bMessage();
        torch::Tensor getLogB();

    private:
        VarNode *from;
        VarNode *action;
        VarNode *to;
        VarNode *B;
    };

}

#endif //HOMING_PIGEON_2_ACTIVETRANSITIONNODE_H
