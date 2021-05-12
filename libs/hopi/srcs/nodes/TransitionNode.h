//
// Created by tmac3 on 28/11/2020.
//

#ifndef HOMING_PIGEON_2_TRANSITIONNODE_H
#define HOMING_PIGEON_2_TRANSITIONNODE_H

#include "FactorNode.h"
#include "api/Aliases.h"
#include <torch/torch.h>
#include <memory>

namespace hopi::nodes {
    class VarNode;
}

namespace hopi::nodes {

    class TransitionNode : public FactorNode {
    public:
        static std::unique_ptr<TransitionNode> create(RV *from, RV *to, RV *a);
        static std::unique_ptr<TransitionNode> create(RV *from, RV *to);

    public:
        TransitionNode(VarNode *from, VarNode *to, VarNode *a);
        TransitionNode(VarNode *from, VarNode *to);
        VarNode *parent(int index) override;
        VarNode *child() override;
        torch::Tensor message(VarNode *to) override;
        double vfe() override;

    private:
        torch::Tensor toMessage();
        torch::Tensor fromMessage();
        torch::Tensor aMessage();
        torch::Tensor getLogA();

    private:
        VarNode *from;
        VarNode *to;
        VarNode *A;
    };

}

#endif //HOMING_PIGEON_2_TRANSITIONNODE_H
