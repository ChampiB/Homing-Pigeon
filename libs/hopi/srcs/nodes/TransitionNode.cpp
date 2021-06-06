//
// Created by Theophile Champion on 28/11/2020.
//

#include "TransitionNode.h"
#include <torch/torch.h>
#include "nodes/VarNode.h"
#include "math/Ops.h"
#include "distributions/Dirichlet.h"

using namespace hopi::nodes;
using namespace hopi::math;
using namespace hopi::distributions;
using namespace torch;

namespace hopi::nodes {

    std::unique_ptr<TransitionNode> TransitionNode::create(RV *from, RV *to, RV *param) {
        return std::make_unique<TransitionNode>(from, to, param);
    }

    std::unique_ptr<TransitionNode> TransitionNode::create(RV *from, RV *to) {
        return std::make_unique<TransitionNode>(from, to);
    }

    TransitionNode::TransitionNode(VarNode *f, VarNode *t, VarNode *param) {
        from = f;
        to = t;
        A = param;
    }

    TransitionNode::TransitionNode(VarNode *f, VarNode *t) : TransitionNode(f, t, nullptr) {}

    VarNode *TransitionNode::parent(int i) {
        if (i == 0)
            return from;
        else if (i == 1)
            return A;
        else
            return nullptr;
    }

    VarNode *TransitionNode::child() {
        return to;
    }

    Tensor TransitionNode::message(VarNode *t) {
        if (t == to) {
            return toMessage();
        } else if (t == from) {
            return fromMessage();
        } else if (A && t == A) {
            return aMessage();
        } else {
            assert(false && "TransitionNode::message, invalid input node.");
        }
    }

    Tensor TransitionNode::toMessage() {
        return matmul(getLogA(), from->posterior()->params());
    }

    Tensor TransitionNode::fromMessage() {
        return matmul(getLogA().permute({1,0}), to->posterior()->params());
    }

    Tensor TransitionNode::aMessage() {
        return outer(from->posterior()->params(), to->posterior()->params());
    }

    double TransitionNode::vfe() {
        double VFE = 0;

        if (child()->type() == HIDDEN) {
            VFE -= child()->posterior()->entropy();
        }
        auto lp = Ops::average(getLogA(), from->posterior()->params(), {1});
        return VFE - Ops::average(lp, to->posterior()->params(), {0}).item<double>();
    }

    Tensor TransitionNode::getLogA() {
        if (A) {
            return Dirichlet::expectedLog(A->posterior()->params()).permute({1,0});
        } else {
            return to->prior()->logParams();
        }
    }

}

