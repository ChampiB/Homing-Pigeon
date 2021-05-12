//
// Created by tmac3 on 28/11/2020.
//

#include "TransitionNode.h"
#include <torch/torch.h>
#include "nodes/VarNode.h"
#include "distributions/Dirichlet.h"

using namespace hopi::nodes;
using namespace hopi::distributions;
using namespace torch;

namespace hopi::nodes {

    std::unique_ptr<TransitionNode> TransitionNode::create(RV *from, RV *to, RV *a) {
        return std::make_unique<TransitionNode>(from, to, a);
    }

    std::unique_ptr<TransitionNode> TransitionNode::create(RV *from, RV *to) {
        return std::make_unique<TransitionNode>(from, to);
    }

    TransitionNode::TransitionNode(VarNode *f, VarNode *t, VarNode *a) {
        from = f;
        to = t;
        A = a;
    }

    TransitionNode::TransitionNode(VarNode *f, VarNode *t) : TransitionNode(f, t, nullptr) {}

    VarNode *TransitionNode::parent(int index) {
        if (index == 0)
            return from;
        else if (index == 1)
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
            throw std::runtime_error("Unsupported: Message towards non-adjacent node.");
        }
    }

    Tensor TransitionNode::toMessage() {
        Tensor A_bar = getLogA();
        Tensor hat = from->posterior()->params()[0];
        return {A_bar * hat};
    }

    Tensor TransitionNode::fromMessage() {
        Tensor A_bar = getLogA();
        Tensor hat = to->posterior()->params()[0];
        return matmul(A_bar.permute({1,0}), hat);
    }

    Tensor TransitionNode::aMessage() {
        Tensor o =   to->posterior()->params()[0];
        Tensor s = from->posterior()->params()[0];
        return matmul(o, s.permute({1,0}));
    }

    double TransitionNode::vfe() {
        double VFE = 0;

        if (child()->type() == HIDDEN) {
            VFE -= child()->posterior()->entropy();
        }
        auto to_p   = to->posterior()->params()[0];
        auto from_p = from->posterior()->params()[0];
        auto lp     = getLogA();
        return VFE - matmul(matmul(to_p.permute({1,0}), lp), from_p).item<double>();
    }

    Tensor TransitionNode::getLogA() {
        if (A) {
            return Dirichlet::expectedLog(A->posterior()->params())[0];
        } else {
            return to->prior()->logParams()[0];
        }
    }

}

