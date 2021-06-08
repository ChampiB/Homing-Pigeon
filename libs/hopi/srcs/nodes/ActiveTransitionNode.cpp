//
// Created by Theophile Champion on 28/11/2020.
//

#include "ActiveTransitionNode.h"
#include "distributions/Dirichlet.h"
#include "nodes/VarNode.h"
#include "math/Ops.h"

using namespace torch;
using namespace hopi::distributions;
using namespace hopi::math;

namespace hopi::nodes {

    std::unique_ptr<ActiveTransitionNode> ActiveTransitionNode::create(RV *f, RV *a, RV *t, RV *b) {
        return std::make_unique<ActiveTransitionNode>(f, a, t, b);
    }

    std::unique_ptr<ActiveTransitionNode> ActiveTransitionNode::create(RV *f, RV *a, RV *t) {
        return std::make_unique<ActiveTransitionNode>(f, a, t);
    }

    ActiveTransitionNode::ActiveTransitionNode(VarNode *f, VarNode *a, VarNode *t, VarNode *b) {
        from = f;
        action = a;
        to = t;
        B = b;
    }

    ActiveTransitionNode::ActiveTransitionNode(VarNode *f, VarNode *a, VarNode *t)
      : ActiveTransitionNode(f, a, t, nullptr) {}

    VarNode *ActiveTransitionNode::parent(int i) {
        switch (i) {
            case 0:
                return from;
            case 1:
                return action;
            case 2:
                return B;
            default:
                return nullptr;
        }
    }

    VarNode *ActiveTransitionNode::child() {
        return to;
    }

    Tensor ActiveTransitionNode::message(VarNode *t) {
        if (t == to) {
            return toMessage();
        } else if (t == from) {
            return fromMessage();
        } else if (t == action) {
            return actionMessage();
        } else if (B && t == B) {
            return bMessage();
        } else {
            assert(false && "ActiveTransitionNode::message, invalid input node.");
        }
    }

    Tensor ActiveTransitionNode::toMessage() {
        Tensor B_bar = Ops::average(getLogB(), action->posterior()->params(), {2});
        return Ops::average(B_bar, from->posterior()->params(), {1});
    }

    Tensor ActiveTransitionNode::fromMessage() {
        Tensor B_bar = Ops::average(getLogB(), action->posterior()->params(), {2});
        return Ops::average(B_bar, to->posterior()->params(), {0});
    }

    Tensor ActiveTransitionNode::actionMessage() {
        Tensor B_bar = Ops::average(getLogB(), from->posterior()->params(), {1});
        return Ops::average(B_bar, to->posterior()->params(), {0});
    }

    Tensor ActiveTransitionNode::bMessage() {
        Tensor to_hat     =     to->posterior()->params();
        Tensor from_hat   =   from->posterior()->params();
        Tensor action_hat = action->posterior()->params();

        return Ops::outer_tensor_product({&from_hat,&action_hat,&to_hat});
    }

    double ActiveTransitionNode::vfe() {
        auto lp       = Ops::average(getLogB(), action->posterior()->params(), {2});
        double VFE    = 0;

        if (child()->type() == HIDDEN) {
            VFE -= child()->posterior()->entropy();
        }
        lp = Ops::average(lp, from->posterior()->params(), {1});
        lp = Ops::average(lp, to->posterior()->params(), {0});
        VFE -= lp.item<double>();
        return VFE;
    }

    Tensor ActiveTransitionNode::getLogB() {
        if (B) {
            return Dirichlet::expectedLog(B->posterior()->params()).permute({2,0,1});
        } else {
            return to->prior()->logParams();
        }
    }

}
