//
// Created by tmac3 on 28/11/2020.
//

#include "ActiveTransitionNode.h"
#include "distributions/Distribution.h"
#include "nodes/VarNode.h"
#include "distributions/Dirichlet.h"

using namespace torch;
using namespace hopi::distributions;

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

    VarNode *ActiveTransitionNode::parent(int index) {
        switch (index) {
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
            throw std::runtime_error("Unsupported: Message towards non-adjacent node.");
        }
    }

    Tensor ActiveTransitionNode::toMessage() {
        Tensor B_bar = getLogB();
        Tensor from_hat = from->posterior()->params();
        Tensor action_hat = action->posterior()->params();

        for (int i = 0; i < B_bar.size(0); ++i) {
            B_bar[i] = action_hat[i] * matmul(B_bar[i], from_hat);
            if (i != 0) {
                B_bar[0] += B_bar[i];
            }
        }
        return {B_bar[0]};
    }

    Tensor ActiveTransitionNode::fromMessage() {
        Tensor B_bar = getLogB();
        Tensor to_hat = to->posterior()->params();
        Tensor action_hat = action->posterior()->params();

        for (int i = 0; i < B_bar.size(0); ++i) {
            B_bar[i] = action_hat[i] * matmul(B_bar[i].permute({1,0}), to_hat);
            if (i != 0) {
                B_bar[0] += B_bar[i];
            }
        }
        return {B_bar[0]};
    }

    Tensor ActiveTransitionNode::actionMessage() {
        Tensor B_bar = getLogB();
        Tensor to_hat   =   to->posterior()->params();
        Tensor from_hat = from->posterior()->params();
        Tensor msg = torch::empty({B_bar.size(0)});

        for (int i = 0; i < B_bar.size(0); ++i) {
            msg[i] = matmul(matmul(to_hat.permute({1,0}), B_bar[i]), from_hat);
        }
        return msg;
    }

    Tensor ActiveTransitionNode::bMessage() {
        Tensor to_hat     =     to->posterior()->params();
        Tensor from_hat   =   from->posterior()->params();
        Tensor action_hat = action->posterior()->params();
        long actions     = action_hat.size(0);
        long to_states   = to_hat.size(0);
        long from_states = from_hat.size(0);
        Tensor msg = torch::zeros({actions, to_states, from_states});

        msg[actions - 1] = matmul(to_hat, from_hat.permute({1,0}));
        for (int i = 0; i < actions; ++i) {
            for (int j = 0; j < to_states; ++j) {
                for (int k = 0; k < from_states; ++k) {
                    msg[i][j][k] = action_hat[i] * msg[actions - 1][j][k];
                }
            }
        }
        return msg;
    }

    double ActiveTransitionNode::vfe() {
        auto to_p     = to->posterior()->params();
        auto from_p   = from->posterior()->params();
        auto action_p = action->posterior()->params();
        auto lp       = getLogB();
        double VFE    = 0;

        if (child()->type() == HIDDEN) {
            VFE -= child()->posterior()->entropy();
        }
        for (int i = 0; i < lp.size(0); ++i) {
            VFE -= (action_p[i] * matmul(matmul(to_p.permute({1,0}), lp[i]), from_p)).item<double>();
        }
        return VFE;
    }

    Tensor ActiveTransitionNode::getLogB() {
        if (B) {
            return Dirichlet::expectedLog(B->posterior()->params());
        } else {
            return to->prior()->logParams();
        }
    }

}
