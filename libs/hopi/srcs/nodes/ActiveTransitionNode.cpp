//
// Created by tmac3 on 28/11/2020.
//

#include "ActiveTransitionNode.h"
#include "distributions/Distribution.h"
#include "nodes/VarNode.h"
#include "distributions/Dirichlet.h"

using namespace Eigen;
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

    std::vector<Eigen::MatrixXd> ActiveTransitionNode::message(VarNode *t) {
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

    std::vector<MatrixXd> ActiveTransitionNode::toMessage() {
        std::vector<MatrixXd> B_bar = getLogB();
        MatrixXd from_hat = from->posterior()->params()[0];
        MatrixXd action_hat = action->posterior()->params()[0];

        for (int i = 0; i < B_bar.size(); ++i) {
            B_bar[i] = action_hat(i, 0) * B_bar[i] * from_hat;
            if (i != 0) {
                B_bar[0] += B_bar[i];
            }
        }
        return {B_bar[0]};
    }

    std::vector<MatrixXd> ActiveTransitionNode::fromMessage() {
        std::vector<MatrixXd> B_bar = getLogB();
        MatrixXd to_hat = to->posterior()->params()[0];
        MatrixXd action_hat = action->posterior()->params()[0];

        for (int i = 0; i < B_bar.size(); ++i) {
            B_bar[i] = action_hat(i, 0) * B_bar[i].transpose() * to_hat;
            if (i != 0) {
                B_bar[0] += B_bar[i];
            }
        }
        return {B_bar[0]};
    }

    std::vector<MatrixXd> ActiveTransitionNode::actionMessage() {
        std::vector<MatrixXd> B_bar = getLogB();
        MatrixXd to_hat   =   to->posterior()->params()[0];
        MatrixXd from_hat = from->posterior()->params()[0];
        MatrixXd msg(B_bar.size(), 1);

        for (int i = 0; i < B_bar.size(); ++i) {
            msg(i, 0) = (to_hat.transpose() * B_bar[i] * from_hat)(0, 0);
        }
        return {msg};
    }

    std::vector<MatrixXd> ActiveTransitionNode::bMessage() {
        MatrixXd to_hat     =     to->posterior()->params()[0];
        MatrixXd from_hat   =   from->posterior()->params()[0];
        MatrixXd action_hat = action->posterior()->params()[0];
        int actions = action_hat.rows();
        std::vector<MatrixXd> msg(actions);

        msg[actions - 1] = to_hat * from_hat.transpose();
        for (int i = 0; i < action_hat.size(); ++i) {
            if (msg[i].rows() == 0 && msg[i].cols() == 0) {
                msg[i] = MatrixXd::Zero(to_hat.rows(), from_hat.rows());
            }
            for (int j = 0; j < to_hat.size(); ++j) {
                for (int k = 0; k < from_hat.size(); ++k) {
                    msg[i](j,k) = action_hat(i,0) * msg[actions - 1](j,k);
                }
            }
        }
        return msg;
    }

    double ActiveTransitionNode::vfe() {
        auto to_p     = to->posterior()->params()[0];
        auto from_p   = from->posterior()->params()[0];
        auto action_p = action->posterior()->params()[0];
        auto lp       = getLogB();
        double VFE    = 0;

        if (child()->type() == HIDDEN) {
            VFE -= child()->posterior()->entropy();
        }
        for (int i = 0; i < lp.size(); ++i) {
            VFE -= action_p(i, 0) * (to_p.transpose() * lp[i] * from_p)(0, 0);
        }
        return VFE;
    }

    std::vector<MatrixXd> ActiveTransitionNode::getLogB() {
        if (B) {
            return Dirichlet::expectedLog(B->posterior()->params());
        } else {
            return to->prior()->logParams();
        }
    }

}
