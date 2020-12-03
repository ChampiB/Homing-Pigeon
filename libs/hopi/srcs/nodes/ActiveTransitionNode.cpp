//
// Created by tmac3 on 28/11/2020.
//

#include "ActiveTransitionNode.h"
#include "distributions/Distribution.h"
#include <Eigen/Dense>
#include "VarNode.h"

using namespace Eigen;

namespace hopi::nodes {

    ActiveTransitionNode::ActiveTransitionNode(VarNode *f, VarNode *a, VarNode *t) {
        from = f;
        action = a;
        to = t;
    }

    VarNode *ActiveTransitionNode::parent(int index) {
        switch (index) {
            case 0:
                return from;
            case 1:
                return action;
            default:
                return nullptr;
        }
    }

    VarNode *ActiveTransitionNode::child() {
        return to;
    }

    MatrixXd ActiveTransitionNode::message(VarNode *t) {
        if (t == to) {
            return toMessage();
        } else if (t == from) {
            return fromMessage();
        } else if (t == action) {
            return actionMessage();
        } else {
            throw std::runtime_error("Unsupported: Message towards non-adjacent node.");
        }
    }

    Eigen::MatrixXd ActiveTransitionNode::toMessage() {
        std::vector<MatrixXd> B = to->prior()->logProbability();
        MatrixXd from_hat = from->posterior()->probability()[0];
        MatrixXd action_hat = action->posterior()->probability()[0];

        for (int i = 0; i < B.size(); ++i) {
            B[i] = action_hat(i, 0) * B[i] * from_hat;
            if (i != 0) {
                B[0] += B[i];
            }
        }
        return B[0];
    }

    Eigen::MatrixXd ActiveTransitionNode::fromMessage() {
        std::vector<MatrixXd> B = to->prior()->logProbability();
        MatrixXd to_hat = to->posterior()->probability()[0];
        MatrixXd action_hat = action->posterior()->probability()[0];

        for (int i = 0; i < B.size(); ++i) {
            B[i] = action_hat(i, 0) * B[i].transpose() * to_hat;
            if (i != 0) {
                B[0] += B[i];
            }
        }
        return B[0];
    }

    Eigen::MatrixXd ActiveTransitionNode::actionMessage() {
        std::vector<MatrixXd> B = to->prior()->logProbability();
        MatrixXd to_hat = to->posterior()->probability()[0];
        MatrixXd from_hat = from->posterior()->probability()[0];
        MatrixXd msg(B.size(), 1);

        for (int i = 0; i < B.size(); ++i) {
            msg(i, 0) = (to_hat.transpose() * B[i] * from_hat)(0, 0);
        }
        return msg;
    }

    double ActiveTransitionNode::vfe() {
        auto to_p     = to->posterior()->probability()[0];
        auto from_p   = from->posterior()->probability()[0];
        auto action_p = action->posterior()->probability()[0];
        auto lp       = to->prior()->logProbability();
        double VFE    = 0;

        for (int i = 0; i < lp.size(); ++i) {
            VFE += action_p(i, 0) * (to_p.transpose() * lp[i] * from_p)(0, 0);
        }
        return VFE;
    }

}
