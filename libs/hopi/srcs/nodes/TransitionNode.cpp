//
// Created by tmac3 on 28/11/2020.
//

#include "TransitionNode.h"
#include <Eigen/Dense>
#include "nodes/VarNode.h"
#include "distributions/Distribution.h"

using namespace hopi::nodes;
using namespace hopi::distributions;
using namespace Eigen;

namespace hopi::nodes {

    TransitionNode::TransitionNode(VarNode *f, VarNode *t) {
        from = f;
        to = t;
    }

    VarNode *TransitionNode::parent(int index) {
        if (index == 0)
            return from;
        else
            return nullptr;
    }

    VarNode *TransitionNode::child() {
        return to;
    }

    MatrixXd TransitionNode::message(VarNode *t) {
        if (t == to) {
            return toMessage();
        } else if (t == from) {
            return fromMessage();
        } else {
            throw std::runtime_error("Unsupported: Message towards non-adjacent node.");
        }
    }

    Eigen::MatrixXd TransitionNode::toMessage() {
        MatrixXd A = to->prior()->logProbability()[0];
        MatrixXd hat = from->posterior()->probability()[0];
        return A * hat;
    }

    Eigen::MatrixXd TransitionNode::fromMessage() {
        MatrixXd A = to->prior()->logProbability()[0];
        MatrixXd hat = to->posterior()->probability()[0];
        return A.transpose() * hat;
    }

    double TransitionNode::vfe() {
        auto to_p   = to->posterior()->probability()[0];
        auto from_p = from->posterior()->probability()[0];
        auto lp     = to->prior()->logProbability()[0];
        return (to_p.transpose() * lp * from_p)(0, 0);
    }

}

