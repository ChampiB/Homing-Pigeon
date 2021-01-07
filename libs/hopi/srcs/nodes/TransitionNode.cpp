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

    std::vector<Eigen::MatrixXd> TransitionNode::message(VarNode *t) {
        if (t == to) {
            return toMessage();
        } else if (t == from) {
            return fromMessage();
        } else {
            throw std::runtime_error("Unsupported: Message towards non-adjacent node.");
        }
    }

    std::vector<Eigen::MatrixXd> TransitionNode::toMessage() {
        MatrixXd A = to->prior()->logParams()[0];
        MatrixXd hat = from->posterior()->params()[0];
        std::vector<Eigen::MatrixXd> msg{A * hat};
        return msg;
    }

    std::vector<Eigen::MatrixXd> TransitionNode::fromMessage() {
        MatrixXd A = to->prior()->logParams()[0];
        MatrixXd hat = to->posterior()->params()[0];
        std::vector<Eigen::MatrixXd> msg{A.transpose() * hat};
        return msg;
    }

    double TransitionNode::vfe() {
        double VFE = 0;

        if (child()->type() == HIDDEN) {
            VFE -= child()->posterior()->entropy();
        }
        auto to_p   = to->posterior()->params()[0];
        auto from_p = from->posterior()->params()[0];
        auto lp     = to->prior()->logParams()[0];
        return VFE - (to_p.transpose() * lp * from_p)(0, 0);
    }

}

