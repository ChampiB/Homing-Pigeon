//
// Created by tmac3 on 28/11/2020.
//

#include "TransitionNode.h"
#include <Eigen/Dense>
#include "nodes/VarNode.h"
#include "distributions/Dirichlet.h"

using namespace hopi::nodes;
using namespace hopi::distributions;
using namespace Eigen;

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

    std::vector<MatrixXd> TransitionNode::message(VarNode *t) {
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

    std::vector<MatrixXd> TransitionNode::toMessage() {
        MatrixXd A_bar = getLogA();
        MatrixXd hat = from->posterior()->params()[0];
        return {A_bar * hat};
    }

    std::vector<MatrixXd> TransitionNode::fromMessage() {
        MatrixXd A_bar = getLogA();
        MatrixXd hat = to->posterior()->params()[0];
        return {A_bar.transpose() * hat};
    }

    std::vector<MatrixXd> TransitionNode::aMessage() {
        MatrixXd o =   to->posterior()->params()[0];
        MatrixXd s = from->posterior()->params()[0];
        return {o * s.transpose()};
    }

    double TransitionNode::vfe() {
        double VFE = 0;

        if (child()->type() == HIDDEN) {
            VFE -= child()->posterior()->entropy();
        }
        auto to_p   = to->posterior()->params()[0];
        auto from_p = from->posterior()->params()[0];
        auto lp     = getLogA();
        return VFE - (to_p.transpose() * lp * from_p)(0, 0);
    }

    MatrixXd TransitionNode::getLogA() {
        if (A) {
            return Dirichlet::expectedLog(A->posterior()->params())[0];
        } else {
            return to->prior()->logParams()[0];
        }
    }

}

