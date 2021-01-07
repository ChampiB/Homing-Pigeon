//
// Created by tmac3 on 28/11/2020.
//

#include "CategoricalNode.h"
#include "VarNode.h"
#include "distributions/Categorical.h"
#include "distributions/Distribution.h"

using namespace hopi::distributions;
using namespace Eigen;

namespace hopi::nodes {

    CategoricalNode::CategoricalNode(VarNode *node) {
        childNode = node;
    }

    VarNode *CategoricalNode::parent(int index) {
        return nullptr;
    }

    VarNode *CategoricalNode::child() {
        return childNode;
    }

    std::vector<Eigen::MatrixXd> CategoricalNode::message(VarNode *t) {
        if (t == childNode) {
            return childMessage();
        } else {
            throw std::runtime_error("Unsupported: Message towards non-adjacent node.");
        }
    }

    std::vector<Eigen::MatrixXd> CategoricalNode::childMessage() {
        std::vector<Eigen::MatrixXd> msg{child()->prior()->logParams()[0]};
        return msg;
    }

    double CategoricalNode::vfe() {
        double VFE = 0;

        if (child()->type() == HIDDEN) {
            VFE -= child()->posterior()->entropy();
        }
        auto p  = child()->posterior()->params()[0];
        auto lp = child()->prior()->logParams()[0];
        return VFE - (p.transpose() * lp)(0, 0);
    }

}