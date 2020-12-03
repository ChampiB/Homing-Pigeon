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

    Eigen::MatrixXd CategoricalNode::message(VarNode *t) {
        if (t == childNode) {
            return childMessage();
        } else {
            throw std::runtime_error("Unsupported: Message towards non-adjacent node.");
        }
    }

    Eigen::MatrixXd CategoricalNode::childMessage() {
        return child()->prior()->logProbability()[0];
    }

    double CategoricalNode::vfe() {
        auto p  = child()->posterior()->probability()[0];
        auto lp = child()->prior()->logProbability()[0];
        return (p.transpose() * lp)(0, 0);
    }

}