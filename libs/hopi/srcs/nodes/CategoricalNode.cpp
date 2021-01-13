//
// Created by tmac3 on 28/11/2020.
//

#include "CategoricalNode.h"
#include "VarNode.h"
#include "math/Functions.h"
#include "distributions/Categorical.h"
#include "distributions/Dirichlet.h"

using namespace hopi::distributions;
using namespace hopi::math;
using namespace Eigen;

namespace hopi::nodes {

    CategoricalNode::CategoricalNode(VarNode *node, VarNode *d) {
        childNode = node;
        D = d;
    }

    CategoricalNode::CategoricalNode(VarNode *node) : CategoricalNode(node, nullptr) {}

    VarNode *CategoricalNode::parent(int index) {
        if (index == 0)
            return D;
        else
            return nullptr;
    }

    VarNode *CategoricalNode::child() {
        return childNode;
    }

    std::vector<MatrixXd> CategoricalNode::message(VarNode *t) {
        if (t == childNode) {
            return childMessage();
        } else if (D && t == D) {
            return dMessage();
        } else {
            throw std::runtime_error("Unsupported: Message towards non-adjacent node.");
        }
    }

    std::vector<MatrixXd> CategoricalNode::childMessage() {
        return {getLogD()};
    }

    std::vector<MatrixXd> CategoricalNode::dMessage() {
        return child()->posterior()->params();
    }

    double CategoricalNode::vfe() {
        double VFE = 0;

        if (child()->type() == HIDDEN) {
            VFE -= child()->posterior()->entropy();
        }
        auto p  = child()->posterior()->params()[0];
        auto lp = getLogD();
        return VFE - (p.transpose() * lp)(0, 0);
    }

    MatrixXd CategoricalNode::getLogD() {
        if (D) {
            return Dirichlet::expectedLog(D->posterior()->params())[0];
        } else {
            return child()->prior()->logParams()[0];
        }
    }

}