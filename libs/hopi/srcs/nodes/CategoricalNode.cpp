//
// Created by tmac3 on 28/11/2020.
//

#include "CategoricalNode.h"
#include "VarNode.h"
#include "math/Ops.h"
#include "distributions/Dirichlet.h"

using namespace hopi::distributions;
using namespace hopi::math;
using namespace torch;

namespace hopi::nodes {

    std::unique_ptr<CategoricalNode> CategoricalNode::create(RV *node, RV *d) {
        return std::make_unique<CategoricalNode>(node, d);
    }

    std::unique_ptr<CategoricalNode> CategoricalNode::create(RV *node) {
        return std::make_unique<CategoricalNode>(node);
    }

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

    Tensor CategoricalNode::message(VarNode *t) {
        if (t == childNode) {
            return childMessage();
        } else if (D && t == D) {
            return dMessage();
        } else {
            throw std::runtime_error("Unsupported: Message towards non-adjacent node.");
        }
    }

    Tensor CategoricalNode::childMessage() {
        return getLogD();
    }

    Tensor CategoricalNode::dMessage() {
        return child()->posterior()->params();
    }

    double CategoricalNode::vfe() {
        double VFE = 0;

        if (child()->type() == HIDDEN) {
            VFE -= child()->posterior()->entropy();
        }
        auto p  = child()->posterior()->params()[0];
        auto lp = getLogD();
        return VFE - matmul(p.permute({1,0}), lp).item<double>();
    }

    Tensor CategoricalNode::getLogD() {
        if (D) {
            return Dirichlet::expectedLog(D->posterior()->params())[0];
        } else {
            return child()->prior()->logParams()[0];
        }
    }

}