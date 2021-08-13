//
// Created by Theophile Champion on 28/11/2020.
//

#include "CategoricalNode.h"
#include "VarNode.h"
#include "math/Ops.h"
#include "distributions/Dirichlet.h"

using namespace hopi::distributions;
using namespace hopi::math;
using namespace torch;

namespace hopi::nodes {

    std::unique_ptr<CategoricalNode> CategoricalNode::create(RV *to, RV *d) {
        return std::make_unique<CategoricalNode>(to, d);
    }

    std::unique_ptr<CategoricalNode> CategoricalNode::create(RV *node) {
        return std::make_unique<CategoricalNode>(node);
    }

    CategoricalNode::CategoricalNode(VarNode *to, VarNode *param) {
        childNode = to;
        D = param;
    }

    CategoricalNode::CategoricalNode(VarNode *to) : CategoricalNode(to, nullptr) {}

    VarNode *CategoricalNode::parent(int i) {
        if (i == 0)
            return D;
        else
            return nullptr;
    }

    VarNode *CategoricalNode::child() const {
        return childNode;
    }

    Tensor CategoricalNode::message(VarNode *t) {
        if (t == childNode) {
            return childMessage();
        } else if (D && t == D) {
            return dMessage();
        } else {
            assert(false && "CategoricalNode::message, invalid input node.");
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
        return VFE - dot(child()->posterior()->params(), getLogD()).item<double>();
    }

    Tensor CategoricalNode::getLogD() {
        if (D) {
            return Dirichlet::expectedLog(D->posterior()->params());
        } else {
            return child()->prior()->logParams();
        }
    }

}