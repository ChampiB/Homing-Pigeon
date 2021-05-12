//
// Created by tmac3 on 06/01/2021.
//

#include "DirichletNode.h"
#include "VarNode.h"
#include "distributions/Distribution.h"
#include "math/Ops.h"

using namespace torch;
using namespace torch::indexing;
using namespace hopi::math;

namespace hopi::nodes {

    std::unique_ptr<DirichletNode> DirichletNode::create(RV *node) {
        return std::make_unique<DirichletNode>(node);
    }

    DirichletNode::DirichletNode(VarNode *node) {
        childNode = node;
    }

    VarNode *DirichletNode::parent(int index) {
        return nullptr;
    }

    VarNode *DirichletNode::child() {
        return childNode;
    }

    Tensor DirichletNode::message(VarNode *to) {
        if (to == childNode) {
            return childMessage();
        } else {
            throw std::runtime_error("Unsupported: Message towards non-adjacent node.");
        }
    }

    double DirichletNode::energy(const Tensor &prior, const Tensor &post) {
        auto s = post.sum().item<double>();
        double acc = 0;

        for (int k = 0; k < post.size(0); ++k) {
            acc += (prior[k].item<double>() - 1) *  (Ops::digamma(post[k].item<double>()) - Ops::digamma(s));
        }
        return acc - std::log(Ops::beta(prior));
    }


    double DirichletNode::vfe() {
        double VFE = 0;
        Tensor post_p  = child()->posterior()->params();
        Tensor prior_p = child()->prior()->params();

        if (child()->type() == HIDDEN) {
            VFE -= child()->posterior()->entropy();
        }
        for (int i = 0; i < post_p.size(0); ++i) {
            for (int j = 0; j < post_p[i].size(1); ++j) {
                VFE -= energy(prior_p.index({i,None,j}), post_p.index({i,None,j}));
            }
        }
        return VFE;
    }

    Tensor DirichletNode::childMessage() {
        return childNode->prior()->params();
    }

}
