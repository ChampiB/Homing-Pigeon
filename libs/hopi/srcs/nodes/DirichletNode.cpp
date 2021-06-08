//
// Created by Theophile Champion on 06/01/2021.
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

    VarNode *DirichletNode::parent(int i) {
        return nullptr;
    }

    VarNode *DirichletNode::child() {
        return childNode;
    }

    Tensor DirichletNode::message(VarNode *to) {
        if (to == childNode) {
            return childMessage();
        } else {
            assert(false && "DirichletNode::message, invalid input node.");
        }
    }

    double DirichletNode::energy(const Tensor &prior, const Tensor &post) {
        auto sum = post.sum().item<double>();
        double acc = 0;

        assert(prior.dim() == 1 && "DirichletNode::energy, prior must have dimension one.");
        assert(post.dim() == 1 && "DirichletNode::energy, post must have dimension one.");
        for (int k = 0; k < post.size(0); ++k) {
            acc += (prior[k].item<double>() - 1) *  (Ops::digamma(post[k].item<double>()) - Ops::digamma(sum));
        }
        return acc - Ops::log_beta(prior);
    }

    double DirichletNode::vfe() {
        double VFE = 0;
        Tensor post_p  = child()->posterior()->params();
        Tensor prior_p = child()->prior()->params();

        assert(prior_p.dim() == post_p.dim() && "DirichletNode::vfe, post and prior parameters must have the same dimension");
        assert(prior_p.sizes() == post_p.sizes() && "DirichletNode::vfe, post and prior parameters must have the same sizes");
        if (child()->type() == HIDDEN) {
            VFE -= child()->posterior()->entropy();
        }
        Ops::unsqueeze(3 - prior_p.dim(), {&prior_p,&post_p});
        for (int i = 0; i < post_p.size(0); ++i) {
            for (int j = 0; j < post_p.size(1); ++j) {
                VFE -= energy(prior_p.index({i,j,Ellipsis}), post_p.index({i,j,Ellipsis}));
            }
        }
        return VFE;
    }

    Tensor DirichletNode::childMessage() {
        return childNode->prior()->params();
    }

}
