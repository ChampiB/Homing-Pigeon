//
// Created by tmac3 on 06/01/2021.
//

#include "DirichletNode.h"
#include "VarNode.h"
#include "distributions/Distribution.h"
#include "math/Functions.h"

using namespace Eigen;
using namespace hopi::math;

namespace hopi::nodes {

    DirichletNode::DirichletNode(VarNode *node) {
        childNode = node;
    }

    VarNode *DirichletNode::parent(int index) {
        return nullptr;
    }

    VarNode *DirichletNode::child() {
        return childNode;
    }

    std::vector<Eigen::MatrixXd> DirichletNode::message(VarNode *to) {
        if (to == childNode) {
            return childMessage();
        } else {
            throw std::runtime_error("Unsupported: Message towards non-adjacent node.");
        }
    }

    double DirichletNode::energy(MatrixXd prior, MatrixXd post) {
        double s = post.sum();
        double acc = 0;

        for (int k = 0; k < post.rows(); ++k) {
            acc += (prior(k) - 1) *  (Functions::digamma(post(k)) - Functions::digamma(s));
        }
        return acc - std::log(Functions::beta(prior));
    }


    double DirichletNode::vfe() {
        double VFE = 0;
        std::vector<MatrixXd> post_p = child()->posterior()->params();
        std::vector<MatrixXd> prior_p = child()->prior()->params();

        if (child()->type() == HIDDEN) {
            VFE -= child()->posterior()->entropy();
        }
        for (int i = 0; i < post_p.size(); ++i) {
            for (int j = 0; j < post_p[i].cols(); ++j) {
                VFE -= energy(prior_p[i].col(j), post_p[i].col(j));
            }
        }
        return VFE;
    }

    std::vector<Eigen::MatrixXd> DirichletNode::childMessage() {
        return childNode->prior()->params();
    }

}
