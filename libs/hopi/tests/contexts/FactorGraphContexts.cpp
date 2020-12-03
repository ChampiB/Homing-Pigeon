//
// Created by tmac3 on 02/12/2020.
//

#include "FactorGraphContexts.h"
#include "distributions/Categorical.h"
#include "distributions/ActiveTransition.h"
#include "distributions/Transition.h"
#include "graphs/FactorGraph.h"
#include "nodes/VarNode.h"
#include <Eigen/Dense>

using namespace Eigen;
using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::distributions;

namespace tests {

    std::shared_ptr<FactorGraph> FactorGraphContexts::context1() {
        FactorGraph::setCurrent(nullptr);
        std::shared_ptr<FactorGraph> fg = FactorGraph::current();
        MatrixXd param = MatrixXd::Constant(3, 1, 1.0 / 3);

        VarNode *c1 = Categorical::create(param);
        c1->setType(VarNodeType::OBSERVED);
        Categorical::create(param); // First hidden variable
        VarNode *c3 = Categorical::create(param);
        c3->setType(VarNodeType::OBSERVED);
        VarNode *c4 = Categorical::create(param);
        c4->setType(VarNodeType::OBSERVED);
        Categorical::create(param); // Second hidden variable
        Categorical::create(param); // Third hidden variable
        VarNode *c6 = Categorical::create(param);
        c6->setType(VarNodeType::OBSERVED);
        VarNode *c7 = Categorical::create(param);
        c7->setType(VarNodeType::OBSERVED);
        VarNode *c8 = Categorical::create(param);
        c8->setType(VarNodeType::OBSERVED);
        return fg;
    }

    std::shared_ptr<hopi::graphs::FactorGraph> FactorGraphContexts::context2() {
        FactorGraph::setCurrent(nullptr);
        std::shared_ptr<FactorGraph> fg = FactorGraph::current();

        /**
         ** Create the model's parameters.
         **/
        MatrixXd U0 = MatrixXd::Constant(5, 1, 1.0 / 5);
        MatrixXd D0 = MatrixXd::Constant(3,  1, 1.0 / 3);

        int A_size = 9 * 3;
        MatrixXd A = MatrixXd::Constant(9, 3, 1.0 / A_size);

        int B_size = 3 * 3;
        MatrixXd B_idle  = MatrixXd::Constant(3, 3, 1.0 / B_size);
        MatrixXd B_up    = MatrixXd::Constant(3, 3, 1.0 / B_size);
        MatrixXd B_down  = MatrixXd::Constant(3, 3, 1.0 / B_size);
        MatrixXd B_right = MatrixXd::Constant(3, 3, 1.0 / B_size);
        MatrixXd B_left  = MatrixXd::Constant(3, 3, 1.0 / B_size);
        std::vector<MatrixXd> B = {B_up, B_down, B_left, B_right, B_idle};

        /**
         ** Create the generative model.
         **/
        VarNode *a0 = Categorical::create(U0);
        VarNode *s0 = Categorical::create(D0);
        VarNode *o0 = Transition::create(s0, A);
        o0->setType(VarNodeType::OBSERVED);
        VarNode *s1 = ActiveTransition::create(s0, a0, B);
        VarNode *o1 = Transition::create(s1, A);
        o1->setType(VarNodeType::OBSERVED);
        fg->setTreeRoot(s1);

        return fg;
    }

    std::shared_ptr<hopi::graphs::FactorGraph> FactorGraphContexts::context3() {
        FactorGraph::setCurrent(nullptr);
        std::shared_ptr<FactorGraph> fg = FactorGraph::current();
        MatrixXd param = MatrixXd::Constant(3, 1, 1.0 / 3);

        Categorical::create(param);
        VarNode *c1 = Categorical::create(param); // First observed variable
        c1->setType(VarNodeType::OBSERVED);
        Categorical::create(param);
        Categorical::create(param);
        VarNode *c3 = Categorical::create(param); // Second observed variable
        c3->setType(VarNodeType::OBSERVED);
        VarNode *c4 = Categorical::create(param); // Third observed variable
        c4->setType(VarNodeType::OBSERVED);
        Categorical::create(param);
        Categorical::create(param);
        Categorical::create(param);
        return fg;
    }

}
