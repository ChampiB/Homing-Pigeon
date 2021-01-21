//
// Created by tmac3 on 02/12/2020.
//

#include "catch.hpp"
#include "algorithms/AlgoVMP.h"
#include "graphs/FactorGraph.h"
#include "contexts/FactorGraphContexts.h"
#include "nodes/VarNode.h"
#include "nodes/FactorNode.h"
#include "distributions/Categorical.h"
#include "distributions/Transition.h"
#include "distributions/ActiveTransition.h"
#include "math/Functions.h"
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace hopi::algorithms;
using namespace hopi::distributions;
using namespace hopi::graphs;
using namespace hopi::nodes;
using namespace hopi::math;
using namespace tests;

TEST_CASE( "Inference process stop when vfe has converged (according to epsilon)" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    auto vars = fg->getNodes();

    AlgoVMP::inference(vars, 0.1);
    double beforeF = AlgoVMP::vfe(vars);
    AlgoVMP::inference(vars[0]);
    double afterF = AlgoVMP::vfe(vars);
    REQUIRE( afterF - beforeF < 0.1 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "During inference the vfe always decreases" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto fg = FactorGraphContexts::context2();
    auto vars = fg->getNodes();

    for (int i = 0; i < vars.size(); ++i) { // Always decreases during the first iteration of updates
        if (vars[i]->type() == HIDDEN) {
            double beforeF = AlgoVMP::vfe(vars);
            AlgoVMP::inference(vars[i]);
            double afterF = AlgoVMP::vfe(vars);
            REQUIRE( beforeF >= afterF );
        }
    }
    for (int i = 0; i < vars.size(); ++i) { // Always decreases during the second iteration of updates
        if (vars[i]->type() == HIDDEN) {
            double beforeF = AlgoVMP::vfe(vars);
            AlgoVMP::inference(vars[i]);
            double afterF = AlgoVMP::vfe(vars);
            REQUIRE( beforeF >= afterF );
        }
    }
    for (int i = 0; i < vars.size(); ++i) { // Always decreases during the third iteration of updates
        if (vars[i]->type() == HIDDEN) {
            double beforeF = AlgoVMP::vfe(vars);
            AlgoVMP::inference(vars[i]);
            double afterF = AlgoVMP::vfe(vars);
            REQUIRE( beforeF >= afterF );
        }
    }
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "AlgoVMP.vfe() returns the variational free energy of the variables sent as parameters" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    auto fg = FactorGraph::current();
    MatrixXd U = MatrixXd::Constant(2, 1, 0.5);
    VarNode *a0 = Categorical::create(U);
    MatrixXd D = MatrixXd::Constant(5, 1, 0.2);
    VarNode *s0 = Categorical::create(D);
    std::vector<MatrixXd> B {
            MatrixXd::Constant(5, 5, 0.2),
            MatrixXd::Constant(5, 5, 0.2)
    };
    VarNode *s1 = ActiveTransition::create(s0, a0, B);
    s1->setType(OBSERVED);

    double F = AlgoVMP::vfe(fg->getNodes());
    double res = 0;
    res += a0->parent()->vfe();
    res += s0->parent()->vfe();
    res += s1->parent()->vfe();
    REQUIRE( F == Approx(res).epsilon(0.1) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}
