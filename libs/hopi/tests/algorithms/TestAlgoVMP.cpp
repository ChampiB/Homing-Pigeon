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
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace hopi::algorithms;
using namespace hopi::distributions;
using namespace hopi::graphs;
using namespace hopi::nodes;
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
    res -= a0->parent()->vfe();
    res -= s0->parent()->vfe();
    res -= s1->parent()->vfe();
    res += (a0->posterior()->params()[0].transpose() * a0->posterior()->logParams()[0])(0, 0);
    res += (s0->posterior()->params()[0].transpose() * s0->posterior()->logParams()[0])(0, 0);
    REQUIRE( F == Approx(res).epsilon(0.1) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Softmax's output sum up to one" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd m1 = MatrixXd::Constant(4, 1, 42);
    m1 = AlgoVMP::softmax(m1);
    REQUIRE( m1.sum() == Approx(1).epsilon(0.1) );

    MatrixXd m2(3, 1);
    m2(0, 0) = 1;
    m2(1, 0) = 11;
    m2(2, 0) = 111;
    m2 = AlgoVMP::softmax(m2);
    REQUIRE( m2.sum() == Approx(1).epsilon(0.1) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Softmax does not overflow" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd m1(2, 1);
    m1(0, 0) = 1000000000000000000;
    m1(1, 0) = 500;
    m1 = AlgoVMP::softmax(m1);
    REQUIRE( m1.sum() == Approx(1).epsilon(0.1) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Softmax's output are correct" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd m1(1, 1);
    m1(0, 0) = 3;
    m1 = AlgoVMP::softmax(m1);
    REQUIRE( m1(0, 0) == 1 );

    MatrixXd m2(3, 1);
    m2(0, 0) = 3;
    m2(1, 0) = 4;
    m2(2, 0) = 1;
    m2 = AlgoVMP::softmax(m2);
    REQUIRE( m2(0, 0) == Approx(0.25949646034242).epsilon(0.1) );
    REQUIRE( m2(1, 0) == Approx(0.70538451269824).epsilon(0.1) );
    REQUIRE( m2(2, 0) == Approx(0.03511902695934).epsilon(0.1) );

    MatrixXd m3(2, 1);
    m3(0, 0) = 500;
    m3(1, 0) = 500;
    m3 = AlgoVMP::softmax(m3);
    REQUIRE( m3(0, 0) == 0.5 );
    REQUIRE( m3(1, 0) == 0.5 );

    MatrixXd m4(5, 1);
    m4(0, 0) = 3;
    m4(1, 0) = 4;
    m4(2, 0) = 1;
    m4(3, 0) = 10;
    m4(4, 0) = 2;
    m4 = AlgoVMP::softmax(m4);
    REQUIRE( m4(0, 0) == Approx(9.0838513102074E-4).epsilon(0.1) );
    REQUIRE( m4(1, 0) == Approx(0.0024692467948961).epsilon(0.1) );
    REQUIRE( m4(2, 0) == Approx(1.2293655899462E-4).epsilon(0.1) );
    REQUIRE( m4(3, 0) == Approx(0.99616525530072).epsilon(0.1) );
    REQUIRE( m4(4, 0) == Approx(3.3417621436836E-4).epsilon(0.1) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}
