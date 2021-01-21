//
// Created by tmac3 on 02/12/2020.
//

#include <iostream>
#include "catch.hpp"
#include "nodes/VarNode.h"
#include "nodes/TransitionNode.h"
#include "distributions/Distribution.h"
#include "distributions/Transition.h"
#include "distributions/Dirichlet.h"
#include "distributions/Categorical.h"
#include "math/Functions.h"
#include "graphs/FactorGraph.h"
#include <Eigen/Dense>

using namespace hopi::math;
using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::distributions;
using namespace Eigen;

TEST_CASE( "TransitionNode.vfe() returns the proper vfe contribution" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    MatrixXd param = MatrixXd::Constant(4, 1, 0.25);
    MatrixXd param2 = MatrixXd::Constant(2, 4, 0.5);
    auto fg = FactorGraph::current();
    auto c1 = Categorical::create(param);
    auto t1 = Transition::create(c1, param2);

    auto poc1 = c1->posterior()->params()[0];
    auto pot1 = t1->posterior()->params()[0];
    auto lpt1 = t1->posterior()->logParams()[0];
    auto prt1 = t1->prior()->logParams()[0];

    auto neg_entropy = (pot1.transpose() * lpt1)(0, 0);
    auto energy = (pot1.transpose() * prt1 * poc1)(0, 0);
    REQUIRE( t1->parent()->vfe() == neg_entropy - energy);
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "TransitionNode returns the correct child and parents" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto from   = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto to     = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto factor = std::make_unique<TransitionNode>(from.get(), to.get());

    REQUIRE( factor->child() == to.get() );
    REQUIRE( factor->parent(0) == from.get() );
    REQUIRE( factor->parent(1) == nullptr );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "TransitionNode: A run_time error is thrown if the parameter is an unknown node" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    MatrixXd param = MatrixXd::Constant(4, 1, 0.25);
    MatrixXd param2 = MatrixXd::Constant(2, 4, 0.5);
    auto t1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto fg = FactorGraph::current();
    auto c1 = Categorical::create(param);
    auto t2 = Transition::create(c1, param2);

    try {
        t2->parent()->message(t1.get());
        REQUIRE( false );
    } catch (const std::runtime_error& error) {
        // Correct
    }
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "TransitionNode's (child and parent) messages are correct (no Dirichlet prior)" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    MatrixXd evidence = MatrixXd::Constant(2, 1, 0.5);
    MatrixXd param1 = MatrixXd::Constant(4, 1, 0.25);
    MatrixXd param2 = MatrixXd::Constant(2, 4, 0.5);
    auto fg = FactorGraph::current();
    auto c1 = Categorical::create(param1);
    auto t1 = Transition::create(c1, param2);
    t1->setPosterior(std::make_unique<Categorical>(evidence));
    t1->setType(VarNodeType::OBSERVED);

    MatrixXd res1 = param2.array().log();
    res1 = res1.transpose() * evidence;
    auto m1 = t1->parent()->message(c1);
    REQUIRE( m1[0].cols() == 1 );
    REQUIRE( m1[0].rows() == 4 );
    REQUIRE( m1[0](0, 0) == res1(0, 0) );
    REQUIRE( m1[0](1, 0) == res1(1, 0) );
    REQUIRE( m1[0](2, 0) == res1(2, 0) );
    REQUIRE( m1[0](3, 0) == res1(3, 0) );

    MatrixXd res2 = param2.array().log();
    res2 = res2 * param1;
    auto m2 = t1->parent()->message(t1);
    REQUIRE( m2[0].cols() == 1 );
    REQUIRE( m2[0].rows() == 2 );
    REQUIRE( m2[0](0, 0) == res2(0, 0) );
    REQUIRE( m2[0](1, 0) == res2(1, 0) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "TransitionNode's (child, param and parent) messages are correct (Dirichlet prior)" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    MatrixXd evidence = MatrixXd::Constant(2, 1, 0.5);
    MatrixXd param1 = MatrixXd::Constant(4, 1, 0.25);
    MatrixXd param2 = MatrixXd::Constant(2, 4, 0.5);
    std::vector<MatrixXd> vec2 { param2 };
    auto fg = FactorGraph::current();
    auto c1 = Categorical::create(param1);
    auto d1 = Dirichlet::create(param2);
    auto t1 = Transition::create(c1, d1);
    t1->setPosterior(std::make_unique<Categorical>(evidence));
    t1->setType(VarNodeType::OBSERVED);

    MatrixXd res1 = Dirichlet::expectedLog(vec2)[0];
    res1 = res1.transpose() * evidence;
    auto m1 = t1->parent()->message(c1);
    REQUIRE( m1[0].cols() == 1 );
    REQUIRE( m1[0].rows() == 4 );
    REQUIRE( m1[0](0, 0) == res1(0, 0) );
    REQUIRE( m1[0](1, 0) == res1(1, 0) );
    REQUIRE( m1[0](2, 0) == res1(2, 0) );
    REQUIRE( m1[0](3, 0) == res1(3, 0) );

    MatrixXd res2 = Dirichlet::expectedLog(vec2)[0];
    res2 = res2 * param1;
    auto m2 = t1->parent()->message(t1);
    REQUIRE( m2[0].cols() == 1 );
    REQUIRE( m2[0].rows() == 2 );
    REQUIRE( m2[0](0, 0) == res2(0, 0) );
    REQUIRE( m2[0](1, 0) == res2(1, 0) );

    MatrixXd res3 = evidence * param1.transpose();
    auto m3 = t1->parent()->message(d1);
    REQUIRE( m3[0].cols() == 4 );
    REQUIRE( m3[0].rows() == 2 );
    REQUIRE( m3[0](0, 0) == res3(0, 0) );
    REQUIRE( m3[0](1, 0) == res3(1, 0) );
    REQUIRE( m3[0](0, 1) == res3(0, 1) );
    REQUIRE( m3[0](1, 1) == res3(1, 1) );
    REQUIRE( m3[0](0, 2) == res3(0, 2) );
    REQUIRE( m3[0](1, 2) == res3(1, 2) );
    REQUIRE( m3[0](0, 3) == res3(0, 3) );
    REQUIRE( m3[0](1, 3) == res3(1, 3) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}
