//
// Created by tmac3 on 02/12/2020.
//

#include <iostream>
#include "catch.hpp"
#include "nodes/VarNode.h"
#include "nodes/TransitionNode.h"
#include "distributions/Distribution.h"
#include "distributions/Transition.h"
#include "distributions/Categorical.h"
#include "graphs/FactorGraph.h"
#include <Eigen/Dense>

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

    auto poc1 = c1->posterior()->probability()[0];
    auto pot1 = t1->posterior()->probability()[0];
    auto prt1 = t1->prior()->logProbability()[0];
    REQUIRE( t1->parent()->vfe() == (pot1.transpose() * prt1 * poc1)(0, 0));    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
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

TEST_CASE( "TransitionNode's messages are correct" ) {
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
    REQUIRE( m1.cols() == 1 );
    REQUIRE( m1.rows() == 4 );
    REQUIRE( m1(0, 0) == res1(0, 0) );
    REQUIRE( m1(1, 0) == res1(1, 0) );
    REQUIRE( m1(2, 0) == res1(2, 0) );
    REQUIRE( m1(3, 0) == res1(3, 0) );

    MatrixXd res2 = param2.array().log();
    res2 = res2 * param1;
    auto m2 = t1->parent()->message(t1);
    REQUIRE( m2.cols() == 1 );
    REQUIRE( m2.rows() == 2 );
    REQUIRE( m2(0, 0) == res1(0, 0) );
    REQUIRE( m2(1, 0) == res1(1, 0) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}
