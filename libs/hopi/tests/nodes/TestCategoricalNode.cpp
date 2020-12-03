//
// Created by tmac3 on 02/12/2020.
//

#include <iostream>
#include "catch.hpp"
#include "nodes/VarNode.h"
#include "nodes/CategoricalNode.h"
#include "distributions/Categorical.h"
#include "graphs/FactorGraph.h"
#include "distributions/Distribution.h"
#include <Eigen/Dense>

using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::distributions;
using namespace Eigen;

TEST_CASE( "CategoricalNode.vfe() returns the proper vfe contribution" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    MatrixXd param = MatrixXd::Constant(4, 1, 0.25);
    MatrixXd param2 = MatrixXd::Constant(2, 4, 0.5);
    auto fg = FactorGraph::current();
    auto c1 = Categorical::create(param);

    auto poc1 = c1->posterior()->probability()[0];
    auto prt1 = c1->prior()->logProbability()[0];
    REQUIRE( c1->parent()->vfe() == (poc1.transpose() * prt1)(0, 0));
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "CategoricalNode.name getter and setter works properly" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto to     = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto factor = std::make_unique<CategoricalNode>(to.get());

    REQUIRE( factor->name().empty() );
    factor->setName("test");
    REQUIRE( factor->name() == "test" );
    factor->setName("abc");
    REQUIRE( factor->name() == "abc" );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "CategoricalNode returns the correct child and parents" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto to     = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto factor = std::make_unique<CategoricalNode>(to.get());

    REQUIRE( factor->child() == to.get() );
    REQUIRE( factor->parent(0) == nullptr );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "CategoricalNode: A run_time error is thrown if the parameter is not the generated node" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    MatrixXd param = MatrixXd::Constant(4, 1, 0.25);
    auto c1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto fg = FactorGraph::current();
    auto c2 = Categorical::create(param);

    try {
        c2->parent()->message(c1.get());
        REQUIRE( false );
    } catch (const std::runtime_error& error) {
        // Correct
    }
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "CategoricalNode's message is correct" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    auto fg = FactorGraph::current();
    MatrixXd param1 = MatrixXd::Constant(4, 1, 0.25);
    auto c1 = Categorical::create(param1);
    auto m1 = c1->parent()->message(c1);

    REQUIRE( m1.cols() == 1 );
    REQUIRE( m1.rows() == 4 );
    REQUIRE( m1(0, 0) == std::log(0.25) );
    REQUIRE( m1(1, 0) == std::log(0.25) );
    REQUIRE( m1(2, 0) == std::log(0.25) );
    REQUIRE( m1(3, 0) == std::log(0.25) );

    FactorGraph::setCurrent(nullptr);
    fg = FactorGraph::current();
    MatrixXd param2 = MatrixXd::Constant(4, 1, 0.25);
    param2(1, 0) = 0.1;
    param2(2, 0) = 0.4;
    auto c2 = Categorical::create(param2);
    auto m2 = c2->parent()->message(c2);

    REQUIRE( m2.cols() == 1 );
    REQUIRE( m2.rows() == 4 );
    REQUIRE( m2(0, 0) == std::log(0.25) );
    REQUIRE( m2(1, 0) == std::log(0.1) );
    REQUIRE( m2(2, 0) == std::log(0.4) );
    REQUIRE( m2(3, 0) == std::log(0.25) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}
