//
// Created by tmac3 on 02/12/2020.
//

#include <iostream>
#include "catch.hpp"
#include "nodes/VarNode.h"
#include "math/Functions.h"
#include "nodes/CategoricalNode.h"
#include "distributions/Categorical.h"
#include "distributions/Dirichlet.h"
#include "graphs/FactorGraph.h"
#include <Eigen/Dense>

using namespace hopi::math;
using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::distributions;
using namespace Eigen;

TEST_CASE( "DirichletNode returns the correct child and parents" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    auto to     = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto factor = std::make_unique<CategoricalNode>(to.get());

    REQUIRE( factor->child() == to.get() );
    REQUIRE( factor->parent(0) == nullptr );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "DirichletNode::message throws a run_time error is thrown if the parameter is not the generated node" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    MatrixXd param = MatrixXd::Constant(4, 1, 0.25);
    auto c1 = std::make_unique<VarNode>(VarNodeType::HIDDEN);
    auto fg = FactorGraph::current();
    auto c2 = Dirichlet::create(param);

    try {
        c2->parent()->message(c1.get());
        REQUIRE( false );
    } catch (const std::runtime_error& error) {
        // Correct
    }
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}
TEST_CASE( "DirichletNode's (child) message is correct" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    auto fg = FactorGraph::current();
    MatrixXd param1 = MatrixXd::Constant(4, 1, 0.25);
    auto c1 = Dirichlet::create(param1);
    auto m1 = c1->parent()->message(c1);

    REQUIRE( m1[0].cols() == 1 );
    REQUIRE( m1[0].rows() == 4 );
    REQUIRE( m1[0](0, 0) == param1(0, 0) );
    REQUIRE( m1[0](1, 0) == param1(1, 0) );
    REQUIRE( m1[0](2, 0) == param1(2, 0) );
    REQUIRE( m1[0](3, 0) == param1(3, 0) );

    FactorGraph::setCurrent(nullptr);
    MatrixXd param2(4, 1);
    param2 << 0.25,
            0.1,
            0.4,
            0.25;
    auto c2 = Dirichlet::create(param2);
    auto m2 = c2->parent()->message(c2);

    REQUIRE( m2[0].cols() == 1 );
    REQUIRE( m2[0].rows() == 4 );
    REQUIRE( m2[0](0, 0) == param2(0, 0) );
    REQUIRE( m2[0](1, 0) == param2(1, 0) );
    REQUIRE( m2[0](2, 0) == param2(2, 0) );
    REQUIRE( m2[0](3, 0) == param2(3, 0) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "DirichletNode.vfe() returns the proper vfe contribution" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    MatrixXd param1 = MatrixXd::Constant(2, 4, 0.5);
    auto c1 = Dirichlet::create(param1);
    REQUIRE( c1->parent()->vfe() == Approx(-7.8540401042) );

    FactorGraph::setCurrent(nullptr);
    MatrixXd param2(4, 1);
    param2 << 1,
              10,
              20,
              5;
    auto c2 = Dirichlet::create(param2);
    REQUIRE( c2->parent()->vfe() == Approx(-0.7301998275) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}
