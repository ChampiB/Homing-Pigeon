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

TEST_CASE( "CategoricalNode.vfe() returns the proper vfe contribution" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    MatrixXd param = MatrixXd::Constant(4, 1, 0.25);
    MatrixXd param2 = MatrixXd::Constant(2, 4, 0.5);
    auto fg = FactorGraph::current();
    auto c1 = Categorical::create(param);

    auto poc1 = c1->posterior()->params()[0];
    auto lpc1 = c1->posterior()->logParams()[0];
    auto prc1 = c1->prior()->logParams()[0];

    auto neg_entropy = (poc1.transpose() * lpc1)(0, 0);
    auto energy = (poc1.transpose() * prc1)(0, 0);
    REQUIRE( c1->parent()->vfe() == neg_entropy - energy);
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

TEST_CASE( "CategoricalNode::message throws a run_time error is thrown if the parameter is not the generated node" ) {
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

TEST_CASE( "CategoricalNode's (child) message is correct (no Dirichlet prior)" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    auto fg = FactorGraph::current();
    MatrixXd param1 = MatrixXd::Constant(4, 1, 0.25);
    auto c1 = Categorical::create(param1);
    auto m1 = c1->parent()->message(c1);

    REQUIRE( m1[0].cols() == 1 );
    REQUIRE( m1[0].rows() == 4 );
    REQUIRE( m1[0](0, 0) == std::log(0.25) );
    REQUIRE( m1[0](1, 0) == std::log(0.25) );
    REQUIRE( m1[0](2, 0) == std::log(0.25) );
    REQUIRE( m1[0](3, 0) == std::log(0.25) );

    FactorGraph::setCurrent(nullptr);
    MatrixXd param2(4, 1);
    param2 << 0.25,
            0.1,
            0.4,
            0.25;
    auto c2 = Categorical::create(param2);
    auto m2 = c2->parent()->message(c2);

    REQUIRE( m2[0].cols() == 1 );
    REQUIRE( m2[0].rows() == 4 );
    REQUIRE( m2[0](0, 0) == std::log(0.25) );
    REQUIRE( m2[0](1, 0) == std::log(0.1) );
    REQUIRE( m2[0](2, 0) == std::log(0.4) );
    REQUIRE( m2[0](3, 0) == std::log(0.25) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "CategoricalNode's (child) message is correct (Dirichlet prior)" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    auto fg = FactorGraph::current();
    MatrixXd param1 = MatrixXd::Constant(4, 1, 0.25);
    auto d1 = Dirichlet::create(param1);
    auto c1 = Categorical::create(d1);
    auto m1 = c1->parent()->message(c1);
    auto res = Dirichlet::expectedLog(d1->posterior()->params())[0];

    REQUIRE( m1[0].cols() == 1 );
    REQUIRE( m1[0].rows() == 4 );
    REQUIRE( m1[0](0, 0) == res(0, 0) );
    REQUIRE( m1[0](1, 0) == res(1, 0) );
    REQUIRE( m1[0](2, 0) == res(2, 0) );
    REQUIRE( m1[0](3, 0) == res(3, 0) );

    FactorGraph::setCurrent(nullptr);
    MatrixXd param2(4, 1);
    param2 << 0.25,
            0.1,
            0.4,
            0.25;
    auto d2 = Dirichlet::create(param2);
    auto c2 = Categorical::create(d2);
    auto m2 = c2->parent()->message(c2);
    res = Dirichlet::expectedLog(d2->posterior()->params())[0];

    REQUIRE( m2[0].cols() == 1 );
    REQUIRE( m2[0].rows() == 4 );
    REQUIRE( m2[0](0, 0) == res(0, 0) );
    REQUIRE( m2[0](1, 0) == res(1, 0) );
    REQUIRE( m2[0](2, 0) == res(2, 0) );
    REQUIRE( m2[0](3, 0) == res(3, 0) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "CategoricalNode's (parent) message is correct (Dirichlet prior)" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    FactorGraph::setCurrent(nullptr);
    auto fg = FactorGraph::current();
    MatrixXd param1 = MatrixXd::Constant(4, 1, 0.25);
    auto d1 = Dirichlet::create(param1);
    auto c1 = Categorical::create(d1);
    auto m1 = c1->parent()->message(d1);

    REQUIRE( m1[0].cols() == 1 );
    REQUIRE( m1[0].rows() == 4 );
    REQUIRE( m1[0](0, 0) == 0.25 );
    REQUIRE( m1[0](1, 0) == 0.25 );
    REQUIRE( m1[0](2, 0) == 0.25 );
    REQUIRE( m1[0](3, 0) == 0.25 );

    FactorGraph::setCurrent(nullptr);
    MatrixXd param2(4, 1);
    param2 << 0.25,
              0.1,
              0.4,
              0.25;
    MatrixXd param3(4, 1);
    param3 << 0.7,
              0.01,
              0.01,
              0.28;
    std::vector<MatrixXd> vec3 {param3};
    auto d2 = Dirichlet::create(param2);
    auto c2 = Categorical::create(d2);
    c2->posterior()->updateParams(vec3);
    auto m2 = c2->parent()->message(d2);
    auto res = Functions::softmax(param3);

    REQUIRE( m2[0].cols() == 1 );
    REQUIRE( m2[0].rows() == 4 );
    REQUIRE( m2[0](0, 0) == res(0, 0) );
    REQUIRE( m2[0](1, 0) == res(1, 0) );
    REQUIRE( m2[0](2, 0) == res(2, 0) );
    REQUIRE( m2[0](3, 0) == res(3, 0) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}
