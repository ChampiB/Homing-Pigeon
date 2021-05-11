//
// Created by tmac3 on 02/12/2020.
//

#include <iostream>
#include "catch.hpp"
#include "nodes/VarNode.h"
#include "nodes/CategoricalNode.h"
#include "distributions/Dirichlet.h"
#include "graphs/FactorGraph.h"
#include "math/Functions.h"
#include "helpers/UnitTests.h"
#include "api/API.h"
#include <Eigen/Dense>

using namespace hopi::nodes;
using namespace hopi::graphs;
using namespace hopi::distributions;
using namespace hopi::api;
using namespace hopi::math;
using namespace Eigen;
using namespace tests;

TEST_CASE( "DirichletNode returns the correct child and parents" ) {
    UnitTests::run([](){
        auto to     = VarNode::create(VarNodeType::HIDDEN);
        auto factor = CategoricalNode::create(to.get());

        REQUIRE( factor->child() == to.get() );
        REQUIRE( factor->parent(0) == nullptr );
    });
}

TEST_CASE( "DirichletNode::message throws a run_time error is thrown if the parameter is not the generated node" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        MatrixXd param = Functions::uniformColumnWise(4, 1);
        auto c1 = VarNode::create(VarNodeType::HIDDEN);
        auto fg = FactorGraph::current();
        auto c2 = API::Dirichlet(param);

        try {
            c2->parent()->message(c1.get());
            REQUIRE( false );
        } catch (const std::runtime_error& error) {
            // Correct
        }
    });
}

TEST_CASE( "DirichletNode's (child) message is correct" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        auto fg = FactorGraph::current();
        MatrixXd param1 = Functions::uniformColumnWise(4, 1);
        auto c1 = API::Dirichlet(param1);
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
        auto c2 = API::Dirichlet(param2);
        auto m2 = c2->parent()->message(c2);

        REQUIRE( m2[0].cols() == 1 );
        REQUIRE( m2[0].rows() == 4 );
        REQUIRE( m2[0](0, 0) == param2(0, 0) );
        REQUIRE( m2[0](1, 0) == param2(1, 0) );
        REQUIRE( m2[0](2, 0) == param2(2, 0) );
        REQUIRE( m2[0](3, 0) == param2(3, 0) );
    });
}

TEST_CASE( "DirichletNode.vfe() returns the proper vfe contribution" ) {
    UnitTests::run([](){
        FactorGraph::setCurrent(nullptr);
        MatrixXd param1 = Functions::uniformColumnWise(2, 4);
        auto c1 = API::Dirichlet(param1);
        REQUIRE( c1->parent()->vfe() == Approx(-7.8540401042) );

        FactorGraph::setCurrent(nullptr);
        MatrixXd param2(4, 1);
        param2 << 1,
                10,
                20,
                5;
        auto c2 = API::Dirichlet(param2);
        REQUIRE( c2->parent()->vfe() == Approx(-0.7301998275) );
    });
}
