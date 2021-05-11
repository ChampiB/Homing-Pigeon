//
// Created by tmac3 on 02/12/2020.
//

#include "catch.hpp"
#include <Eigen/Dense>
#include <iostream>
#include "distributions/ActiveTransition.h"
#include "math/Functions.h"
#include "helpers/UnitTests.h"

using namespace Eigen;
using namespace hopi::distributions;
using namespace hopi::math;
using namespace tests;

TEST_CASE( "ActiveTransition distribution returns the proper type" ) {
    UnitTests::run([](){
        MatrixXd param(3, 1);
        std::vector<MatrixXd> p{param};
        ActiveTransition t = ActiveTransition(p);
        REQUIRE( t.type() == DistributionType::ACTIVE_TRANSITION );
    });
}

TEST_CASE( "ActiveTransition distribution returns the correct log params" ) {
    UnitTests::run([](){
        MatrixXd param1(3, 2);
        param1 << 0.1, 0.3,
                0.7, 0.4,
                0.2, 0.3;
        std::vector<MatrixXd> p1{ param1 };
        ActiveTransition d1 = ActiveTransition(p1);
        auto lp1 = d1.logParams();
        REQUIRE( lp1.size() == 1 );
        REQUIRE( lp1[0](0, 0) == std::log(0.1) );
        REQUIRE( lp1[0](1, 0) == std::log(0.7) );
        REQUIRE( lp1[0](2, 0) == std::log(0.2) );
        REQUIRE( lp1[0](0, 1) == std::log(0.3) );
        REQUIRE( lp1[0](1, 1) == std::log(0.4) );
        REQUIRE( lp1[0](2, 1) == std::log(0.3) );

        MatrixXd param2_1(2, 2);
        param2_1 << 0.5, 0.2,
                0.5, 0.8;
        MatrixXd param2_2(2, 2);
        param2_2 << 0.3, 0.05,
                0.7, 0.95;
        std::vector<MatrixXd> p2{ param2_1, param2_2 };
        ActiveTransition d2 = ActiveTransition(p2);
        auto lp2 = d2.logParams();
        REQUIRE( lp2.size() == 2 );
        REQUIRE( lp2[0](0,0) == std::log(0.5) );
        REQUIRE( lp2[0](1,0) == std::log(0.5) );
        REQUIRE( lp2[0](0,1) == std::log(0.2) );
        REQUIRE( lp2[0](1,1) == std::log(0.8) );
        REQUIRE( lp2[1](0,0) == std::log(0.3) );
        REQUIRE( lp2[1](1,0) == std::log(0.7) );
        REQUIRE( lp2[1](0,1) == std::log(0.05) );
        REQUIRE( lp2[1](1,1) == std::log(0.95) );
    });
}

TEST_CASE( "ActiveTransition parameters update and getter work properly" ) {
    UnitTests::run([](){
        // Create parameters
        MatrixXd param0(2, 2);
        param0 << 0.5, 0.2, 0.5, 0.8;
        MatrixXd param1(2, 2);
        param1 << 0.3, 0.05, 0.7, 0.95;
        MatrixXd param2(2, 2);
        param2 << 0.15, 0.04, 0.85, 0.96;

        // Test retrieving parameters
        std::vector<MatrixXd> p { param0, param1 };
        ActiveTransition d = ActiveTransition(p);
        auto output = d.params();
        REQUIRE(output.size() == 2 );
        REQUIRE(output[0](0, 0) == 0.5 );
        REQUIRE(output[0](1, 0) == 0.5 );
        REQUIRE(output[0](0, 1) == 0.2 );
        REQUIRE(output[0](1, 1) == 0.8 );
        REQUIRE(output[1](0, 0) == 0.3 );
        REQUIRE(output[1](1, 0) == 0.7 );
        REQUIRE(output[1](0, 1) == 0.05 );
        REQUIRE(output[1](1, 1) == 0.95 );

        // Test retrieving parameters after update
        std::vector<MatrixXd> p2 { param0, param2 };
        d.updateParams(p2);
        output = d.params();
        auto res0 = Functions::softmax(param0);
        auto res1 = Functions::softmax(param2);
        REQUIRE(output.size() == 2 );
        REQUIRE(output[0](0, 0) == res0(0, 0) );
        REQUIRE(output[0](1, 0) == res0(1, 0) );
        REQUIRE(output[0](0, 1) == res0(0, 1) );
        REQUIRE(output[0](1, 1) == res0(1, 1) );
        REQUIRE(output[1](0, 0) == res1(0, 0) );
        REQUIRE(output[1](1, 0) == res1(1, 0) );
        REQUIRE(output[1](0, 1) == res1(0, 1) );
        REQUIRE(output[1](1, 1) == res1(1, 1) );
    });
}

TEST_CASE( "ActiveTransition::updateParams() throws an exception if the sizes of the new and old parameters does not match" ) {
    UnitTests::run([](){
        std::vector<MatrixXd> p1 {
            Functions::uniformColumnWise(3, 3),
            Functions::uniformColumnWise(2, 3)
        };
        std::vector<MatrixXd> p2{
            Functions::uniformColumnWise(3, 3)
        };
        ActiveTransition d = ActiveTransition(p1);

        try {
            d.updateParams(p2);
            REQUIRE(false);
        } catch (std::exception &e) {
            REQUIRE(true);
        }
    });
}

TEST_CASE( "ActiveTransition::entropy() throws an exception (unsupported feature)" ) {
    UnitTests::run([](){
        std::vector<MatrixXd> p {
            Functions::uniformColumnWise(3,3),
            Functions::uniformColumnWise(3,3)
        };
        ActiveTransition d = ActiveTransition(p);

        try {
            d.entropy();
            REQUIRE(false);
        } catch (std::exception &e) {
            REQUIRE(true);
        }
    });
}
