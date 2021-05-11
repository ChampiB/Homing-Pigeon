//
// Created by tmac3 on 02/12/2020.
//

#include "catch.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <helpers/UnitTests.h>
#include "distributions/Categorical.h"
#include "math/Functions.h"

using namespace Eigen;
using namespace hopi::distributions;
using namespace hopi::math;
using namespace tests;

TEST_CASE( "Categorical distribution returns correct cardinality" ) {
    UnitTests::run([](){
        MatrixXd param1(3, 1);
        Categorical c1 = Categorical(param1);
        REQUIRE( c1.cardinality() == 3 );

        MatrixXd param2(5, 1);
        Categorical c2 = Categorical(param2);
        REQUIRE( c2.cardinality() == 5 );

        MatrixXd param3(10, 1);
        Categorical c3 = Categorical(param3);
        REQUIRE( c3.cardinality() == 10 );
    });
}

TEST_CASE( "Categorical distribution returns the proper type" ) {
    UnitTests::run([](){
        MatrixXd param(3, 1);
        Categorical c = Categorical(param);
        REQUIRE( c.type() == DistributionType::CATEGORICAL );
    });
}

TEST_CASE( "Categorical distribution returns the correct params" ) {
    UnitTests::run([](){
        MatrixXd param1(3, 1);
        param1 << 0.1,
                0.7,
                0.2;
        Categorical c1 = Categorical(param1);
        REQUIRE( c1.p(0) == 0.1 );
        REQUIRE( c1.p(1) == 0.7 );
        REQUIRE( c1.p(2) == 0.2 );

        MatrixXd param2 = Functions::uniformColumnWise(2, 1);
        Categorical c2 = Categorical(param2);
        REQUIRE( c2.p(0) == 0.5 );
        REQUIRE( c2.p(1) == 0.5 );
    });
}

TEST_CASE( "Categorical distribution returns the correct log params" ) {
    UnitTests::run([](){
        MatrixXd param1(3, 1);
        param1 << 0.1,
                0.7,
                0.2;
        Categorical c1 = Categorical(param1);
        auto lp1 = c1.logParams();
        REQUIRE( lp1.size() == 1 );
        REQUIRE( lp1[0](0, 0) == std::log(0.1) );
        REQUIRE( lp1[0](1, 0) == std::log(0.7) );
        REQUIRE( lp1[0](2, 0) == std::log(0.2) );

        MatrixXd param2 = Functions::uniformColumnWise(2, 1);
        Categorical c2 = Categorical(param2);
        auto lp2 = c2.logParams();
        REQUIRE( lp2.size() == 1 );
        REQUIRE( lp2[0](0, 0) == std::log(0.5) );
        REQUIRE( lp2[0](1, 0) == std::log(0.5) );
    });
}

TEST_CASE( "Categorical::entropy() of [0 0 0 1 0] is zero" ) {
    UnitTests::run([](){
        MatrixXd param = MatrixXd::Zero(3, 1);
        param(1, 0) = 1;
        Categorical c = Categorical(param);

        REQUIRE( c.entropy() == 0 );
    });
}

TEST_CASE( "Categorical::entropy() returns the proper results" ) {
    UnitTests::run([](){
        MatrixXd param(3, 1);
        param << 0.7,
                0.2,
                0.1;
        Categorical c = Categorical(param);
        REQUIRE( c.entropy() == Approx(0.801819) );

        MatrixXd param1(3, 1);
        param1 << 0.5,
                0.2,
                0.3;
        Categorical c1 = Categorical(param1);
        REQUIRE( c1.entropy() == Approx(1.029653) );

        MatrixXd param2(3, 1);
        param2 << 0.3,
                0.3,
                0.4;
        Categorical c2 = Categorical(param2);
        REQUIRE( c2.entropy() == Approx(1.0889) );
    });
}

TEST_CASE( "Categorical parameters update and getter work properly" ) {
    UnitTests::run([](){
        MatrixXd param1(2, 1);
        param1 << 0.2,
                0.8;
        MatrixXd param2(2, 1);
        param2 << 0.3,
                0.7;
        Categorical d = Categorical(param1);
        auto param = d.params();
        REQUIRE( param.size() == 1 );
        REQUIRE( param[0](0,0) == 0.2 );
        REQUIRE( param[0](1,0) == 0.8 );
        std::vector<MatrixXd> p2{ param2 };
        d.updateParams(p2);
        auto res = Functions::softmax(param2);
        param = d.params();
        REQUIRE( param.size() == 1 );
        REQUIRE( param[0](0,0) == res(0,0) );
        REQUIRE( param[0](1,0) == res(1,0) );
    });
}

TEST_CASE( "Categorical::updateParams() throws an exception if the size of the new parameters is not one" ) {
    UnitTests::run([](){
        MatrixXd param1 = Functions::uniformColumnWise(3, 3);
        MatrixXd param2 = Functions::uniformColumnWise(3, 3);
        Categorical d = Categorical(param1);

        try {
            std::vector<MatrixXd> p{ param1, param2 };
            d.updateParams(p);
            REQUIRE(false);
        } catch (std::exception &e) {
            REQUIRE(true);
        }
    });
}
