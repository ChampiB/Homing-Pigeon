//
// Created by tmac3 on 02/12/2020.
//

#include "catch.hpp"
#include <Eigen/Dense>
#include <iostream>
#include "distributions/Transition.h"
#include "math/Functions.h"

using namespace Eigen;
using namespace hopi::distributions;
using namespace hopi::math;

TEST_CASE( "Transition distribution returns the proper type" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param(3, 1);
    Transition t = Transition(param);
    REQUIRE( t.type() == DistributionType::TRANSITION );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Transition distribution returns the correct log params" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param1(3, 2);
    param1 << 0.1, 0.3,
              0.7, 0.4,
              0.2, 0.3;
    Transition d1 = Transition(param1);
    auto lp1 = d1.logParams();
    REQUIRE( lp1.size() == 1 );
    REQUIRE( lp1[0](0, 0) == std::log(0.1) );
    REQUIRE( lp1[0](1, 0) == std::log(0.7) );
    REQUIRE( lp1[0](2, 0) == std::log(0.2) );
    REQUIRE( lp1[0](0, 1) == std::log(0.3) );
    REQUIRE( lp1[0](1, 1) == std::log(0.4) );
    REQUIRE( lp1[0](2, 1) == std::log(0.3) );

    MatrixXd param2(2, 3);
    param2 << 0.5, 0.2, 0.9,
              0.5, 0.8, 0.1;
    Transition d2 = Transition(param2);
    auto lp2 = d2.logParams();
    REQUIRE( lp2.size() == 1 );
    REQUIRE( lp2[0](0,0) == std::log(0.5) );
    REQUIRE( lp2[0](1,0) == std::log(0.5) );
    REQUIRE( lp2[0](0,1) == std::log(0.2) );
    REQUIRE( lp2[0](1,1) == std::log(0.8) );
    REQUIRE( lp2[0](0,2) == std::log(0.9) );
    REQUIRE( lp2[0](1,2) == std::log(0.1) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Transition parameters update and getter work properly" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param1(2, 2);
    param1 << 0.5, 0.2,
              0.5, 0.8;
    MatrixXd param2(2, 2);
    param2 << 0.3, 0.05,
              0.7, 0.95;
    Transition d = Transition(param1);
    auto param = d.params();
    REQUIRE( param.size() == 1 );
    REQUIRE( param[0](0,0) == 0.5 );
    REQUIRE( param[0](1,0) == 0.5 );
    REQUIRE( param[0](0,1) == 0.2 );
    REQUIRE( param[0](1,1) == 0.8 );
    std::vector<MatrixXd> p2{ param2 };
    d.updateParams(p2);
    param = d.params();
    auto res = Functions::softmax(param2);
    REQUIRE( param[0](0,0) == res(0,0) );
    REQUIRE( param[0](1,0) == res(1,0) );
    REQUIRE( param[0](0,1) == res(0,1) );
    REQUIRE( param[0](1,1) == res(1,1) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Transition::updateParams() throws an exception if the size of the new parameters is not one" ) {
    std::cout << "Start: " << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param1 = MatrixXd::Constant(3, 3, 1.0 / 3);
    MatrixXd param2 = MatrixXd::Constant(3, 3, 1.0 / 3);
    Transition d = Transition(param1);

    try {
        std::vector<MatrixXd> p{ param1, param2 };
        d.updateParams(p);
        REQUIRE(false);
    } catch (std::exception &e) {
        REQUIRE(true);
    }
    std::cout << "End: " << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Transition::entropy() throws an exception (unsupported feature)" ) {
    std::cout << "Start: " << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param1 = MatrixXd::Constant(3, 3, 1.0 / 3);
    Transition d = Transition(param1);

    try {
        d.entropy();
        REQUIRE(false);
    } catch (std::exception &e) {
        REQUIRE(true);
    }
    std::cout << "End: " << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

