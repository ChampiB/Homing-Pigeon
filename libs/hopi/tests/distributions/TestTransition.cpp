//
// Created by tmac3 on 02/12/2020.
//

#include "catch.hpp"
#include <Eigen/Dense>
#include <iostream>
#include "distributions/Transition.h"

using namespace Eigen;
using namespace hopi::distributions;

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
    param1(0, 0) = 0.1;
    param1(1, 0) = 0.7;
    param1(2, 0) = 0.2;
    param1(0, 1) = 0.3;
    param1(1, 1) = 0.4;
    param1(2, 1) = 0.3;
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
    param2(0, 0) = 0.5;
    param2(1, 0) = 0.5;
    param2(0, 1) = 0.2;
    param2(1, 1) = 0.8;
    param2(0, 2) = 0.9;
    param2(1, 2) = 0.1;
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
