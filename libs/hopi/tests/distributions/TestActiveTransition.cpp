//
// Created by tmac3 on 02/12/2020.
//

#include "catch.hpp"
#include <Eigen/Dense>
#include <iostream>
#include "distributions/ActiveTransition.h"

using namespace Eigen;
using namespace hopi::distributions;

TEST_CASE( "ActiveTransition distribution returns the proper type" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param(3, 1);
    std::vector<MatrixXd> p{param};
    ActiveTransition t = ActiveTransition(p);
    REQUIRE( t.type() == DistributionType::ACTIVE_TRANSITION );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "ActiveTransition distribution returns the correct log params" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param1(3, 2);
    param1(0, 0) = 0.1;
    param1(1, 0) = 0.7;
    param1(2, 0) = 0.2;
    param1(0, 1) = 0.3;
    param1(1, 1) = 0.4;
    param1(2, 1) = 0.3;
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
    param2_1(0, 0) = 0.5;
    param2_1(1, 0) = 0.5;
    param2_1(0, 1) = 0.2;
    param2_1(1, 1) = 0.8;
    MatrixXd param2_2(2, 2);
    param2_2(0, 0) = 0.3;
    param2_2(1, 0) = 0.7;
    param2_2(0, 1) = 0.05;
    param2_2(1, 1) = 0.95;
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
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}
