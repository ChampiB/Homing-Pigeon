//
// Created by tmac3 on 02/12/2020.
//

#include "catch.hpp"
#include <Eigen/Dense>
#include <iostream>
#include "distributions/Categorical.h"

using namespace Eigen;
using namespace hopi::distributions;

TEST_CASE( "Categorical distribution returns correct cardinality" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param1(3, 1);
    Categorical c1 = Categorical(param1);
    REQUIRE( c1.cardinality() == 3 );

    MatrixXd param2(5, 1);
    Categorical c2 = Categorical(param2);
    REQUIRE( c2.cardinality() == 5 );

    MatrixXd param3(10, 1);
    Categorical c3 = Categorical(param3);
    REQUIRE( c3.cardinality() == 10 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Categorical distribution returns the proper type" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param(3, 1);
    Categorical c = Categorical(param);
    REQUIRE( c.type() == DistributionType::CATEGORICAL );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Categorical distribution returns the correct probability" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param1(3, 1);
    param1(0, 0) = 0.1;
    param1(1, 0) = 0.7;
    param1(2, 0) = 0.2;
    Categorical c1 = Categorical(param1);
    REQUIRE( c1.p(0) == 0.1 );
    REQUIRE( c1.p(1) == 0.7 );
    REQUIRE( c1.p(2) == 0.2 );

    MatrixXd param2(2, 1);
    param2(0, 0) = 0.5;
    param2(1, 0) = 0.5;
    Categorical c2 = Categorical(param2);
    REQUIRE( c2.p(0) == 0.5 );
    REQUIRE( c2.p(1) == 0.5 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Categorical distribution returns the correct log probability" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param1(3, 1);
    param1(0, 0) = 0.1;
    param1(1, 0) = 0.7;
    param1(2, 0) = 0.2;
    Categorical c1 = Categorical(param1);
    auto lp1 = c1.logProbability();
    REQUIRE( lp1.size() == 1 );
    REQUIRE( lp1[0](0, 0) == std::log(0.1) );
    REQUIRE( lp1[0](1, 0) == std::log(0.7) );
    REQUIRE( lp1[0](2, 0) == std::log(0.2) );

    MatrixXd param2(2, 1);
    param2(0, 0) = 0.5;
    param2(1, 0) = 0.5;
    Categorical c2 = Categorical(param2);
    auto lp2 = c2.logProbability();
    REQUIRE( lp2.size() == 1 );
    REQUIRE( lp2[0](0, 0) == std::log(0.5) );
    REQUIRE( lp2[0](1, 0) == std::log(0.5) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}
