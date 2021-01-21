//
// Created by tmac3 on 02/12/2020.
//

#include "catch.hpp"
#include <Eigen/Dense>
#include <iostream>
#include "distributions/Dirichlet.h"
#include "math/Functions.h"

using namespace Eigen;
using namespace hopi::distributions;
using namespace hopi::math;

TEST_CASE( "Dirichlet distribution returns the proper type" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param(3, 1);
    std::vector p {param};
    Dirichlet c = Dirichlet(p);
    REQUIRE( c.type() == DistributionType::DIRICHLET );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Dirichlet distribution returns the correct log params" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param1(3, 1);
    param1 << 0.1,
              0.7,
              0.2;
    std::vector<MatrixXd> p1{param1};
    Dirichlet c1 = Dirichlet(p1);
    auto lp1 = c1.logParams();
    REQUIRE( lp1.size() == 1 );
    REQUIRE( lp1[0](0, 0) == std::log(0.1) );
    REQUIRE( lp1[0](1, 0) == std::log(0.7) );
    REQUIRE( lp1[0](2, 0) == std::log(0.2) );

    MatrixXd param2(2, 1);
    param2 << 0.5,
              0.5;
    std::vector<MatrixXd> p2{param2};
    Dirichlet c2 = Dirichlet(p2);
    auto lp2 = c2.logParams();
    REQUIRE( lp2.size() == 1 );
    REQUIRE( lp2[0](0, 0) == std::log(0.5) );
    REQUIRE( lp2[0](1, 0) == std::log(0.5) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Dirichlet::entropy() returns the proper results (1D)" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param(3, 1);
    param << 0.7,
             0.2,
             0.1;
    std::vector<MatrixXd> p{param};
    Dirichlet c = Dirichlet(p);
    REQUIRE( c.entropy() == Approx(-7.8079249494) );

    MatrixXd param1(3, 1);
    param1 << 0.5,
              0.2,
              0.3;
    std::vector<MatrixXd> p1{param1};
    Dirichlet c1 = Dirichlet(p1);
    REQUIRE( c1.entropy() == Approx(-1.65334191) );

    MatrixXd param2(3, 1);
    param2 << 0.3,
              0.3,
              0.4;
    std::vector<MatrixXd> p2{param2};
    Dirichlet c2 = Dirichlet(p2);
    REQUIRE( c2.entropy() == Approx(-0.8572948628) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Dirichlet::entropy() returns the proper results (2D)" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param(3, 2);
    param << 0.7, 1,
             0.2, 2,
             0.1, 3;
    std::vector<MatrixXd> p{param};
    Dirichlet c = Dirichlet(p);
    REQUIRE( c.entropy() == Approx(-7.7839165065) );

    MatrixXd param1(3, 2);
    param1 << 0.5, 3,
              0.2, 10,
              0.3, 10;
    std::vector<MatrixXd> p1{param1};
    Dirichlet c1 = Dirichlet(p1);
    REQUIRE( c1.entropy() == Approx(-2.5557694149) );

    MatrixXd param2(3, 2);
    param2 << 0.3, 100,
              0.3, 100,
              0.4, 100;
    std::vector<MatrixXd> p2{param2};
    Dirichlet c2 = Dirichlet(p2);
    REQUIRE( c2.entropy() == Approx(-4.1286262513) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Dirichlet::entropy() returns the proper results (3D)" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param0(3, 2);
    param0 << 0.7, 1,
              0.2, 2,
              0.1, 3;
    MatrixXd param1(3, 2);
    param1 << 0.7, 1,
              0.2, 2,
              0.1, 3;
    std::vector<MatrixXd> p{param0, param1};
    Dirichlet c = Dirichlet(p);
    REQUIRE( c.entropy() == Approx(-15.5678330129) );

    MatrixXd param2(3, 2);
    param2 << 0.5, 3,
              0.2, 10,
              0.3, 10;
    MatrixXd param3(3, 2);
    param3 << 0.5, 3,
              0.2, 10,
              0.3, 10;
    std::vector<MatrixXd> p1{param2,param3};
    Dirichlet c1 = Dirichlet(p1);
    REQUIRE( c1.entropy() == Approx(-5.1115388297) );

    MatrixXd param4(3, 2);
    param4 << 0.3, 100,
              0.3, 100,
              0.4, 100;
    MatrixXd param5(3, 2);
    param5 << 0.3, 100,
              0.3, 100,
              0.4, 100;
    std::vector<MatrixXd> p2{param2};
    Dirichlet c2 = Dirichlet(p2);
    REQUIRE( c2.entropy() == Approx(-2.5557694149) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Dirichlet::entropy() returns the proper results (static)" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param(3, 1);
    param << 0.7,
             0.2,
             0.1;
    REQUIRE( Dirichlet::entropy(param) == Approx(-7.8079249494) );

    MatrixXd param1(3, 1);
    param1 << 0.5,
              0.2,
              0.3;
    std::vector<MatrixXd> p1{param1};
    REQUIRE( Dirichlet::entropy(param1) == Approx(-1.65334191) );

    MatrixXd param2(3, 1);
    param2 << 0.3,
              0.3,
              0.4;
    std::vector<MatrixXd> p2{param2};
    REQUIRE( Dirichlet::entropy(param2) == Approx(-0.8572948628) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Dirichlet parameters setter and getter work properly" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param1(2, 1);
    param1 << 0.2,
              0.8;
    std::vector<MatrixXd> p1{ param1 };
    Dirichlet d = Dirichlet(p1);
    auto param = d.params();
    REQUIRE( param.size() == 1 );
    REQUIRE( param[0](0,0) == 0.2 );
    REQUIRE( param[0](1,0) == 0.8 );

    MatrixXd param2(2, 1);
    param2 << 0.3,
              0.7;
    std::vector<MatrixXd> p2{ param2 };
    d.updateParams(p2);
    param = d.params();
    REQUIRE( param.size() == 1 );
    REQUIRE( param[0](0,0) == 0.3 );
    REQUIRE( param[0](1,0) == 0.7 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Dirichlet::expectedLog() returns the proper results (1D)" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param(3, 1);
    param << 0.7,
            0.2,
            0.1;
    std::vector<MatrixXd> p{param};
    auto predicted = Dirichlet::expectedLog(p);
    REQUIRE( predicted[0](0,0) == Approx(Functions::digamma(0.7) - Functions::digamma(1)) );
    REQUIRE( predicted[0](1,0) == Approx(Functions::digamma(0.2) - Functions::digamma(1)) );
    REQUIRE( predicted[0](2,0) == Approx(Functions::digamma(0.1) - Functions::digamma(1)) );

    MatrixXd param1(3, 1);
    param1 << 5,
            2,
            3;
    std::vector<MatrixXd> p1{param1};
    predicted = Dirichlet::expectedLog(p1);
    REQUIRE( predicted[0](0,0) == Approx(Functions::digamma(5) - Functions::digamma(10)) );
    REQUIRE( predicted[0](1,0) == Approx(Functions::digamma(2) - Functions::digamma(10)) );
    REQUIRE( predicted[0](2,0) == Approx(Functions::digamma(3) - Functions::digamma(10)) );

    MatrixXd param2(3, 1);
    param2 << 30,
            30,
            40;
    std::vector<MatrixXd> p2{param2};
    predicted = Dirichlet::expectedLog(p2);
    REQUIRE( predicted[0](0,0) == Approx(Functions::digamma(30) - Functions::digamma(100)) );
    REQUIRE( predicted[0](1,0) == Approx(Functions::digamma(30) - Functions::digamma(100)) );
    REQUIRE( predicted[0](2,0) == Approx(Functions::digamma(40) - Functions::digamma(100)) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Dirichlet::expectedLog() returns the proper results (2D)" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param(3, 2);
    param << 0.7, 1,
             0.2, 2,
             0.1, 3;
    std::vector<MatrixXd> p{param};
    auto predicted = Dirichlet::expectedLog(p);
    REQUIRE( predicted[0](0,0) == Approx(Functions::digamma(0.7) - Functions::digamma(1)) );
    REQUIRE( predicted[0](1,0) == Approx(Functions::digamma(0.2) - Functions::digamma(1)) );
    REQUIRE( predicted[0](2,0) == Approx(Functions::digamma(0.1) - Functions::digamma(1)) );
    REQUIRE( predicted[0](0,1) == Approx(Functions::digamma(1)   - Functions::digamma(6)) );
    REQUIRE( predicted[0](1,1) == Approx(Functions::digamma(2)   - Functions::digamma(6)) );
    REQUIRE( predicted[0](2,1) == Approx(Functions::digamma(3)   - Functions::digamma(6)) );

    MatrixXd param1(3, 2);
    param1 << 5, 10,
              2, 2,
              3, 12;
    std::vector<MatrixXd> p1{param1};
    predicted = Dirichlet::expectedLog(p1);
    REQUIRE( predicted[0](0,0) == Approx(Functions::digamma(5)  - Functions::digamma(10)) );
    REQUIRE( predicted[0](1,0) == Approx(Functions::digamma(2)  - Functions::digamma(10)) );
    REQUIRE( predicted[0](2,0) == Approx(Functions::digamma(3)  - Functions::digamma(10)) );
    REQUIRE( predicted[0](0,1) == Approx(Functions::digamma(10) - Functions::digamma(24)) );
    REQUIRE( predicted[0](1,1) == Approx(Functions::digamma(2)  - Functions::digamma(24)) );
    REQUIRE( predicted[0](2,1) == Approx(Functions::digamma(12) - Functions::digamma(24)) );

    MatrixXd param2(3, 2);
    param2 << 30, 1,
              30, 1,
              40, 1;
    std::vector<MatrixXd> p2{param2};
    predicted = Dirichlet::expectedLog(p2);
    REQUIRE( predicted[0](0,0) == Approx(Functions::digamma(30) - Functions::digamma(100)) );
    REQUIRE( predicted[0](1,0) == Approx(Functions::digamma(30) - Functions::digamma(100)) );
    REQUIRE( predicted[0](2,0) == Approx(Functions::digamma(40) - Functions::digamma(100)) );
    REQUIRE( predicted[0](0,1) == Approx(Functions::digamma(1)  - Functions::digamma(3)) );
    REQUIRE( predicted[0](1,1) == Approx(Functions::digamma(1)  - Functions::digamma(3)) );
    REQUIRE( predicted[0](2,1) == Approx(Functions::digamma(1)  - Functions::digamma(3)) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Dirichlet::expectedLog() returns the proper results (3D)" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param0(3, 2);
    param0 << 0.7, 1,
              0.2, 2,
              0.1, 3;
    MatrixXd param1(3, 2);
    param1 << 5, 10,
              2, 2,
              3, 12;
    std::vector<MatrixXd> p0{param0, param1};
    auto predicted = Dirichlet::expectedLog(p0);

    REQUIRE( predicted[0](0,0) == Approx(Functions::digamma(0.7) - Functions::digamma(1)) );
    REQUIRE( predicted[0](1,0) == Approx(Functions::digamma(0.2) - Functions::digamma(1)) );
    REQUIRE( predicted[0](2,0) == Approx(Functions::digamma(0.1) - Functions::digamma(1)) );
    REQUIRE( predicted[0](0,1) == Approx(Functions::digamma(1)   - Functions::digamma(6)) );
    REQUIRE( predicted[0](1,1) == Approx(Functions::digamma(2)   - Functions::digamma(6)) );
    REQUIRE( predicted[0](2,1) == Approx(Functions::digamma(3)   - Functions::digamma(6)) );
    REQUIRE( predicted[1](0,0) == Approx(Functions::digamma(5)   - Functions::digamma(10)) );
    REQUIRE( predicted[1](1,0) == Approx(Functions::digamma(2)   - Functions::digamma(10)) );
    REQUIRE( predicted[1](2,0) == Approx(Functions::digamma(3)   - Functions::digamma(10)) );
    REQUIRE( predicted[1](0,1) == Approx(Functions::digamma(10)  - Functions::digamma(24)) );
    REQUIRE( predicted[1](1,1) == Approx(Functions::digamma(2)   - Functions::digamma(24)) );
    REQUIRE( predicted[1](2,1) == Approx(Functions::digamma(12)  - Functions::digamma(24)) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}
