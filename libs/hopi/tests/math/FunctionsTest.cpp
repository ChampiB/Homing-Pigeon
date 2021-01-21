//
// Created by tmac3 on 02/12/2020.
//

#include "catch.hpp"
#include "contexts/FactorGraphContexts.h"
#include "math/Functions.h"
#include "distributions/Distribution.h"
#include "distributions/Categorical.h"
#include "distributions/Transition.h"
#include "distributions/ActiveTransition.h"
#include "graphs/FactorGraph.h"
#include "nodes/FactorNode.h"
#include "nodes/VarNode.h"
#include <Eigen/Dense>
#include <iostream>
#include <distributions/Dirichlet.h>

using namespace hopi::distributions;
using namespace hopi::nodes;
using namespace hopi::math;
using namespace Eigen;

TEST_CASE( "Softmax's output sum up to one" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd m1 = MatrixXd::Constant(4, 1, 42);
    m1 = Functions::softmax(m1);
    REQUIRE( m1.sum() == Approx(1).epsilon(0.1) );

    MatrixXd m2(3, 1);
    m2 << 1,
          11,
          111;
    m2 = Functions::softmax(m2);
    REQUIRE( m2.sum() == Approx(1).epsilon(0.1) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Softmax does not overflow" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd m1(2, 1);
    m1 << 1000000000000000000,
          500;
    m1 = Functions::softmax(m1);
    REQUIRE( m1.sum() == Approx(1).epsilon(0.1) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Softmax's output are correct" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd m1(1, 1);
    m1 << 3;
    m1 = Functions::softmax(m1);
    REQUIRE( m1(0, 0) == 1 );

    MatrixXd m2(3, 1);
    m2 << 3,
          4,
          1;
    m2 = Functions::softmax(m2);
    REQUIRE( m2(0, 0) == Approx(0.25949646034242).epsilon(0.1) );
    REQUIRE( m2(1, 0) == Approx(0.70538451269824).epsilon(0.1) );
    REQUIRE( m2(2, 0) == Approx(0.03511902695934).epsilon(0.1) );

    MatrixXd m3(2, 1);
    m3 << 500,
          500;
    m3 = Functions::softmax(m3);
    REQUIRE( m3(0, 0) == 0.5 );
    REQUIRE( m3(1, 0) == 0.5 );

    MatrixXd m4(5, 1);
    m4 << 3,
          4,
          1,
          10,
          2;
    m4 = Functions::softmax(m4);
    REQUIRE( m4(0, 0) == Approx(9.0838513102074E-4).epsilon(0.1) );
    REQUIRE( m4(1, 0) == Approx(0.0024692467948961).epsilon(0.1) );
    REQUIRE( m4(2, 0) == Approx(1.2293655899462E-4).epsilon(0.1) );
    REQUIRE( m4(3, 0) == Approx(0.99616525530072).epsilon(0.1) );
    REQUIRE( m4(4, 0) == Approx(3.3417621436836E-4).epsilon(0.1) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

// TODO what if p(X = x) = 0 for some x ?
TEST_CASE( "KL of identical (Categorical) distributions is zero." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param = MatrixXd::Constant(5, 1, 1.0 / 5);
    std::unique_ptr<Distribution> d1 = std::make_unique<Categorical>(param);
    std::unique_ptr<Distribution> d2 = std::make_unique<Categorical>(param);

    REQUIRE( Functions::KL(d1.get(), d2.get()) == 0 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "KL of identical (Dirichlet) distributions is zero." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param = MatrixXd::Constant(5, 1, 1.0 / 5);
    std::vector<MatrixXd> vec{param};
    std::unique_ptr<Distribution> d1 = std::make_unique<Dirichlet>(vec);
    std::unique_ptr<Distribution> d2 = std::make_unique<Dirichlet>(vec);

    std::cout << "1" << std::endl;
    std::cout << d1.get() << std::endl;
    std::cout << "2" << std::endl;
    std::cout << d2.get() << std::endl;
    std::cout << "3" << std::endl;
    REQUIRE( Functions::KL(d1.get(), d2.get()) == 0 );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "KL between Transition distributions is not supported." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param = MatrixXd::Constant(4, 4, 1.0 / 4);
    std::unique_ptr<Distribution> d1 = std::make_unique<Transition>(param);
    std::unique_ptr<Distribution> d2 = std::make_unique<Transition>(param);

    try {
        Functions::KL(d1.get(), d2.get());
        REQUIRE( false );
    } catch (const std::runtime_error& error) {
        REQUIRE( true );
    }
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "KL between ActiveTransition distributions is not supported." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param = MatrixXd::Constant(5, 4, 1.0 / 5);
    std::vector<MatrixXd> vec {param};
    std::unique_ptr<Distribution> d1 = std::make_unique<ActiveTransition>(vec);
    std::unique_ptr<Distribution> d2 = std::make_unique<ActiveTransition>(vec);

    try {
        Functions::KL(d1.get(), d2.get());
        REQUIRE( false );
    } catch (const std::runtime_error& error) {
        REQUIRE( true );
    }
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "KL of non identical (Categorical) distributions is not zero." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    // First distribution
    MatrixXd param1 = MatrixXd(2, 1);
    param1 << 0.5,
              0.5;
    std::unique_ptr<Distribution> d1 = std::make_unique<Categorical>(param1);
    // Second distribution
    MatrixXd param2 = MatrixXd(2, 1);
    param2 << 0.1,
              0.9;
    std::unique_ptr<Distribution> d2 = std::make_unique<Categorical>(param2);
    // True KL divergence
    double kl_result = 0.5 * (std::log(0.5) - std::log(0.1)) + 0.5 * (std::log(0.5) - std::log(0.9));

    REQUIRE( Functions::KL(d1.get(), d2.get()) == kl_result );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "KL of non identical (Dirichlet) distributions is not zero." ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    // First distribution
    MatrixXd param1 = MatrixXd(2, 1);
    param1 << 0.5,
              0.5;
    std::vector<MatrixXd> vec1 {param1};
    std::unique_ptr<Distribution> d1 = std::make_unique<Dirichlet>(vec1);
    // Second distribution
    MatrixXd param2 = MatrixXd(2, 1);
    param2 << 0.1,
              0.9;
    std::vector<MatrixXd> vec2 {param2};
    std::unique_ptr<Distribution> d2 = std::make_unique<Dirichlet>(vec2);

    REQUIRE( Functions::KL(d1.get(), d2.get()) == Approx(1.6803477088) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Beta function output correct values" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param1 = MatrixXd(2, 1);
    param1 << 1.5,
              0.2;
    REQUIRE( Functions::beta(param1) == Approx(4.477609374347168810412) );

    MatrixXd param2 = MatrixXd(2, 1);
    param2 << 2,
              2;
    REQUIRE( Functions::beta(param2) == Approx(0.1666666666666666666667) );

    MatrixXd param3 = MatrixXd(2, 1);
    param3 << 0.01,
              3.5;
    REQUIRE( Functions::beta(param3) == Approx(98.34009340030244331049) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Logarithm of the beta function are correctly computed" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    MatrixXd param1 = MatrixXd(2, 1);
    param1 << 1.5,
              0.2;
    REQUIRE( Functions::log_beta(param1) == Approx(std::log(Functions::beta(param1))) );

    MatrixXd param2 = MatrixXd(2, 1);
    param2 << 2,
              2;
    REQUIRE( Functions::log_beta(param2) == Approx(std::log(Functions::beta(param2))) );

    MatrixXd param3 = MatrixXd(2, 1);
    param3 << 0.01,
              3.5;
    REQUIRE( Functions::log_beta(param3) == Approx(std::log(Functions::beta(param3))) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}

TEST_CASE( "Digamma function output correct values" ) {
    std::cout << "Start: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
    REQUIRE( Functions::digamma(0.3458) == Approx(-3.0103) );
    REQUIRE( Functions::digamma(2) == Approx(0.4227).epsilon(0.1) );
    REQUIRE( Functions::digamma(1) == Approx(-0.5773).epsilon(0.1) );
    REQUIRE( Functions::digamma(100) == Approx(4.5952).epsilon(0.1) );
    std::cout << "End: "  << Catch::getResultCapture().getCurrentTestName() << std::endl;
}
