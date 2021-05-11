//
// Created by tmac3 on 02/12/2020.
//

#include "catch.hpp"
#include "math/Functions.h"
#include "distributions/Categorical.h"
#include "distributions/Transition.h"
#include "distributions/ActiveTransition.h"
#include "distributions/Dirichlet.h"
#include "nodes/VarNode.h"
#include <Eigen/Dense>
#include <iostream>
#include <helpers/UnitTests.h>

using namespace hopi::distributions;
using namespace hopi::nodes;
using namespace hopi::math;
using namespace Eigen;
using namespace tests;

TEST_CASE( "Softmax's output sum up to one" ) {
    UnitTests::run([](){
        MatrixXd m1 = MatrixXd::Constant(4, 1, 42);
        m1 = Functions::softmax(m1);
        REQUIRE( m1.sum() == Approx(1).epsilon(0.1) );

        MatrixXd m2(3, 1);
        m2 << 1,
                11,
                111;
        m2 = Functions::softmax(m2);
        REQUIRE( m2.sum() == Approx(1).epsilon(0.1) );
    });
}

TEST_CASE( "Softmax does not overflow" ) {
    UnitTests::run([](){
        MatrixXd m1(2, 1);
        m1 << 1000000000000000000,
                500;
        m1 = Functions::softmax(m1);
        REQUIRE( m1.sum() == Approx(1).epsilon(0.1) );
    });
}

TEST_CASE( "Softmax's output are correct" ) {
    UnitTests::run([](){
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
    });
}

// TODO what if p(X = x) = 0 for some x ?
TEST_CASE( "KL of identical (Categorical) distributions is zero." ) {
    UnitTests::run([](){
        MatrixXd param = MatrixXd::Constant(5, 1, 1.0 / 5);
        auto d1 = Categorical::create(param);
        auto d2 = Categorical::create(param);

        REQUIRE( Functions::KL(d1.get(), d2.get()) == 0 );
    });
}

TEST_CASE( "KL of identical (Dirichlet) distributions is zero." ) {
    UnitTests::run([](){
        MatrixXd param = Functions::uniformColumnWise(5, 1);
        std::vector<MatrixXd> vec{param};
        auto d1 = Dirichlet::create(vec);
        auto d2 = Dirichlet::create(vec);

        std::cout << d1.get() << std::endl;
        std::cout << d2.get() << std::endl;
        REQUIRE( Functions::KL(d1.get(), d2.get()) == 0 );
    });
}

TEST_CASE( "KL between Transition distributions is not supported." ) {
    UnitTests::run([](){
        MatrixXd param = Functions::uniformColumnWise(4, 4);
        auto d1 = Transition::create(param);
        auto d2 = Transition::create(param);

        try {
            Functions::KL(d1.get(), d2.get());
            REQUIRE( false );
        } catch (const std::runtime_error& error) {
            REQUIRE( true );
        }
    });
}

TEST_CASE( "KL between ActiveTransition distributions is not supported." ) {
    UnitTests::run([](){
        MatrixXd param = Functions::uniformColumnWise(5, 4);
        std::vector<MatrixXd> vec {param};
        auto d1 = ActiveTransition::create(vec);
        auto d2 = ActiveTransition::create(vec);

        try {
            Functions::KL(d1.get(), d2.get());
            REQUIRE( false );
        } catch (const std::runtime_error& error) {
            REQUIRE( true );
        }
    });
}

TEST_CASE( "KL of non identical (Categorical) distributions is not zero." ) {
    UnitTests::run([](){
        // First distribution
        MatrixXd param1 = Functions::uniformColumnWise(2, 1);
        auto d1 = Categorical::create(param1);
        // Second distribution
        MatrixXd param2(2, 1);
        param2 << 0.1,
                0.9;
        auto d2 = Categorical::create(param2);
        // True KL divergence
        double kl_result = 0.5 * (std::log(0.5) - std::log(0.1)) + 0.5 * (std::log(0.5) - std::log(0.9));

        REQUIRE( Functions::KL(d1.get(), d2.get()) == kl_result );
    });
}

TEST_CASE( "KL of non identical (Dirichlet) distributions is not zero." ) {
    UnitTests::run([](){
        // First distribution
        MatrixXd param1 = Functions::uniformColumnWise(2, 1);
        std::vector<MatrixXd> vec1 {param1};
        auto d1 = Dirichlet::create(vec1);
        // Second distribution
        MatrixXd param2(2, 1);
        param2 << 0.1,
                0.9;
        std::vector<MatrixXd> vec2 {param2};
        auto d2 = Dirichlet::create(vec2);

        REQUIRE( Functions::KL(d1.get(), d2.get()) == Approx(1.6803477088) );
    });
}

TEST_CASE( "Beta function output correct values" ) {
    UnitTests::run([](){
        MatrixXd param1(2, 1);
        param1 << 1.5,
                0.2;
        REQUIRE( Functions::beta(param1) == Approx(4.477609374347168810412) );

        MatrixXd param2(2, 1);
        param2 << 2,
                2;
        REQUIRE( Functions::beta(param2) == Approx(0.1666666666666666666667) );

        MatrixXd param3(2, 1);
        param3 << 0.01,
                3.5;
        REQUIRE( Functions::beta(param3) == Approx(98.34009340030244331049) );
    });
}

TEST_CASE( "Logarithm of the beta function are correctly computed" ) {
    UnitTests::run([](){
        MatrixXd param1(2, 1);
        param1 << 1.5,
                0.2;
        REQUIRE( Functions::log_beta(param1) == Approx(std::log(Functions::beta(param1))) );

        MatrixXd param2(2, 1);
        param2 << 2,
                2;
        REQUIRE( Functions::log_beta(param2) == Approx(std::log(Functions::beta(param2))) );

        MatrixXd param3(2, 1);
        param3 << 0.01,
                3.5;
        REQUIRE( Functions::log_beta(param3) == Approx(std::log(Functions::beta(param3))) );
    });
}

TEST_CASE( "Digamma function output correct values" ) {
    UnitTests::run([](){
        REQUIRE( Functions::digamma(0.3458) == Approx(-3.0103) );
        REQUIRE( Functions::digamma(2) == Approx(0.4227).epsilon(0.1) );
        REQUIRE( Functions::digamma(1) == Approx(-0.5773).epsilon(0.1) );
        REQUIRE( Functions::digamma(100) == Approx(4.5952).epsilon(0.1) );
    });
}

TEST_CASE( "OneHot returns correct one hot vectors" ) {
    UnitTests::run([](){
        MatrixXd m1 = Functions::oneHot(2, 1);
        REQUIRE( m1(0, 0) == 0 );
        REQUIRE( m1(1, 0) == 1 );

        MatrixXd m2 = Functions::oneHot(5, 0);
        REQUIRE( m2(0, 0) == 1 );
        REQUIRE( m2(1, 0) == 0 );
        REQUIRE( m2(2, 0) == 0 );
        REQUIRE( m2(3, 0) == 0 );
        REQUIRE( m2(4, 0) == 0 );

        MatrixXd m3 = Functions::oneHot(3, 2);
        REQUIRE( m3(0, 0) == 0 );
        REQUIRE( m3(1, 0) == 0 );
        REQUIRE( m3(2, 0) == 1 );
    });
}

TEST_CASE( "Function::constant(...) returns a vector containing constant matrices" ) {
    UnitTests::run([](){
        std::vector<MatrixXd> vec = Functions::constant(4, 3, 2, 42);
        REQUIRE( vec.size() == 4 );
        for (int i = 0; i < 4; ++i) {
            REQUIRE( vec[i].rows() == 3 );
            REQUIRE( vec[i].cols() == 2 );
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 2; ++k) {
                    REQUIRE( vec[i](j, k) == 42 );
                }
            }
        }
    });
}

TEST_CASE( "Function::uniformColumnWise(...) returns a matrix whose column sum to one with only positive elements" ) {
    UnitTests::run([](){
        MatrixXd mat = Functions::uniformColumnWise(3, 2);
        REQUIRE(mat.rows() == 3 );
        REQUIRE(mat.cols() == 2 );
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 2; ++k) {
                REQUIRE(mat(j, k) == Approx(0.3333333).epsilon(0.1) );
            }
        }

        mat = Functions::uniformColumnWise(5, 2);
        REQUIRE(mat.rows() == 5 );
        REQUIRE(mat.cols() == 2 );
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 2; ++k) {
                REQUIRE(mat(j, k) == 0.2 );
            }
        }
    });
}

TEST_CASE( "Function::uniformColumnWise(...) returns a vector of matrices representing a probability distribution" ) {
    UnitTests::run([](){
        std::vector<MatrixXd> vec = Functions::uniformColumnWise(2, 3, 2);
        REQUIRE(vec.size() == 2 );
        for (int i = 0; i < 2; ++i) {
            REQUIRE(vec[i].rows() == 3 );
            REQUIRE(vec[i].cols() == 2 );
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 2; ++k) {
                    REQUIRE(vec[i](j, k) == Approx(0.3333333).epsilon(0.1) );
                }
            }
        }

        vec = Functions::uniformColumnWise(5, 4, 2);
        REQUIRE(vec.size() == 5 );
        for (int i = 0; i < 5; ++i) {
            REQUIRE(vec[i].rows() == 4 );
            REQUIRE(vec[i].cols() == 2 );
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 2; ++k) {
                    REQUIRE(vec[i](j, k) == 0.25 );
                }
            }
        }
    });
}
