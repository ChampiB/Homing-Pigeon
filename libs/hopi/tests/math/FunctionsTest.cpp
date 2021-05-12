//
// Created by tmac3 on 02/12/2020.
//

#include "catch.hpp"
#include "math/Ops.h"
#include "distributions/Categorical.h"
#include "distributions/Transition.h"
#include "distributions/ActiveTransition.h"
#include "distributions/Dirichlet.h"
#include <torch/torch.h>
#include <iostream>
#include <helpers/UnitTests.h>

using namespace hopi::distributions;
using namespace hopi::nodes;
using namespace hopi::math;
using namespace torch;
using namespace tests;

// TODO what if p(X = x) = 0 for some x ?
TEST_CASE( "KL of identical (Categorical) distributions is zero." ) {
    UnitTests::run([](){
        Tensor param = Ops::uniformColumnWise({5});
        auto d1 = Categorical::create(param);
        auto d2 = Categorical::create(param);

        REQUIRE( Ops::KL(d1.get(), d2.get()) == 0 );
    });
}

TEST_CASE( "KL of identical (Dirichlet) distributions is zero." ) {
    UnitTests::run([](){
        Tensor param = Ops::uniformColumnWise({2, 5, 2});
        auto d1 = Dirichlet::create(param);
        auto d2 = Dirichlet::create(param);

        REQUIRE( Ops::KL(d1.get(), d2.get()) == 0 );
    });
}

TEST_CASE( "KL between Transition distributions is not supported." ) {
    UnitTests::run([](){
        Tensor param = Ops::uniformColumnWise({4,4});
        auto d1 = Transition::create(param);
        auto d2 = Transition::create(param);

        try {
            Ops::KL(d1.get(), d2.get());
            REQUIRE( false );
        } catch (const std::runtime_error& error) {
            REQUIRE( true );
        }
    });
}

TEST_CASE( "KL between ActiveTransition distributions is not supported." ) {
    UnitTests::run([](){
        Tensor param = Ops::uniformColumnWise({2,5,4});
        auto d1 = ActiveTransition::create(param);
        auto d2 = ActiveTransition::create(param);

        try {
            Ops::KL(d1.get(), d2.get());
            REQUIRE( false );
        } catch (const std::runtime_error& error) {
            REQUIRE( true );
        }
    });
}

TEST_CASE( "KL of non identical (Categorical) distributions is not zero." ) {
    UnitTests::run([](){
        auto d1 = Categorical::create(Ops::uniformColumnWise({2}));
        auto d2 = Categorical::create(torch::tensor({0.1,0.9}));
        double kl_result = 0.5 * (std::log(0.5) - std::log(0.1)) + 0.5 * (std::log(0.5) - std::log(0.9));

        REQUIRE( Ops::KL(d1.get(), d2.get()) == kl_result );
    });
}

TEST_CASE( "KL of non identical (Dirichlet) distributions is not zero." ) {
    UnitTests::run([](){
        auto d1 = Dirichlet::create(Ops::uniformColumnWise({2}));
        auto d2 = Dirichlet::create(torch::tensor({0.1,0.9}).view({1,2,1}));

        REQUIRE( Ops::KL(d1.get(), d2.get()) == Approx(1.6803477088) );
    });
}

TEST_CASE( "Beta function output correct values" ) {
    UnitTests::run([](){
        REQUIRE( Ops::beta(torch::tensor({1.5,0.2}))  == Approx(4.477609374347168810412)  );
        REQUIRE( Ops::beta(torch::tensor({2,2}))      == Approx(0.1666666666666666666667) );
        REQUIRE( Ops::beta(torch::tensor({0.01,3.5})) == Approx(98.34009340030244331049)  );
    });
}

TEST_CASE( "Logarithm of the beta function are correctly computed" ) {
    UnitTests::run([](){
        Tensor param1 = torch::tensor({1.5,0.2});
        REQUIRE( Ops::log_beta(param1) == Approx(std::log(Ops::beta(param1))) );

        Tensor param2 = torch::tensor({2,2});
        REQUIRE( Ops::log_beta(param2) == Approx(std::log(Ops::beta(param2))) );

        Tensor param3 = torch::tensor({0.01,3.5});
        REQUIRE( Ops::log_beta(param3) == Approx(std::log(Ops::beta(param3))) );
    });
}

TEST_CASE( "Digamma function output correct values" ) {
    UnitTests::run([](){
        REQUIRE( Ops::digamma(0.3458) == Approx(-3.0103)              );
        REQUIRE( Ops::digamma(2)      == Approx(0.4227).epsilon(0.1)  );
        REQUIRE( Ops::digamma(1)      == Approx(-0.5773).epsilon(0.1) );
        REQUIRE( Ops::digamma(100)    == Approx(4.5952).epsilon(0.1)  );
    });
}

TEST_CASE( "OneHot returns correct one hot vectors" ) {
    UnitTests::run([](){
        Tensor m1 = Ops::oneHot(2, 1);
        REQUIRE( torch::equal(m1, torch::tensor({0,1})) );

        Tensor m2 = Ops::oneHot(5, 0);
        REQUIRE( torch::equal(m2, torch::tensor({1,0,0,0,0})) );

        Tensor m3 = Ops::oneHot(3, 2);
        REQUIRE( torch::equal(m3, torch::tensor({0,0,1})) );
    });
}

TEST_CASE( "Function::uniformColumnWise(...) returns a matrix whose column sum to one with only positive elements" ) {
    UnitTests::run([](){
        Tensor output = Ops::uniformColumnWise({3, 2});
        Tensor result = torch::full({3,2}, 0.3333333);
        UnitTests::require_approximately_equal(output, result, 0.1);

        output = Ops::uniformColumnWise({5, 2});
        result = torch::full({5,2}, 0.2);
        UnitTests::require_approximately_equal(output, result);
    });
}

TEST_CASE( "Function::uniformColumnWise(...) returns a vector of matrices representing a probability distribution" ) {
    UnitTests::run([](){
        Tensor output = Ops::uniformColumnWise({2, 3, 2});
        Tensor result = torch::full({2,3,2}, 0.3333333);
        UnitTests::require_approximately_equal(output, result, 0.1);

        output = Ops::uniformColumnWise({5, 4, 2});
        result = torch::full({5,4,2}, 0.25);
        UnitTests::require_approximately_equal(output, result);
    });
}
