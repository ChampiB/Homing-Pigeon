//
// Created by tmac3 on 02/12/2020.
//

#include "catch.hpp"
#include <torch/torch.h>
#include <iostream>
#include <helpers/UnitTests.h>
#include "distributions/Categorical.h"
#include "math/Ops.h"

using namespace torch;
using namespace hopi::distributions;
using namespace hopi::math;
using namespace tests;

TEST_CASE( "Categorical distribution returns the proper type" ) {
    UnitTests::run([](){
        Categorical c = Categorical(Ops::uniformColumnWise({3}));
        REQUIRE( c.type() == DistributionType::CATEGORICAL );
    });
}

TEST_CASE( "Categorical distribution returns the correct params" ) {
    UnitTests::run([](){
        Tensor param1 = torch::tensor({0.1,0.7,0.2});
        Categorical c1 = Categorical(param1);
        REQUIRE(torch::equal(c1.params(), param1));

        Tensor param2 = Ops::uniformColumnWise({2});
        Categorical c2 = Categorical(param2);
        REQUIRE(torch::equal(c2.params(), param2));
    });
}

TEST_CASE( "Categorical distribution returns the correct log params" ) {
    UnitTests::run([](){
        Tensor param1 = torch::tensor({0.1,0.7,0.2});
        Categorical c1 = Categorical(param1);
        REQUIRE(torch::equal(c1.logParams(), param1.log()));

        Tensor param2 = Ops::uniformColumnWise({2});
        Categorical c2 = Categorical(param2);
        REQUIRE(torch::equal(c2.logParams(), param2.log()));
    });
}

TEST_CASE( "Categorical::entropy() of [0 0 0 1 0] is zero" ) {
    UnitTests::run([](){
        Categorical c = Categorical(Ops::oneHot(3,1));

        REQUIRE( c.entropy() == 0 );
    });
}

TEST_CASE( "Categorical::entropy() returns the proper results" ) {
    UnitTests::run([](){
        Categorical c = Categorical(torch::tensor({0.7,0.2,0.1}));
        REQUIRE( c.entropy() == Approx(0.801819) );

        Categorical c1 = Categorical(torch::tensor({0.5,0.2,0.3}));
        REQUIRE( c1.entropy() == Approx(1.029653) );

        Categorical c2 = Categorical(torch::tensor({0.3,0.3,0.4}));
        REQUIRE( c2.entropy() == Approx(1.0889) );
    });
}

TEST_CASE( "Categorical parameters update and getter work properly" ) {
    UnitTests::run([](){
        Tensor param1 = torch::tensor({0.2,0.8});
        Categorical d = Categorical(param1);
        REQUIRE(torch::equal(d.params(), param1));

        Tensor param2 = torch::tensor({0.3,0.7});
        d.updateParams(param2.view({1,2,1}));
        REQUIRE(torch::equal(d.params(), torch::softmax(param2, 0)));
    });
}

TEST_CASE( "Categorical::updateParams() throws an exception if the size of the new parameters is not one" ) {
    UnitTests::run([](){
        Categorical d = Categorical(Ops::uniformColumnWise({3,3}));

        try {
            d.updateParams(Ops::uniformColumnWise({2,3,3}));
            REQUIRE(false);
        } catch (std::exception &e) {
            REQUIRE(true);
        }
    });
}
