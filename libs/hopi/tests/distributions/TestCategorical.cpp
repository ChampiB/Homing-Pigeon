//
// Created by Theophile Champion on 02/12/2020.
//

#include "catch.hpp"
#include <torch/torch.h>
#include <iostream>
#include <helpers/UnitTests.h>
#include "distributions/Categorical.h"
#include "math/Ops.h"
#include "api/API.h"

using namespace torch;
using namespace hopi::distributions;
using namespace hopi::math;
using namespace hopi::api;
using namespace tests;

TEST_CASE( "Categorical distribution returns the proper type" ) {
    UnitTests::run([](){
        Categorical c = Categorical(Ops::uniform({3}));
        REQUIRE( c.type() == DistributionType::CATEGORICAL );
    });
}

TEST_CASE( "Categorical distribution returns the correct params" ) {
    UnitTests::run([](){
        Tensor param1 = API::tensor({0.1,0.7,0.2});
        Categorical c1 = Categorical(param1);
        REQUIRE(equal(c1.params(), param1));

        Tensor param2 = Ops::uniform({2});
        Categorical c2 = Categorical(param2);
        REQUIRE(equal(c2.params(), param2));
    });
}

TEST_CASE( "Categorical distribution returns the correct log params" ) {
    UnitTests::run([](){
        Tensor param1 = API::tensor({0.1,0.7,0.2});
        Categorical c1 = Categorical(param1);
        REQUIRE(equal(c1.logParams(), param1.log()));

        Tensor param2 = Ops::uniform({2});
        Categorical c2 = Categorical(param2);
        REQUIRE(equal(c2.logParams(), param2.log()));
    });
}

TEST_CASE( "Categorical::entropy() of [0 0 0 1 0] is zero" ) {
    UnitTests::run([](){
        Categorical c = Categorical(Ops::one_hot(3, 1));

        REQUIRE( c.entropy() == 0 );
    });
}

TEST_CASE( "Categorical::entropy() returns the proper results" ) {
    UnitTests::run([](){
        Categorical c = Categorical(API::tensor({0.7,0.2,0.1}));
        REQUIRE( c.entropy() == Approx(0.801819) );

        Categorical c1 = Categorical(API::tensor({0.5,0.2,0.3}));
        REQUIRE( c1.entropy() == Approx(1.029653) );

        Categorical c2 = Categorical(API::tensor({0.3,0.3,0.4}));
        REQUIRE( c2.entropy() == Approx(1.0889) );
    });
}

TEST_CASE( "Categorical parameters update and getter work properly" ) {
    UnitTests::run([](){
        Tensor param1 = API::tensor({0.2,0.8});
        Categorical d = Categorical(param1);
        REQUIRE(equal(d.params(), param1));

        Tensor param2 = API::tensor({0.3,0.7});
        d.updateParams(param2);
        REQUIRE(equal(d.params(), softmax(param2, 0)));
    });
}
