//
// Created by Theophile Champion on 02/12/2020.
//

#include "catch.hpp"
#include <torch/torch.h>
#include <iostream>
#include <helpers/UnitTests.h>
#include "distributions/Transition.h"
#include "math/Ops.h"
#include "api/API.h"

using namespace torch;
using namespace hopi::distributions;
using namespace hopi::math;
using namespace hopi::api;
using namespace tests;

TEST_CASE( "Transition distribution returns the proper type" ) {
    UnitTests::run([](){
        Transition t = Transition(Ops::uniform({3}));
        REQUIRE( t.type() == DistributionType::TRANSITION );
    });
}

TEST_CASE( "Transition distribution returns the correct log params" ) {
    UnitTests::run([](){
        Tensor p1 = API::tensor({0.1,0.3,0.7,0.4,0.2,0.3}).view({3,2});
        Transition d1 = Transition(p1);
        REQUIRE( torch::equal(d1.logParams(),p1.log()) );

        Tensor p2 = API::tensor({0.5,0.2,0.9,0.5,0.8,0.1}).view({3,2});
        Transition d2 = Transition(p2);
        REQUIRE( equal(d2.logParams(),p2.log()) );
    });
}
