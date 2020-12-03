//
// Created by Theophile Champion on 02/12/2020.
//

#include "catch.hpp"
#include <torch/torch.h>
#include <iostream>
#include "distributions/ActiveTransition.h"
#include "math/Ops.h"
#include "helpers/UnitTests.h"
#include "api/API.h"

using namespace torch;
using namespace torch::indexing;
using namespace hopi::distributions;
using namespace hopi::math;
using namespace hopi::api;
using namespace tests;

TEST_CASE( "ActiveTransition distribution returns the proper type" ) {
    UnitTests::run([](){
        ActiveTransition t = ActiveTransition(Ops::uniform({2,2,3}));
        REQUIRE( t.type() == DistributionType::ACTIVE_TRANSITION );
    });
}

TEST_CASE( "ActiveTransition distribution returns the correct log params" ) {
    UnitTests::run([](){
        Tensor p1 = API::tensor({0.1, 0.3, 0.7, 0.4, 0.2, 0.3}).view({1, 3, 2});
        ActiveTransition d1 = ActiveTransition(p1);
        REQUIRE( equal(d1.logParams(), p1.log()) );

        Tensor p2 = API::tensor({0.5,0.2,0.5,0.8,0.3,0.05,0.7,0.95}).view({2, 2, 2});
        ActiveTransition d2 = ActiveTransition(p2);
        REQUIRE( equal(d2.logParams(), p2.log()) );
    });
}
