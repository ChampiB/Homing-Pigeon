//
// Created by tmac3 on 02/12/2020.
//

#include "catch.hpp"
#include <torch/torch.h>
#include <iostream>
#include "distributions/ActiveTransition.h"
#include "math/Ops.h"
#include "helpers/UnitTests.h"

using namespace torch;
using namespace torch::indexing;
using namespace hopi::distributions;
using namespace hopi::math;
using namespace tests;

TEST_CASE( "ActiveTransition distribution returns the proper type" ) {
    UnitTests::run([](){
        ActiveTransition t = ActiveTransition(Ops::uniformColumnWise({2, 3, 2}));
        REQUIRE( t.type() == DistributionType::ACTIVE_TRANSITION );
    });
}

TEST_CASE( "ActiveTransition distribution returns the correct log params" ) {
    UnitTests::run([](){
        Tensor p1 = torch::tensor({0.1, 0.3, 0.7, 0.4, 0.2, 0.3}).view({1, 3, 2});
        ActiveTransition d1 = ActiveTransition(p1);
        REQUIRE( torch::equal(d1.logParams(), p1.log()) );

        Tensor p2 = torch::tensor({0.5,0.2,0.5,0.8,0.3,0.05,0.7,0.95}).view({2, 3, 2});
        ActiveTransition d2 = ActiveTransition(p2);
        REQUIRE( torch::equal(d2.logParams(), p2.log()) );
    });
}

TEST_CASE( "ActiveTransition parameters update and getter work properly" ) {
    UnitTests::run([](){
        // Test retrieving parameters
        Tensor p = torch::tensor({0.5,0.2,0.5,0.8,0.3,0.05,0.7,0.95}).view({2,2,2});
        ActiveTransition d = ActiveTransition(p);
        REQUIRE( torch::equal(d.params(), p) );

        // Test retrieving parameters after update
        Tensor p2 = torch::tensor({0.5,0.2,0.5,0.8,0.15,0.04,0.85,0.96}).view({2,2,2});
        d.updateParams(p2);
        auto output = d.params();
        REQUIRE(output.size(0) == 2 );
        REQUIRE( torch::equal(output.index({0,None,None}), torch::softmax(p2.index({0,None,None}), 0)));
        REQUIRE( torch::equal(output.index({1,None,None}), torch::softmax(p2.index({1,None,None}), 0)));
    });
}

TEST_CASE( "ActiveTransition::updateParams() throws an exception if the sizes of the new and old parameters does not match" ) {
    UnitTests::run([](){
        ActiveTransition d = ActiveTransition(Ops::uniformColumnWise({2,3,3}));

        try {
            d.updateParams(Ops::uniformColumnWise({3,3}));
            REQUIRE(false);
        } catch (std::exception &e) {
            REQUIRE(true);
        }
    });
}

TEST_CASE( "ActiveTransition::entropy() throws an exception (unsupported feature)" ) {
    UnitTests::run([](){
        ActiveTransition d = ActiveTransition(Ops::uniformColumnWise({2,3,3}));

        try {
            d.entropy();
            REQUIRE(false);
        } catch (std::exception &e) {
            REQUIRE(true);
        }
    });
}
