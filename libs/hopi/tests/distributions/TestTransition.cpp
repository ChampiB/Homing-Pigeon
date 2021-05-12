//
// Created by tmac3 on 02/12/2020.
//

#include "catch.hpp"
#include <torch/torch.h>
#include <iostream>
#include <helpers/UnitTests.h>
#include "distributions/Transition.h"
#include "math/Ops.h"

using namespace torch;
using namespace hopi::distributions;
using namespace hopi::math;
using namespace tests;

TEST_CASE( "Transition distribution returns the proper type" ) {
    UnitTests::run([](){
        Transition t = Transition(Ops::uniformColumnWise({3}));
        REQUIRE( t.type() == DistributionType::TRANSITION );
    });
}

TEST_CASE( "Transition distribution returns the correct log params" ) {
    UnitTests::run([](){
        Tensor p1 = torch::tensor({0.1,0.3,0.7,0.4,0.2,0.3}).view({3,2});
        Transition d1 = Transition(p1);
        REQUIRE( torch::equal(d1.logParams(),p1.log()) );

        Tensor p2 = torch::tensor({0.5,0.2,0.9,0.5,0.8,0.1}).view({3,2});
        Transition d2 = Transition(p2);
        REQUIRE( torch::equal(d2.logParams(),p2.log()) );
    });
}

TEST_CASE( "Transition parameters update and getter work properly" ) {
    UnitTests::run([](){
        Tensor p1 = torch::tensor({0.5,0.2,0.5,0.8}).view({2,2});
        Transition d = Transition(p1);
        REQUIRE( torch::equal(d.params(), p1) );

        Tensor p2 = torch::tensor({0.3,0.05,0.7,0.95}).view({2,2});
        d.updateParams(p2);
        REQUIRE( torch::equal(d.params(), torch::softmax(p2, 0)) );
    });
}

TEST_CASE( "Transition::updateParams() throws an exception if the size of the new parameters is not one" ) {
    UnitTests::run([](){
        Transition d = Transition(Ops::uniformColumnWise({3,3}));

        try {
            d.updateParams(Ops::uniformColumnWise({2,3,3}));
            REQUIRE(false);
        } catch (std::exception &e) {
            REQUIRE(true);
        }
    });
}

TEST_CASE( "Transition::entropy() throws an exception (unsupported feature)" ) {
    UnitTests::run([](){
        Transition d = Transition(Ops::uniformColumnWise({3,3}));

        try {
            d.entropy();
            REQUIRE(false);
        } catch (std::exception &e) {
            REQUIRE(true);
        }
    });
}

