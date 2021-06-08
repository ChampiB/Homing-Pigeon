//
// Created by Theophile Champion on 02/12/2020.
//

#include <iostream>
#include <helpers/UnitTests.h>
#include "catch.hpp"
#include "iterators/AdjacentFactorsIter.h"
#include "contexts/FactorGraphContexts.h"
#include "graphs/FactorGraph.h"

using namespace tests;
using namespace hopi::iterators;

TEST_CASE( "AdjacentFactorsIter do not crash when no more factors are available" ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto it = AdjacentFactorsIter(fg->node(0));

        for (int i = 0; i < 10; ++i) {
            ++it;
        }
    });
}

TEST_CASE( "AdjacentFactorsIter return nullptr when no more factors are available (!=,==)" ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto it = AdjacentFactorsIter(fg->node(0));

        REQUIRE( *it != nullptr );
        ++it;
        REQUIRE( *it != nullptr );
        ++it;
        REQUIRE( *it == nullptr );
    });
}


TEST_CASE( "AdjacentFactorsIter properly support copy (=,==)" ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context2();
        auto it = AdjacentFactorsIter(fg->node(0));

        REQUIRE( *it == fg->factor(3) );
        ++it;
        auto it2 = it;
        REQUIRE( *it == fg->factor(0) );
        REQUIRE( *it2 == fg->factor(0) );

        ++it;
        REQUIRE( *it == nullptr );
        ++it2;
        REQUIRE( *it2 == nullptr );
    });
}
