//
// Created by tmac3 on 02/12/2020.
//

#include <iostream>
#include <helpers/UnitTests.h>
#include "catch.hpp"
#include "iterators/HiddenVarIter.h"
#include "contexts/FactorGraphContexts.h"
#include "graphs/FactorGraph.h"

using namespace hopi::iterators;
using namespace tests;

TEST_CASE( "HiddenVarIter do not crash when no more hidden variables are available" ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context1();
        auto it = HiddenVarIter(fg->getNodes());

        for (int i = 0; i < 10; ++i) {
            ++it;
        }
    });
}

TEST_CASE( "HiddenVarIter return nullptr when no more hidden variables are available" ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context1();
        auto it = HiddenVarIter(fg->getNodes());

        REQUIRE( *it != nullptr );
        ++it;
        REQUIRE( *it != nullptr );
        ++it;
        REQUIRE( *it != nullptr );
        ++it;
        REQUIRE( *it == nullptr );
    });
}

TEST_CASE( "HiddenVarIter return the correct hidden variable (==)" ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context1();
        auto it = HiddenVarIter(fg->getNodes());

        REQUIRE( *it == fg->node(1) );
        ++it;
        REQUIRE( *it == fg->node(4) );
        ++it;
        REQUIRE( *it == fg->node(5) );
    });
}

TEST_CASE( "HiddenVarIter return the correct hidden variable (!=)" ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context1();
        auto it = HiddenVarIter(fg->getNodes());

        REQUIRE( *it != fg->node(4) );
        REQUIRE( *it != fg->node(5) );
        ++it;
        REQUIRE( *it != fg->node(1) );
        REQUIRE( *it != fg->node(5) );
        ++it;
        REQUIRE( *it != fg->node(1) );
        REQUIRE( *it != fg->node(4) );
    });
}

TEST_CASE( "HiddenVarIter properly support copy (=)" ) {
    UnitTests::run([](){
        auto fg = FactorGraphContexts::context1();
        auto it = HiddenVarIter(fg->getNodes());

        REQUIRE( *it == fg->node(1) );
        ++it;
        auto it2 = it;
        REQUIRE( *it == fg->node(4) );
        REQUIRE( *it2 == fg->node(4) );

        ++it;
        REQUIRE( *it == fg->node(5) );
        ++it2;
        REQUIRE( *it2 == fg->node(5) );

        ++it;
        REQUIRE( *it == nullptr );
        ++it2;
        REQUIRE( *it2 == nullptr );
    });
}
